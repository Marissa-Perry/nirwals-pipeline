# ---------------------------------------------------------------------------- #
'''
SALT NIRWALS science pipeline:
- nirwalsreduce controls the science reduction of NIRWALS exposures
'''
# ---------------------------------------------------------------------------- #

# Standard library imports
import os
import sys
import time
import glob
import copy
from datetime import datetime

# numpy import
import numpy as np
# scipy imports
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
# astropy imports
from astropy.io import fits
from astropy.stats import sigma_clip, mad_std
# matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid

# Local application imports
# - saltutility.keys
from .....saltutility.keys import key_values
from .....saltutility.keys import key_values_id
# - saltutility.dirs
from .....saltutility.dirs import make_directory
# - saltutility.files
from .....saltutility.files import load_log_file
from .....saltutility.files import load_json_file
from .....saltutility.files import dump_json_file
# - saltutility.logging
from .....saltutility.logging import logging

# Application imports
# - functions
from ...functions import Fit1D
from ...functions import air_to_vac
from ...functions import get_evenly_spaced_array
# - spectroscopy.ifu
from ...spectroscopy.ifu import extract_fibres
# - spectroscopy.spectrograph
from ...spectroscopy.spectrograph import set_spectrograph
# - spectroscopy.spectrum
from ...spectroscopy.spectrum import artificial_spectrum
# - spectroscopy.wavelength (wavelength calibration utilities)
from ...spectroscopy.wavelength import fit_zero_point
# - spectroscopy.wavelength (wavelength calibration utilities)
from ...spectroscopy.wavelength import calculate_wavelength_fit
# - spectroscopy.wavelength.WavelengthFit
from ...spectroscopy.wavelength.WavelengthFit import WavelengthFit

# ---------------------------------------------------------------------------- #

MYNAME = 'nirwalsreduce'

# ---------------------------------------------------------------------------- #

# Primary fits extension
PRIMARY = 'Primary'
# Science fits extension
SCI = 'SCI'
# Science error fits extension
ERR = 'ERR'
# Bad pixel map fits extension
BPM = 'BPM'
# Continuum Fit fits extension
FIT = 'FIT'
# Good pixel count fits extension
GPCNT = 'GPCNT'
# Observation log file
OBSLOG = '{0}{1}OBSLOG.fits'
# Observation date format for SALT data archive directory
OBSDATE = '{0}/{1}'
# Wavelength decimal precision
DEC = 8
# Default kernel to convolve arrays
KERN = [0, -1, -2, -3, -2, -1, 0, 1, 2, 3, 2, 1, 0]

# ---------------------------------------------------------------------------- #
def reduce_data(obs_date, log_file, **kwargs):
# ---------------------------------------------------------------------------- #

    '''
    Control the science reduction of NIR exposures. The processing steps are
    determined from the relevant parameters.

    obs_date    : Observation date [CCYYMMDD] <str>
    log_file    : Log file <str>
    kwargs      : Various keyword arguments <dict>
    '''

    # Set current module object
    me = sys.modules[__name__]

    # Check if 'params' is not a dictionary
    if not isinstance(kwargs['params'], dict):
        # Set parameter file name
        params_file = os.path.join(kwargs['param_dir'], kwargs['params'])
        # Load parameters (JSON input file)
        params = load_json_file(params_file)
        # Set parameters in keyword arguments
        kwargs['params'] = params

    # Set SALT data archive directory with observation date format
    saltdata_dir = os.path.join(kwargs['saltdata_dir'], OBSDATE)
    # Set date directory
    date_dir = kwargs['work_dir']
    # Set data work directory
    data_dir = kwargs['params']['data_dir']
    # Set main work (output) directory
    wrk_dir = os.path.join(data_dir, kwargs['params']['wrk_dir'])
    # Set output sub directories from keyword arguments
    sub_dirs = kwargs['params']['sub_dirs']
    # Set product data work directory
    prd_dir = os.path.join(data_dir, 'product')
    # Set config directory from keyword arguments
    config_dir = kwargs['config_dir']

    # Set with stdout indicator
    with_ = kwargs['with_stdout']
    # Set only stdout indicator
    only_ = kwargs['only_stdout']

    # Start log
    with logging(log_file, with_stdout=with_, only_stdout=only_) as log:

        # Check if product directory for observation date doesn't exist
        if not os.path.isdir(prd_dir):
            # Add message to log
            msg = '{0} -- No data'.format(MYNAME.upper())
            log.message(msg)
            # Get out!
            return

        # Set observation log file name
        obslog_name = OBSLOG.format(kwargs['params']['raw_prefix'], obs_date)
        # Add product directory to observation log file name
        obslog_file = os.path.join(prd_dir, obslog_name)
        # Check observation log file
        if not os.path.isfile(obslog_file):
            # Add message to log
            msg = '{0} -- No observation log\n'.format(MYNAME.upper())
            log.message(msg)
            # No observation log file... Get out!
            return
        # Load observation log of instrument files
        obslog = load_log_file(obslog_file, 'OBSLOG')
        # Get all files, observation modes, detector modes and exposure types
        all_files = obslog.field('FILENAME')
        exp_types = obslog.field('OBSTYPE')

        # Loop for task action(s)...
        for action in kwargs['actions']:

            # Check action status
            if action['status'] != 1:
                # Move on!
                continue

            # Check if action has specific parameters
            if action['params']:
                # Add action specific parameters to keyword arguments
                kwargs['params'] = {**kwargs['params'], **action['params']}

            # Set reduce config file name
            config_file = kwargs['params']['config']
            # Add config directory config file name
            config_file = os.path.join(config_dir, config_file)
            # Load reduce (work) config (JSON input file)
            config_r = load_json_file(config_file)

            # Set exposure type associated with this action
            exp_type = config_r[action['name']]['exp_type']
            # Set mask for exposure type
            e_mask = exp_types==exp_type
            # Get raw file names for file mask
            raw_names = all_files[e_mask]

            # If raw files exist...
            if raw_names.size > 0:
                # Set product file prefix
                prefix = kwargs['params']['prd_prefix']
                # Add product file prefix to file names
                prd_names = ['{0}{1}'.format(prefix, f) for f in raw_names]
                # Add product data work directory path to product file names
                prd_files = [os.path.join(prd_dir, f) for f in prd_names]

                # Set observation date in reduce config
                config_r['obs_date'] = obs_date
                # Set exposure type in reduce config
                config_r['exp_type'] = exp_type.lower()
                # Set recalc indicator in reduce config
                config_r['recalc'] = kwargs['params']['recalc']
                # Set raw prefix in reduce config
                config_r['raw_prefix'] = kwargs['params']['raw_prefix']

                # Set SALT data archive directory in reduce config
                config_r['saltdata_dir'] = saltdata_dir
                # Set config directory in reduce config
                config_r['config_dir'] = config_dir
                # Set date directory in reduce config
                config_r['date_dir'] = date_dir
                # Set product directory (with date directory) in reduce config
                config_r['prd_dir'] = prd_dir
                # Set work directory in reduce config
                config_r['wrk_dir'] = wrk_dir

                # Set db fibre traces config file
                config_file = os.path.join(config_dir, config_r['fibre_traces_db'])
                # Update db fibre traces config file in reduce config
                config_r['fibre_traces_db'] = config_file
                # Set work fibre traces config file
                config_file = os.path.join(wrk_dir, config_r['fibre_traces_wrk'])
                # Update work fibre traces config file in reduce config
                config_r['fibre_traces_wrk'] = config_file

                # Set spectrograph config file
                config_file = os.path.join(config_dir, config_r['spectrograph'])
                # Set spectrograph config file in reduce config
                config_r['spectrograph'] = config_file
                # Set db wavelength solutions config file
                config_file = os.path.join(config_dir, config_r['solutions_db'])
                # Update db wavelength solutions file in reduce config
                config_r['solutions_db'] = config_file
                # Set work wavelength solutions config file
                config_file = os.path.join(wrk_dir, config_r['solutions_wrk'])
                # Update work wavelength solutions file in reduce config
                config_r['solutions_wrk'] = config_file
                # Set work exposures file
                exp_file = os.path.join(wrk_dir, config_r['exposures_wrk'])
                # Update work exposures in reduce config
                config_r['exposures_wrk'] = exp_file

                # Make work directory (in needed)
                make_directory(wrk_dir)
                # Add work directory path to output directory
                out_dir = os.path.join(wrk_dir, sub_dirs[exp_type])
                # Make work output directory
                make_directory(out_dir)
                # Set output directory in reduce config
                config_r['output']['dir'] = out_dir

                # Add message to log
                msg = '{0} -- Reduce {1} files:\n'
                log.message(msg.format(MYNAME.upper(), exp_type.lower()))

                # Call action procedure dynamically
                getattr(me, action['name'])(config_r, prd_files, log)

            else:
                # Add message to log
                msg = '{0} -- Reduce {1} files: No data\n'
                log.message(msg.format(MYNAME.upper(), exp_type.lower()))

    return

# ---------------------------------------------------------------------------- #
def reduce_arc(config_r, fits_files, log):
# ---------------------------------------------------------------------------- #

    # Set raw prefix and observation date
    raw_prefix, obs_date = config_r['raw_prefix'], config_r['obs_date']
    # Set wildcard for combined arc file(s)
    wildcard = '{0}{1}Arc*.fits'.format(raw_prefix, obs_date)
    # Add product data directory to wildcard
    wildcard = os.path.join(config_r['prd_dir'], wildcard)
    # Get combined arc file(s) with wildcard in product data directory
    combined_files = sorted(glob.glob(wildcard))
    # Prepend combined arc files BEFORE other arc files
    fits_files = combined_files + fits_files

    # Reduce arc product fits files
    reduce_files(config_r, fits_files, log)

    return

# ---------------------------------------------------------------------------- #
def reduce_sky(config_r, fits_files, log):
# ---------------------------------------------------------------------------- #

    # Reduce sky product fits files
    reduce_files(config_r, fits_files, log)

    return

# ---------------------------------------------------------------------------- #
def reduce_object(config_r, fits_files, log):
# ---------------------------------------------------------------------------- #

    # Reduce object product fits files
    reduce_files(config_r, fits_files, log)

    return

# ---------------------------------------------------------------------------- #
def reduce_files(config_r, fits_files, log):
# ---------------------------------------------------------------------------- #

    # Set exposures
    set_exposures(config_r)
    # Initialise fibres config dictionary
    config_f = {}
    # Load db fibre traces config (JSON input file) dictionary
    config_f['db'] = load_json_file(config_r['fibre_traces_db'])
    # Load work fibre traces config (JSON input file) dictionary
    config_f['wrk'] = load_json_file(config_r['fibre_traces_wrk'])
    # Set wavelength solutions
    solutions = set_wavelength_solutions(config_r)

    # Loop for product fits files...
    for fits_file in fits_files:

        # Set fits name (without full path)
        fits_name = os.path.basename(fits_file)
        # Initialise work dictionary as a copy of reduce config
        work = config_r.copy()
        # Set fits file name (without extension) in work dictionary
        work['file'] = os.path.splitext(fits_name)[0]

        # Add message to log
        now = datetime.now().strftime('%H:%M:%S')
        msg = ' {0} ({1})\n'.format(fits_name, now)
        log.message(msg, with_header=False)

        # Open product science fits file
        with fits.open(fits_file, mode='readonly') as hdu:

            # Set exposure config ids
            set_exposure_config_ids(hdu, work)

            # Check if exposure type is arc, i.e., reference spectrum
            if work['exp_type'] == 'arc' and work['check_combined_solution']:
                # Check if solution already exists for a combined arc
                found = check_combined_arc_solution(fits_name, solutions, work)
                # Check found indicator
                if found:
                    # Add message to log
                    msg = (' - combined arc solution ({0}) found!\n'
                           ).format(work['wrk_config'])
                    log.message(msg, with_header=False)
                    # We'll go with the combined solution, so move on!
                    continue

            # Set fibre traces
            traces = set_fibre_traces(config_f, work, log)
            # Check traces dictionary
            if not traces:
                # Add message to log
                msg = (' - fibre traces ({0}) not found!\n'
                       ).format(work['db_config'])
                log.message(msg, with_header=False)
                # Move on!
                continue

            # Set other work variables
            set_other_work_variables(hdu, traces, work)
            # Initialise good pixels dictionary in work dictionary
            work['good_pixels'] = {}
            # Extract fibres
            fibres = extract_fibres_from_image(hdu, traces, work, log)

            # Check if exposure type is arc, i.e., reference spectrum
            if work['exp_type'] == 'arc':
                # Reduce reference (arc) exposure
                reduce_reference(hdu, solutions, traces, fibres, work, log)

            else:
                # Reduce science (sky and object) exposure
                reduce_science(hdu, solutions, traces, fibres, work, log)

        # Beautify log
        log.message('', with_header=False)

    # Link exposure files to proposal directory
    link_exposure_files(work, log)
    # Dump exposures dictionary
    dump_exposures(work, log)

    # Check if exposure type is arc, i.e., reference spectrum
    if config_r['exp_type'] == 'arc':
        # Dump wavelength solutions (as needed)
        dump_wavelength_solutions(solutions, work, log)

    # Beautify log
    log.message('', with_header=False)

    return

# ---------------------------------------------------------------------------- #
def set_exposures(config_r):
# ---------------------------------------------------------------------------- #

    # Load exposures dictionary (JSON input file)
    exposures = load_json_file(config_r['exposures_wrk'])
    # Check if exposures dictionary is None
    if exposures is None:
        # Initialise exposures dictionary with exposure type being processed
        exposures = {config_r['exp_type']: {}}

    else:
        # Re-initialise exposure type being processed
        exposures[config_r['exp_type']] = {}

    # Update exposures dictionary in reduce config
    config_r['exposures'] = exposures

    return

# ---------------------------------------------------------------------------- #
def set_wavelength_solutions(config_r):
# ---------------------------------------------------------------------------- #

    # Initialise solutions
    solutions = {}
    # Set 'empty' solutions dictionary
    empty = {'obs_date': config_r['obs_date'], 'solutions': {}}
    # Load db wavelength solutions config (JSON input file)
    solutions['db'] = load_json_file(config_r['solutions_db'])
    # Load work wavelength solutions config (JSON input file)
    solutions['wrk'] = load_json_file(config_r['solutions_wrk'], default=empty)

    # Check if exposure type is arc, i.e., reference spectrum, and recalc
    if config_r['exp_type'] == 'arc' and config_r['recalc']:
        # Re-initialise work wavelength solutions config
        solutions['wrk'] = empty

    return solutions

# ---------------------------------------------------------------------------- #
def set_exposure_config_ids(hdu, work):
# ---------------------------------------------------------------------------- #

    # Loop for exposure config key lists...
    for key, key_list in work['config_key']['lists'].items():

        # Get exposure config key value dictionary
        key_dict = key_values(hdu, key_list, include_blank=False)
        # Set exposure config id in work dictionary
        work[key] = key_values_id(work['config_key']['frmts'], key_dict)

    # Check if work config id is not in exposures dictionary for exposure type
    if work['wrk_config'] not in work['exposures'][work['exp_type']]:
        # Initialise exposures list for work config id
        work['exposures'][work['exp_type']][work['wrk_config']] = {}

    return

# ---------------------------------------------------------------------------- #
def check_combined_arc_solution(fits_name, solutions, work):
# ---------------------------------------------------------------------------- #

    # Initialise found indicator: False
    found = False

    # Set combined arc file name (without file extension)
    combined_arc = '{0}{1}Arc{2}'.format(
        work['raw_prefix'], work['obs_date'], work['wrk_config'])

    # Check if current exposure is not combined arc
    if os.path.splitext(fits_name)[0] != combined_arc:
        # Check if solution exists for work config id
        if work['wrk_config'] in solutions['wrk']['solutions'].keys():
            # Set work solution for work config id
            solution = solutions['wrk']['solutions'][work['wrk_config']]
            # Check if solution is for combined arc
            if solution['reference'] == combined_arc:
                # Set found indicator: True
                found = True

    return found

# ---------------------------------------------------------------------------- #
def set_fibre_traces(config_f, work, log):
# ---------------------------------------------------------------------------- #

    # Add message to log
    msg = ' - set fibre traces'
    log.message(msg, with_header=False)

    # Initialise traces dictionary
    traces_dict = {}
    # Set exposure config id as db config id
    config = work['db_config']

    # Check if exposure config id is in work fibre traces
    if config_f['wrk'] and config in config_f['wrk']['traces'].keys():
        # Set existing fibre traces dictionary for db config id
        traces_dict = config_f['wrk']['traces'][config]

    # Else check if exposure config id is in db fibre traces
    elif config_f['db'] and config in config_f['db']['traces'].keys():
        # Set dates for db config
        dates = sorted(config_f['db']['traces'][config], reverse=True)
        # Set date <= work observation date
        date = next((o for o in dates if o <= work['obs_date']), None)
        # Check date <= work observation date not found
        if not date:
            # Re-sort dates for db config
            dates = sorted(config_f['db']['traces'][config], reverse=False)
            # Set date > work observation date
            date = next((o for o in dates if o > work['obs_date']), None)

        # Check if date was found
        if date:
            # Load archive fibre traces dictionary
            archive = load_archive_dictionary(
                date, work['fibre_traces_wrk'], work)
            # Check archive dictionary
            if archive and config in archive['traces'].keys():
                # Set archive fibre traces dictionary for db config id
                traces_dict = archive['traces'][config]

    # Check traces dictionary
    if traces_dict:
        # Add message to log
        msg = '   - set for key: {0}'.format(config)
        log.message(msg, with_header=False)
        # Set traces as the fibres dictionary
        traces = traces_dict['fibres']
        # Set flat field image file name
        flat_file = traces_dict['reference']
        # Add message to log
        msg = '   - set flat field file: {0}'.format(flat_file)
        log.message(msg, with_header=False)
        # Add product directory to file name
        flat_file = os.path.join(work['prd_dir'], flat_file)
        # Set flat field image file name in work dictionary
        work['flat']['reference'] = flat_file

    else:
        # Set empty traces
        traces = {}

    return traces

# ---------------------------------------------------------------------------- #
def set_other_work_variables(hdu, traces, work):
# ---------------------------------------------------------------------------- #

    # Set dispersion axis (0=vertical, 1=horizontal)
    work['axis'] = hdu[PRIMARY].header['DISPAXIS']
    # Set data shape (rows and columns)
    work['rows'], work['cols'] = hdu[SCI].data.shape
    # Set 1D wavelength pixel (columns) array
    work['p'] = np.arange(work['cols'], dtype=np.float32)
    # Set 1D spectral channel array
    work['s'] = np.arange(work['nr_of_object_fibres'], dtype=np.float32)
    # Set spatial pixel (fibre rows) array
    work['r'] = np.zeros(work['nr_of_fibres'], dtype=np.float32)
    # Set wavelength zero points array
    work['zps'] = np.zeros(work['nr_of_fibres'], dtype=np.float32)
    # Set slit id for 'centre' fibre id
    work['slit_id'] = traces[work['centre_id']]['slit_id']

    return

# DEBUGGING
# ---------------------------------------------------------------------------- #
def set_dark_file(hdu, work):
    '''
    find master dark frame with the same exposure time as current image using HDU
    '''
# ---------------------------------------------------------------------------- #

    # check if dark subtraction
    dark_cfg = work.get('dark', {})
    if not dark_cfg or not dark_cfg.get('subtract', False):
        return None

    # get exposure time of current frame
    exp_time = hdu[PRIMARY].header['EXPTIME']
    # set raw prefix and observation date
    raw_prefix, obs_date = work['raw_prefix'], work['obs_date']
    # Set wildcard for master dark file(s)
    wildcard = '{0}{1}Dark*.fits'.format(raw_prefix, obs_date)
    # Add product data directory to wildcard
    wildcard = os.path.join(work['prd_dir'], wildcard)
    # Get master dark file(s)
    dark_files = sorted(glob.glob(wildcard))

    if not dark_files:
        raise FileNotFoundError('No master dark files found with wildcard: {0}'.format(wildcard))

    for dark_file in dark_files:

        with fits.open(dark_file, mode='readonly') as dark_hdu:

            # match by exposure time
            dark_exp_time = dark_hdu[PRIMARY].header['EXPTIME']
            if dark_exp_time == exp_time:
                return dark_file

    raise ValueError('No matching master dark found for EXPTIME={0} using wildcard {1}'.format(exp_time, wildcard))

# ---------------------------------------------------------------------------- #
def set_read_noise(hdu, work):
# ---------------------------------------------------------------------------- #
    '''
    Select per-pixel read noise [e-] by nearest EXPTIME from config lookup.
    '''
    exp_time = hdu[PRIMARY].header['EXPTIME']  # exp time from header
    read_noise_map = work['extract']['read_noise_electrons']  # read noise as a function of exp time computed during up-the-ramp sampling in pre-processing. 
    exp_times = np.array([float(k) for k in read_noise_map.keys()])  # get list of exp times
    closest_exp_idx = np.argmin(np.abs(exp_times - exp_time))  # get closest one
    closest_exp = exp_times[closest_exp_idx] 
    read_noise = read_noise_map[str(int(closest_exp))]  # get associated read noise
    return read_noise


# ---------------------------------------------------------------------------- #
def extract_fibres_from_image(hdu, traces, work, log):
# ---------------------------------------------------------------------------- #

    '''
    hdu: <list> opened fits list of data units
    traces: <dictionary> fibre traces dictionary
    work: <dictionary> work dictionary

    return: <dictionary> extracted fibres and good pixels dictionaries
    '''

    # Check if debug
    if work['debug']:
        # Load previously extratced fibres (if any)
        fibres = load_extracted_fibres(work)
        # Check extracted fibres
        if fibres:
            # All done! Return extracted fibres dictionary...
            return fibres

    # Set science image
    # sci = hdu[SCI].data.copy()  # renaming for diagnostic plotting !!!
    sci_raw = hdu[SCI].data.copy()

    ######## DARK SUBTRACTION ##########
    # find master dark frame with the same exposure time as current image (HDU)
    dark_file = set_dark_file(hdu, work)  
    
    if dark_file is not None:
        
        # read and save master dark frame
        with fits.open(dark_file, mode='readonly') as dark_hdu:
            dark = dark_hdu[SCI].data.copy()

        # dark subtract this frame
        sci = sci_raw - dark

        # debug
        dark_sub_diagnostic_plot(work, sci_raw=sci_raw, sci_dark_sub=sci)

    else:
        sci = sci_raw.copy()
    ############################################

    # saving science image without bad-pixel masking for optimal extraction
    sci_unmasked = sci.copy()
    bpm = hdu[BPM].data.copy()
    gpm = np.ones(bpm.shape, dtype=np.float32)
    if work['apply_gpm']:
        gpm[bpm == 1] = 0.
    # science image with bad-pixel masking
    sci *= gpm

    # save extraction method and what type of flat fielding to apply
    method = work.get('extract_method', 'optimal')
    flat_type = work['flat']['type'][work['exp_type']]

    # Initialise flat field / continuum fit image as None
    flt = None
    # Check if flat exposure is needed for extraction
    need_flat = (flat_type in ['flat', 'fit']) or (method == 'optimal')
    if need_flat and (work['flat']['reference'] is not None):

        # Open flat field image file
        with fits.open(work['flat']['reference'], mode='readonly') as fltlist:

            # Set flat field and continuum fit images as needed
            if flat_type == 'fit':
                # Set flat field continuum fit image
                flt = fltlist[FIT].data.copy()
            else:
                # Set flat field image
                flt = fltlist[SCI].data.copy()

        # Set flat field or flat field continuum fit with gpm applied
        flt *= gpm

    # set gain and read noise from headers
    gain = hdu[PRIMARY].header['GAIN']
    read_noise = set_read_noise(hdu, work)

    # fiber extraction
    fibres, good_pixels = extract_fibres(sci, sci_unmasked, gpm, flt, traces, gain, read_noise, work, log)

    # Set good pixels dictionary in work dictionary
    work['good_pixels'] = good_pixels

    # Check if debug
    if work['debug']:
        # Dump extracted fibres to file
        dump_extracted_fibres(fibres, work)

    return fibres

# TEST DIAGNOSTIC PLOT
# ---------------------------------------------------------------------------- #
def dark_sub_diagnostic_plot(work, sci_raw, sci_dark_sub):
# ---------------------------------------------------------------------------- #
    '''
    plotting the science image before and after dark subtraction
    '''

    # difference between raw and dark subtracted image
    difference = sci_raw - sci_dark_sub
    # save scaling values derived from raw image (make sure saturated pixels don't dominate)
    vmin_raw, vmax_raw = np.nanpercentile(sci_raw, [1,99.5])
    vmin_sub, vmax_sub = np.nanpercentile(sci_dark_sub, [1,99.5])
    vmin_dif, vmax_dif = np.nanpercentile(difference, [1,99.5])

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), tight_layout=True)

    im0 = axes[0].imshow(sci_raw, origin='lower', aspect='auto', cmap='gray', vmin=vmin_raw, vmax=vmax_raw)
    axes[0].set_title('raw image')
    axes[0].set_ylabel('detector row')
    plt.colorbar(im0, ax=axes[0], label='flux')

    im1 = axes[1].imshow(sci_dark_sub, origin='lower', aspect='auto', cmap='gray', vmin=vmin_sub, vmax=vmax_sub)
    axes[1].set_title('dark-subtracted image')
    axes[1].set_ylabel('detector row')
    plt.colorbar(im1, ax=axes[1], label='flux')

    im1 = axes[2].imshow(difference, origin='lower', aspect='auto', cmap='gray', vmin=vmin_dif, vmax=vmax_dif)
    axes[2].set_title('difference')
    axes[2].set_ylabel('detector row')
    plt.colorbar(im1, ax=axes[2], label='flux')
    
    # Set png file
    plot_dir = os.path.join(work['output']['dir'],'plots')
    os.makedirs(plot_dir, exist_ok=True)
    png_file = '{0}_dark_sub.png'.format(work['file'])
    # Add output directory path to png file
    filepath = os.path.join(plot_dir, png_file)
    # Save plot as png
    plt.savefig(filepath, dpi=180, format='png', bbox_inches="tight")
    plt.close()

    return


# ---------------------------------------------------------------------------- #
def load_extracted_fibres(work):
# ---------------------------------------------------------------------------- #

    # Initialise extracted fibres dictionary
    fibres = {}

    # Set extracted fibres output file name (JSON file)
    fibres_file = '{0}_extracted.json'.format(work['file'])
    # Add output directory to extracted fibres output file name
    fibres_file = os.path.join(work['output']['dir'], fibres_file)
    # Load extracted fibres from file (if it exists)
    fibres_out = load_json_file(fibres_file)
    # Check extracted fibres
    if fibres_out:
        # Convert extracted fibres lists to arrays
        fibres = {k: np.array(v) for (k, v) in fibres_out.items()}

    return fibres

# ---------------------------------------------------------------------------- #
def dump_extracted_fibres(fibres, work):
# ---------------------------------------------------------------------------- #

    # Set extracted fibres output file name (JSON file)
    fibres_file = '{0}_extracted.json'.format(work['file'])
    # Add output directory to extracted fibres output file name
    fibres_file = os.path.join(work['output']['dir'], fibres_file)
    # Convert extracted fibres arrays to lists
    fibres_out = {k: v.tolist() for (k, v) in fibres.items()}
    # Dump extracted fibres to file
    dump_json_file(fibres_out, fibres_file)

    return

# ---------------------------------------------------------------------------- #
def reduce_reference(hdu, solutions, traces, fibres, work, log):
# ---------------------------------------------------------------------------- #

    # Check lamp id for 'on-detector-combined' format
    if '-' in hdu[PRIMARY].header['LAMPID']:
        # Split lamp id entry on '-'
        lamp_list = hdu[PRIMARY].header['LAMPID'].split('-')
        # Set sorted combined lamp id
        lamp = ''.join(sorted(lamp_list))

    else:
        # Set lamp id as is
        lamp = hdu[PRIMARY].header['LAMPID']

    # Set arc dictionary in work dictionary:
    # - lamp (without any spaces)
    work['arc']['lamp'] = lamp.replace(' ', '')
    # - grating
    work['arc']['grating'] = hdu[PRIMARY].header['GRATING']
    # - camera angle
    work['arc']['camang'] = hdu[PRIMARY].header['CAMANG']
    # - grating tilt
    work['arc']['grtilt'] = hdu[PRIMARY].header['GRTILT']
    # - wavelength key (vac or air)
    work['arc']['wavelength_key'] = work['wavelength_key']

    # Load line list for lamp and grating
    swarr, sfarr, starr = load_line_list(work, log)
    # Create artificial spectrum from line list arrays
    aswarr, asfarr = artificial_spectrum(
        swarr, sfarr, **work['artificial_spectrum'])
    # Set line list dictionary in work dictionary
    work['line_list'] = {'swarr': swarr, 'sfarr': sfarr, 'starr': starr,
                         'aswarr': aswarr, 'asfarr': asfarr}

    # Add message to log
    msg = ' - set observed 1D flux array'
    log.message(msg, with_header=False)
    # Set 'centre' fibre 1D flux array in work dictionary
    work['f'] = fibres[work['centre_id']].copy()

    ###### DEBUGGING #######
    # flip flux array to correct for detector geometry
    work['f'] = orient_fibre_flux(work['f'])
    ########################

    # Smooth 'centre' fibre 1D flux array (if needed)
    work['f'] = smooth_flux_array(work['f'], **work['smooth']['arc'])
    ####>
    # # Fit and subtract continuum
    # work['f'] = fit_and_subtract_continuum_for_fibre(
    #     'arc', work['centre_id'], work, log)
    ####<
    # Set wavelength solution
    ws = set_wavelength_solution(solutions, work, log)
    # Set wavelength solution in work dictionary
    work['ws'] = ws

    # Initialise wavelength fit
    wf, w = initialise_wavelength_fit(hdu, traces, ws, work, log)

    # Check wavelength fit
    if not wf:
        # Add message to log
        msg = '   - wavelength fit not initialised!'
        log.message(msg, with_header=False)
        # Get outa here!
        return

    # Check if either recalc or different obs date
    if work['recalc'] or work['obs_date'] != ws['obs_date']:
        # Determine new wavelength fit
        wf, w = determine_new_wavelength_fit(ws, wf, work, log)
        # Check wavelength fit
        if not wf:
            # Add message to log
            msg = '   - new wavelength fit not found!'
            log.message(msg, with_header=False)
            # Get outa here!
            return

        # Check wavelength range
        if (abs(w.min() - work['w_mod']['min']) > work['w_mod']['tol'] or
          abs(w.max() - work['w_mod']['max']) > work['w_mod']['tol']):
            # Add message to log
            msg = "   - 'bad' wavelength fit (wavelength range outside tolerance)!"
            log.message(msg, with_header=False)
            # Get outa here!
            return

        # Check wavelength fit rms
        if (ws['w_rms'] < work['w_rms_tol']['min'] or
          ws['w_rms'] > work['w_rms_tol']['max']):
            # Add message to log
            msg = "   - 'bad' wavelength fit (rms outside tolerance)!"
            log.message(msg, with_header=False)
            # Get outa here!
            return

        # Get evenly spaced interpolation wavelength array
        we, w1, dw = get_evenly_spaced_array(w.min(), w.max(), w.size)

        # Set wavelength array extend
        ext = work['wavelength']['extend']
        # Extend evenly spaced interpolation wavelength array to shorter wavelengths
        we = np.concatenate((np.arange(w1 - ext * dw, w1 - 1e-8, dw), we))
        # Reset w1
        w1 = we.min()
        # Reset columns in work dictionary
        work['cols'] = len(we)

        # Interpolate 1D row flux array for evenly spaced wavelength array
        f = np.interp(we, w, work['f'], left=0., right=0.)
        # Set flux and wavelength arrays, ref and dispersion in work dictionary
        work['f'] = f; work['we'] = we; work['w1'] = w1; work['dw'] = dw

    # Plot wavelength fit
    plot_wavelength_fit(ws, work, log)

    # Check if either recalc or different obs date
    if work['recalc'] or work['obs_date'] != ws['obs_date']:
        # Reset fibre offsets dictionary
        ws['fibre_offsets'] = {}
        # Fit wavelength coordinates to image (zero points fit)
        zf = fit_wavelength_coordinates(traces, fibres, ws, wf, work, log)
        # Check zero points fit
        if not zf:
            # Add message to log
            msg = '   - zero points fit not found!'
            log.message(msg, with_header=False)
            # Get outa here!
            return

        # Check zero points fit rms
        if (ws['z_rms'] < work['z_rms_tol']['min'] or
          ws['z_rms'] > work['z_rms_tol']['max']):
            # Add message to log
            msg = "   - 'bad' zero points fit (rms outside tolerance)!"
            log.message(msg, with_header=False)
            # Get outa here!
            return

    # Plot zero points fit
    plot_zero_points_fit(ws, work, log)

    # Rectify extracted fibres
    rectify_fibres(traces, fibres, ws, wf, work, log)

    # Stack fibre flux image: fibre type 'all'
    fibre_image = stack_fibre_image(traces, fibres)
    # Stack good pixels count image: fibre type 'all'
    gpcnt_image = stack_fibre_image(traces, work['good_pixels'])
    # Add rectified (wavelength calibrated) header key
    value = time.asctime(time.localtime())
    comment = 'Image has been wavelength calibrated'
    hdu['Primary'].header['WAVECAL'] = (value, comment)
    # Write new fits file
    write_new_fits(hdu, fibre_image, gpcnt_image, '', 'a', work, log)

    # Add work wavelength solution (as needed)
    add_wavelength_solution(ws, solutions, work, log)

    return

# ---------------------------------------------------------------------------- #
def reduce_science(hdu, solutions, traces, fibres, work, log):
# ---------------------------------------------------------------------------- #

    # Set wavelength solution
    ws = set_wavelength_solution(solutions, work, log)
    # Set wavelength solution in work dictionary
    work['ws'] = ws
    # Check wavelength solution
    if not ws:
        # Add message to log
        msg = '   - wavelength solution not found!'
        log.message(msg, with_header=False)
        # Get outa here!
        return

    # Set wavelength fit
    wf, w = set_wavelength_fit(ws, work, log)
    # Get evenly spaced interpolation wavelength array
    we, w1, dw = get_evenly_spaced_array(w.min(), w.max(), w.size)

    # Set wavelength array extend
    ext = work['wavelength']['extend']
    # Extend evenly spaced interpolation wavelength array to shorter wavelengths
    we = np.concatenate((np.arange(w1 - ext * dw, w1 - 1e-8, dw), we))
    # Reset w1
    w1 = we.min()
    # Set wavelength array, ref and dispersion in work dictionary
    work['we'] = we; work['w1'] = w1; work['dw'] = dw
    # Reset columns in work dictionary
    work['cols'] = len(we)

    # Add message to log
    msg = '   - solution wavelength range: {0:.3f} - {1:.3f}'
    log.message(msg.format(ws['w_min'], ws['w_max']), with_header=False)
    # Add message to log
    msg = '   - wavelength fit rms: {0:.4f}'
    log.message(msg.format(ws['w_rms']), with_header=False)
    # Add message to log
    msg = '   - zero points fit rms: {0:.4f}'
    log.message(msg.format(ws['z_rms']), with_header=False)

    # Rectify extracted fibres
    rectify_fibres(traces, fibres, ws, wf, work, log)
    # Stack fibre flux image: fibre type 'all'
    fibre_image = stack_fibre_image(traces, fibres)
    # Stack good pixels count image: fibre type 'all'
    gpcnt_image = stack_fibre_image(traces, work['good_pixels'])
    # Add rectified (wavelength calibrated) header key
    value = time.asctime(time.localtime())
    comment = 'Image has been wavelength calibrated'
    hdu['Primary'].header['WAVECAL'] = (value, comment)
    # Write new fits file: extracted 'all' fibres
    write_new_fits(hdu, fibre_image, gpcnt_image, '', 'a', work, log)

    # Reset 1D wavelength pixel (columns) array for original nr of columns
    work['p'] = np.arange(work['cols'], dtype=np.float32)
    # Stack fibre flux image: fibre type 'obj'
    image = stack_fibre_image(traces, fibres, fibre_type='obj')
    # Stack good pixels count image: fibre type 'obj'
    gpcnt_image = stack_fibre_image(
        traces, work['good_pixels'], fibre_type='obj')

    # Fit and subtract continuum
    cf_image, cs_image = fit_and_subtract_continuum_for_image(
        'sci', image, work, log)
    # Write new fits file: continuum fit
    write_new_fits(hdu, cf_image, None, '', 'cf', work, log)
    # Write new fits file: continuum subtracted
    write_new_fits(hdu, cs_image, None, '', 'cs', work, log)

    # Fit spectral channels
    sf_image = fit_spectral_channels(cs_image, work, log, gpcnt_image=gpcnt_image)  # DEBUG added gpcnt_image as a param for diagnostic plotting
    # Write new fits file: spectral channels fit
    write_new_fits(hdu, sf_image, None, '', 'sf', work, log)

    # Check if exposure type is 'science'
    if work['exp_type'] == 'science':
        # Subtract sky
        sci_image, sci_image_with_cont = subtract_sky(
            hdu, cf_image, cs_image, sf_image, work, log)
        # Write new fits file: sky subtracted without continuum
        write_new_fits(hdu, sci_image, gpcnt_image, '', 'ss', work, log)
        # Write new fits file: sky subtracted with continuum
        write_new_fits(
            hdu, sci_image_with_cont, gpcnt_image, '', 'ssc', work, log)

    return

# ---------------------------------------------------------------------------- #
def set_wavelength_solution(solutions, work, log):
# ---------------------------------------------------------------------------- #

    # Add message to log
    msg = ' - set wavelength solution'
    log.message(msg, with_header=False)

    # Initialise wavelength solution dictionary as None
    ws = None
    # Set work config id
    wrk_config = work['wrk_config']
    # Set db config id
    db_config = work['db_config']

    # Check if exposure type is arc, i.e., reference spectrum
    if work['exp_type'].lower() == 'arc':
        # Add message to log
        msg = '   - initialise for new key: {0}'.format(wrk_config)
        log.message(msg, with_header=False)
        # Set 'empty' wavelength solution dictionary
        ws = work['solution'].copy()
        return ws

    # Set existing work wavelength solutions
    solutions_wrk = solutions['wrk']

    # Check if work config id is in existing work wavelength solutions
    if wrk_config in solutions_wrk['solutions'].keys():
        # Add message to log
        msg = '   - set for existing key: {0}'.format(wrk_config)
        log.message(msg, with_header=False)
        # Set wavelength solution
        ws = solutions_wrk['solutions'][wrk_config].copy()
        # Set 'model' wavelength range in work dictionary
        work['w_mod']['min'], work['w_mod']['max'] = ws['w_min'], ws['w_max']
        # Add message to log
        msg = ('solution wavelength range: {0:.3f} - {1:.3f}'
               ).format(ws['w_min'], ws['w_max'])
        log.message('     - {0}'.format(msg), with_header=False)
        return ws

    # Loop for existing work wavelength solutions keys...
    for key in solutions_wrk['solutions'].keys():

        # Check for config match
        if db_config == key.split('BV')[0]:
            # Add message to log
            msg = '   - set for existing key: {0} ({1})'.format(db_config, key)
            log.message(msg, with_header=False)
            # Set wavelength solution
            ws = solutions_wrk['solutions'][key].copy()
            # Set 'model' wavelength range in work dictionary
            work['w_mod']['min'], work['w_mod']['max'] = ws['w_min'], ws['w_max']
            # Add message to log
            msg = ('solution wavelength range: {0:.3f} - {1:.3f}'
                   ).format(ws['w_min'], ws['w_max'])
            log.message('     - {0}'.format(msg), with_header=False)
            return ws

    # Set existing db solutions
    solutions_db = solutions['db']
    # Check if db exposure config id is in existing db solutions
    if db_config in solutions_db['solutions'].keys():
        # Set dates for db config
        dates = sorted(solutions_db['solutions'][db_config], reverse=True)
        # Set date <= work observation date
        date = next((o for o in dates if o <= work['obs_date']), None)

        # Check date <= work observation date not found
        if not date:
            # Re-sort dates for db config
            dates = sorted(solutions_db['solutions'][db_config], reverse=False)
            # Set date > work observation date
            date = next((o for o in dates if o > work['obs_date']), None)

        # Check if date was found
        if date:
            # Load archive work wavelength solutions dictionary
            archive = load_archive_dictionary(date, work['solutions_wrk'], work)
            # Check archive dictionary
            if archive:

                # Loop for archive work wavelength solutions keys...
                for key in archive['solutions'].keys():

                    # Check for config match
                    if db_config == key.split('BV')[0]:
                        # Add message to log
                        msg = '   - set for existing key: {0}'.format(db_config)
                        log.message(msg, with_header=False)
                        # Set wavelength solution
                        ws = archive['solutions'][key].copy()
                        # Set 'model' wavelength range in work dictionary
                        work['w_mod']['min'] = ws['w_min']
                        work['w_mod']['max'] = ws['w_max']
                        # Add message to log
                        msg = ('solution wavelength range: {0:.3f} - {1:.3f}'
                               ).format(ws['w_min'], ws['w_max'])
                        log.message('     - {0}'.format(msg), with_header=False)
                        return ws

    return ws

# ---------------------------------------------------------------------------- #
def load_archive_dictionary(archive_date, dict_file, work):
# ---------------------------------------------------------------------------- #

    # Set source directory for reduction dictionaries
    src_dir = work['wrk_dir']
    # Set source file name
    src_name = os.path.basename(dict_file)
    # Split archive date
    ccyy, mmdd = archive_date[0:4], archive_date[4:8]
    # Format SALT data archive directory for archive date
    data_dir = work['saltdata_dir'].format(ccyy, mmdd)
    # Set archive work file name (with full path)
    archive_file = os.path.join(data_dir, src_dir, src_name)
    # Load archive work file
    archive = load_json_file(archive_file)

    return archive

# ---------------------------------------------------------------------------- #
def initialise_wavelength_fit(hdu, traces, ws, work, log):
# ---------------------------------------------------------------------------- #

    # Check if existing solution was found for exposure config id
    if ws['reference'] and ws['lamp'] == work['arc']['lamp']:
        # Set wavelength fit
        wf, w = set_wavelength_fit(ws, work, log)

    else:
        # Set articulation (camera) angle adjustment arrays
        ar_angs = np.array(work['ar_ang_adjust']['ar_ang'])
        adjusts = np.array(work['ar_ang_adjust']['adjust'])
        # Set articulation angle
        ar_ang = float(hdu[PRIMARY].header['AR-ANGLE'])
        # Set articulation angle adjustment
        adjust = np.interp(ar_ang, ar_angs, adjusts)
        # Adjust articulation angle for better model wavelength solution
        hdu[PRIMARY].header['AR-ANGLE'] *= 1. + adjust

        # Load spectrograph config (JSON input file): For each fits
        config_s = load_json_file(work['spectrograph'])
        # Add message to log
        msg = ' - set spectrograph: {0}'.format(config_s['name'])
        log.message(msg, with_header=False)
        # Set spectrograph for getting model wavelength array
        spectrograph = set_spectrograph(hdu, config_s)
        # Set m: nr of columns divide by 2
        m = int(work['cols'] / 2.)
        # Set j: fibre centre (row pixel) at column m
        j = get_fibre_centre(traces[work['centre_id']], m, work)
        # Add message to log
        msg = ' - get model wavelength array with grating equation'
        log.message(msg, with_header=False)
        # Get model wavelength array for j (centre row)
        w_mod = 1e7 * spectrograph.get_wavelength(work['p'], j=j)

        ##### DEBUGGING ########
        # notice that model wavelength increases (from blue to red), unlike for raw fibers due to detector geometry....
        # this is why the raw fiber flux is flipped throughout the reduction steps
        print()
        print(f'grating model: w_mod[0]={w_mod[0]:.3f}, w_mod[-1]={w_mod[-1]:.3f}')
        print()
        ########################

        # Check wavelength key (vac, air)
        if work['wavelength_key'] == 'vac':
            # Add message to log
            msg = ' - convert model wavelength array to vacuum'
            log.message(msg, with_header=False)
            # Convert model wavelength array to vacuum wavelengths
            w_mod = air_to_vac(w_mod)

        # Round wavelength array
        w = np.around(w_mod, decimals=DEC)
        # Set model wavelength range in work dictionary
        work['w_mod']['min'], work['w_mod']['max'] = w.min(), w.max()
        # Add message to log
        msg = ' - model wavelength range: {0:.3f} - {1:.3f}'
        log.message(msg.format(w.min(), w.max()), with_header=False)

        # Add message to log
        msg = ' - set wavelength fit from grating equation solution'
        log.message(msg, with_header=False)
        # Initialise wavelength fit
        wf = WavelengthFit(work['p'], w, **work['wavelength']['fit'])

    return wf, w

# ---------------------------------------------------------------------------- #
def set_wavelength_fit(ws, work, log):
# ---------------------------------------------------------------------------- #

    # Add message to log
    msg = ' - set wavelength fit from wavelength solution'
    log.message(msg, with_header=False)

    # Initialise wavelength fit (with wavelength solution arrays)
    wf = WavelengthFit(np.array(ws['pm']), np.array(ws['wm']), **ws['fit'])
    # Set wavelength array for fit
    w = np.around(wf.value(work['p']), decimals=DEC)

    return wf, w

# ---------------------------------------------------------------------------- #
def determine_new_wavelength_fit(ws, wf, work, log):
# ---------------------------------------------------------------------------- #

    # Add message to log
    msg = ' - determine new wavelength fit'
    log.message(msg, with_header=False)

    # Set wavelength fit keywords dictionary
    kw = work['wavelength']['fit_keywords'].copy()
    # Update match line list parameters for lamp id
    kw['match_line_list'] = work['match_line_list'][work['arc']['lamp'].lower()]

    # Calculate wavelength fit for row
    wf = calculate_wavelength_fit(
        work['p'], work['f'], work['line_list'], wf, log, **kw)
    # Check wavelength fit
    if wf is not None:
        # Set wavelength solution identifiers in solution dictionary:
        # - reference spectrum name
        ws['reference'] = work['file']
        # - reference spectrum lamp
        ws['lamp'] = work['arc']['lamp']
        # Set wavelength fit values in solution dictionary:
        # - wavelength fit dictionary
        ws['fit'] = work['wavelength']['fit']
        # - matched pixel array
        ws['pm'] = wf.p.tolist()
        # - matched wavelength array
        ws['wm'] = wf.w.tolist()
        # - wavelength fit coefficients
        ws['w_coef'] = wf.coef.tolist()
        # - wavelength fit rms
        ws['w_rms'] = wf.rms(wf.p, wf.w)
        # Set wavelength fit residuals in work dictionary
        work['w_res'] = wf.res(wf.p, wf.w)
        # Set wavelength array for fit
        w = np.around(wf.value(work['p']), decimals=DEC)
        # Set wavelength array values in solution dictionary:
        # - nr of columns
        ws['n'] = w.size
        # - minimum wavelength
        ws['w_min'] = w.min()
        # - maximum wavelength
        ws['w_max'] = w.max()

        # Add message to log
        msg = '   - solution wavelength range: {0:>.3f} - {1:>.3f}'
        log.message(msg.format(w.min(), w.max()), with_header=False)
        # Add message to log
        msg = '   - wavelength fit rms: {0:.4f}'
        log.message(msg.format(ws['w_rms']), with_header=False)

        return wf, w

    else:
        return None, None

# ---------------------------------------------------------------------------- #
def fit_wavelength_coordinates(traces, fibres, ws, wf, work, log):
# ---------------------------------------------------------------------------- #

    # Add message to log
    msg = ' - fit wavelength coordinates (zero points)'
    log.message(msg, with_header=False)

    # Set wavelength fit keywords dictionary
    kw = work['wavelength']['fit_keywords']
    # Set line list arrays
    sw, sf = work['line_list']['swarr'], work['line_list']['sfarr']
    # Set wavelength range minimum and maximum
    w_min, w_max = ws['w_min'] - kw['pad'], ws['w_max'] + kw['pad']
    # Set wavelength range filter for line list arrays
    w_filter = (sw > w_min) * (sw < w_max)
    # Set wavelength range filtered line list arrays
    sw, sf = sw[w_filter], sf[w_filter]

    # Initialsie fibre ranges list
    fibre_ranges = []
    # Set 'centre' fibre id (str -> int)
    centre = int(work['centre_id'])
    # Set fibre range from 'centre' down (left)
    r1i2, r1i1 = centre - 1, -1
    # Add to fibre ranges list with step -1
    fibre_ranges.append((r1i2, r1i1, -1))
    # Set fibre range from 'centre' + 1 up (right)
    r2i1, r2i2 = centre, len(fibres)
    # Add to fibre ranges list with step 1
    fibre_ranges.append((r2i1, r2i2, 1))
    # Set m: nr of columns divide by 2
    m = int(work['cols'] / 2.)
    # Set zero point difference tolerance
    z_tol = work['zero_points']['tolerance']

    # Loop for fibre ranges...
    for fibre_range in fibre_ranges:

        # Copy 'centre' wavelength fit class object
        twf = copy.deepcopy(wf)

        # Loop for range in fibre range...
        for i in range(fibre_range[0], fibre_range[1], fibre_range[2]):

            # Format fibre id (int -> str)
            fibre_id = '{0:03d}'.format(i + 1)
            # Set 1D fibre flux array
            f = fibres[fibre_id].copy()

            ####### DEBUGGING #########
            # flip flux array to correct for detector geometry
            f = orient_fibre_flux(f)
            ###########################

            # Smooth flux array... if needed
            f = smooth_flux_array(f, **work['smooth']['arc'])

            # Fit wavelength coordinates zero point for row j
            nwf = fit_zero_point(
                work['p'], f, sw, sf, twf, **kw['fit_zero_point'])

            # Check zero point difference
            if abs(nwf.coef[0] - twf.coef[0]) < z_tol:
                # Set zero point in zero points array
                work['zps'][i] = nwf.coef[0]
                # Update twf
                twf = nwf

            # Set r[i]: fibre centre (row pixel) at column m
            work['r'][i] = get_fibre_centre(traces[fibre_id], m, work)

    # Set mask for non-zero zero points
    nz = work['zps'] != 0.
    # Check zero points
    if not work['zps'][nz].size > 0:
        return None

    # Fit wavelength coordinates zero points
    zf = Fit1D(work['r'][nz], work['zps'][nz], **work['zero_points']['fit'])
    # Get zero points using fit coefficients
    zero_points = zf(work['r'])
    # Calculate residuals
    residuals = work['zps'] - zero_points
    # Sigma clip residuals
    res = sigma_clip(residuals, **work['zero_points']['sigma_clip_residuals'])
    # Set mask for clipped residuals
    clipped = residuals != res
    # 'Remove' zero points with clipped residuals
    work['zps'][clipped] = 0.
    # # Set mask for non-zero zero points... again
    nz = work['zps'] != 0.
    # Check zero points
    if not work['zps'][nz].size > 0:
        return None

    # Fit wavelength coordinates zero points
    zf = Fit1D(work['r'][nz], work['zps'][nz], **work['zero_points']['fit'])
    # Check wavelength fit
    if zf is not None:
        # Set zero points fit coefficients in solution dictionary
        ws['z_coef'] = zf.coef.tolist()
        # Set zero points fit rms in solution dictionary
        ws['z_rms'] = zf.sigma(work['r'][nz], work['zps'][nz])

        # Add message to log
        msg = '   - zero points fit rms: {0:.4f}'
        log.message(msg.format(ws['z_rms']), with_header=False)

    return zf

# ---------------------------------------------------------------------------- #
def add_wavelength_solution(solution, solutions, work, log):
# ---------------------------------------------------------------------------- #

    # Set datetime added in wavelength solution dictionary
    solution['added'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

    # Set existing work wavelength solutions
    solutions_wrk = solutions['wrk']['solutions']
    # Check if work config id is in existing work wavelength solutions
    if work['wrk_config'] in solutions_wrk.keys():
        # Check if existing solution must be updated
        if solutions_wrk[work['wrk_config']]['w_rms'] > solution['w_rms']:
            # Add message to log
            msg = (' - update wavelength solution in work solutions (key: {0})'
                   ).format(work['wrk_config'])
            log.message(msg, with_header=False)
            # Update existing solution
            solutions_wrk[work['wrk_config']] = solution

    else:
        # Add message to log
        msg = (' - add wavelength solution to work solutions (key: {0})'
               ).format(work['wrk_config'])
        log.message(msg, with_header=False)
        # Set current solution in work wavelength solutions
        solutions_wrk[work['wrk_config']] = solution

    return

# ---------------------------------------------------------------------------- #
def rectify_fibres(traces, fibres, ws, wf, work, log):
# ---------------------------------------------------------------------------- #

    # Add message to log
    msg = ' - rectify extracted fibres'
    log.message(msg, with_header=False)

    # Check if exposure type is arc, i.e., reference spectrum
    if work['exp_type'] == 'arc' and not ws['fibre_offsets']:
        # Make a temporary work copy of extracted fibres
        tmp = fibres.copy()
        # Rectify fibres (without fibre offsets)
        rectify(traces, tmp, ws, wf, work)

        # Add message to log
        msg = '   - set fibre offsets'
        log.message(msg, with_header=False)
        # Set fibre offsets
        set_fibre_offsets(traces, tmp, ws, work)

    # Rectify fibres (with fibre offsets)
    rectify(traces, fibres, ws, wf, work)

    return

# ---------------------------------------------------------------------------- #
def rectify(traces, fibres, ws, wf, work):
# ---------------------------------------------------------------------------- #

    # Set zero points fit
    zf = Fit1D([], [], coef=ws['z_coef'], **work['zero_points']['fit'])
    # Set m: nr of columns divide by 2
    m = int(work['cols'] / 2.)
    # Set 'centre': fibre centre (row pixel) at column m
    centre = get_fibre_centre(traces[work['centre_id']], m, work)
    # Set zero point shift in wavelength solution
    zps_shift = wf.coef[0] - zf(centre)
    # Initialise fibre offset
    fibre_offset = 0.

    # Loop for extracted fibres...
    for fibre_id, f in fibres.items():

        # Copy 'centre' wavelength fit class object
        twf = copy.deepcopy(wf)
        # Set j: fibre centre (row pixel) at column m
        j = get_fibre_centre(traces[fibre_id], m, work)
        # Check if fibre offsets are set
        if ws['fibre_offsets']:
            # Set offset for fibre
            fibre_offset = ws['fibre_offsets'][fibre_id]

        # Set zero point in wavelength coordinates (zero points) fit
        twf.coef[0] = zf(j) + zps_shift + fibre_offset
        # Update wavelength fit coefficients
        twf.set_coef(twf.coef)
        # Set wavelength array for fit
        w = np.around(twf.value(work['p']), decimals=DEC)
        # Interpolate 1D fibre flux array for evenly spaced wavelength array

        ######## DEBUGGING ###########
        # flip flux array to correct for detector geometry
        f = orient_fibre_flux(f) 
        ##############################

        f = np.interp(work['we'], w, f) # , left=0., right=0.)
        # Update 1D fibre flux array in extracted fibres dictionary
        fibres[fibre_id] = f

    return

# ---------------------------------------------------------------------------- #
def set_fibre_offsets(traces, fibres, ws, work):
# ---------------------------------------------------------------------------- #

    # Set m: nr of columns divide by 2
    m = int(work['cols'] / 2.)
    # Set reference list: matched wavelengths
    w_ref_list = ws['wm']

    # Initialise fibre offsets dictionary
    fibre_offsets = {}

    # Loop for fibre traces...
    for fibre_id in traces.keys():

        # Initialise fibre offsets list
        fibre_offsets[fibre_id] = []

    # Make copy of find peaks dictionary
    find_peaks_dict = work['zero_points']['find_peaks'].copy()

    # Loop for reference wavelengths...
    for w_ref in w_ref_list:

        # Set reference wavelength range filter
        w_filter = (work['we'] > w_ref - 30.) * (work['we'] < w_ref + 30.)
        # Set filtered wavelength array
        w = work['we'][w_filter]
        # Initialise spatial pixel (rows) array
        r = np.zeros(len(traces), dtype=np.float32)
        # Initialise peaks array
        peaks = np.zeros(len(traces), dtype=np.float32)
        # Set nr of pixels around guess to use for convolution
        diff = int(0.5 * len(KERN) + 1)

        # Loop for extracted fibres...
        for fibre_id, fe in fibres.items():

            # Set i
            i = int(fibre_id) - 1
            # Smooth flux
            f = smooth_flux_array(fe[w_filter], **work['smooth']['arc'])
            # Set height for finding peak(s): 80% flux in range
            find_peaks_dict['height'] = 0.8 * f.max()
            # Find all peaks / features: scipy.signal.find_peaks
            pks, _ = find_peaks(f, **find_peaks_dict)
            # Check nr of peaks found
            if len(pks) == 1:
                # Set guess for peak
                guess = w[pks[0]]
                # Get peak centroid - wavelength
                peak = centroid(w, f, kern=KERN, guess=guess, diff=diff)
                # Set peak centroid
                peaks[i] = peak
                # Set r[i]: fibre centre (row pixel) at column m
                r[i] = get_fibre_centre(traces[fibre_id], m, work)
                # Check offset
                if abs(w_ref - peak) < work['zero_points']['offset_tolerance']:
                    # Add fibre offset
                    fibre_offsets[fibre_id].append(w_ref - peak)

    # Loop for fibre offsets...
    for fibre_id, offsets in fibre_offsets.items():

        # Set median offsets
        fibre_offsets[fibre_id] = np.median(offsets)

    # Set fibre offsets in solution dictionary
    ws['fibre_offsets'] = fibre_offsets

    return

# ---------------------------------------------------------------------------- #
def fit_and_subtract_continuum_for_fibre(key, id, work, log):
# ---------------------------------------------------------------------------- #

    # Set flux array (copy from work dictionary)
    farr = work['f'].copy()
    
    # Check if continuum fit and subtraction are needed
    if work['continuum'][key]:

        # Add message to log
        msg = ' - fit and subtract continuum for fibre {0}'.format(id)
        log.message(msg, with_header=False)

        # Set columns range
        cols = work['continuum'][key]['cols']
        # Set mask for fibre
        fm = (work['p'] > cols[0]) * (work['p'] < cols[1])
        # Fit continuum
        cf = fit_continuum(key, 'fibre', id, work['p'], farr, fm, work, log)
        # Set fibre flux minus continuum fit
        farr -= cf(work['p'])

    return farr

# ---------------------------------------------------------------------------- #
def fit_and_subtract_continuum_for_image(key, image, work, log):
# ---------------------------------------------------------------------------- #

    # Add message to log
    msg = ' - fit and subtract continuum'
    log.message(msg, with_header=False)

    # Set nr of image rows and columns
    rows, cols = image.shape
    # Initialise continuum fit image array
    cf_image = np.zeros((rows, cols), dtype=np.float32)
    # Initialise continuum subtracted image array
    cs_image = np.zeros((rows, cols), dtype=np.float32)
    # Initialise mask array
    mask = np.ones((rows, cols), dtype=bool)
    # Set 'x' (wavelength) array
    xarr = work['we']

    # Loop for image rows...
    for i in range(rows):

        # Set 'y' (row flux) array
        yarr = image[i, :].copy()
        # Set mask to exclude 'empty' curved edges of image
        mask = set_curve_mask_for_row(mask, i, yarr, work)
        # Set mask for row
        rm = mask[i, :]
        # Fit continuum
        cf = fit_continuum(key, 'row', i + 1, xarr, yarr, rm, work, log)
        
        #### DEBUGGING ####
        # # Set continuum fit in continuum fit image array
        # cf_image[i][rm] = cf(xarr)[rm]
        # # Set row flux minus continuum fit in continuum subtracted image array
        # cs_image[i][rm] = yarr[rm] - cf(xarr)[rm]

        # Evaluate continuum fit
        cf_vals = cf(xarr)
        # If fit failed on a fibre (with few good pixels), replace NaN or Inf values with zeros
        if not np.all(np.isfinite(cf_vals[rm])):
            cf_vals = np.zeros_like(cf_vals)
        # Set continuum fit in continuum fit image array
        cf_image[i][rm] = cf_vals[rm]
        # Set row flux minus continuum fit in continuum subtracted image array
        cs_image[i][rm] = yarr[rm] - cf_vals[rm]
        ####################

    # Set mask in work dictionary
    work['image_mask'] = mask

    # same plotting as in below commented out block ...
    image_mask_diagnostic_plot(work, xarr, mask)
####>
    # plt.figure(1, figsize=(16, 9), tight_layout=True)
    # ext = (xarr.min(), xarr.max(), 0, mask.shape[0])
    # plt.imshow(mask, extent=ext, origin='lower', aspect='auto', cmap='plasma')
    # plt.show()
    # plt.close()
    # stop()
####<
    return cf_image, cs_image

# ---------------------------------------------------------------------------- #
def fit_continuum(key, tag, id, xarr, yarr, mask, work, log):
# ---------------------------------------------------------------------------- #

    # Check if flux must be sigma clipped
    if work['continuum'][key]['sigma_clip']:
        # Clip 'y' (flux) array
        ysig = sigma_clip(yarr, **work['continuum'][key]['sigma_clip'])

    else:
        ysig = yarr

    # Fit continuum
    cf = Fit1D(xarr[mask], ysig[mask], **work['continuum'][key]['fit'])
    # Plot continuum fit (if needed)
    plot_continuum_fit(key, tag, id, xarr, yarr, ysig, cf, work, log)

    return cf

# ---------------------------------------------------------------------------- #
def fit_spectral_channels(image, work, log, gpcnt_image=None):
    # DEBUG added gpcnt_image as a param for diagnostic plotting
# ---------------------------------------------------------------------------- #

    # Add message to log
    msg = ' - fit spectral channels'
    log.message(msg, with_header=False)

    # Initialise new spectral fit image array
    sf_image = np.zeros(image.shape, dtype=np.float32)
    # Set 1D fitting dictionary
    fit = work['spectral_channels']['fit']

    ####### DEBUG ######
    wav_grid = work['we']
    # wmin = 10250  # straddling a strong sky-line at ~10289 A
    # wmax = 10350  # ''
    wmin = 9780
    wmax = 9820

    # evenly spaced spectral channel wavelengths
    sample_wavs = np.linspace(wmin, wmax, 10)

    # wavelengths to column indices
    diagnostic_cols = np.array([np.argmin(np.abs(wav_grid - w)) for w in sample_wavs])

    # initialize list of fiber numbers that are major outliers in each diagnostic column
    outlier_fiber_list = []
    #####################

    # Loop for columns...
    for j in range(work['cols']):

        # Set column (spectral channel) flux
        farr = image[:, j].copy()
        # Set mask for column
        cm = work['image_mask'][:, j]
        # Check if there are no columns to fit
        if work['s'][cm].size == 0:
            # Move on!
            continue

        # Check nr of data points for fit against fitting order
        if work['s'][cm].size <= fit['order']:
            # Set constant in spectral fit image array
            sf_image[:, j][cm] = farr[cm]
            # Move on!
            continue

        # Fit column (spectral channel) flux
        sf = Fit1D(work['s'][cm], farr[cm], **fit)
        # Set spectral fit in spectral fit image array
        sf_image[:, j][cm] = sf(work['s'])[cm]

        ####### DEBUG ######
        if j in diagnostic_cols:
            outlier_fibers = spec_channel_fit_diagnostic_plot(
                work, 
                input_flux=farr,
                fit_flux=sf(work['s']),
                mask=cm,
                col=j
                )

            # for all outlier fibers ...
            if (len(outlier_fibers) > 0):
                for fib_num in outlier_fibers:
                    outlier_fiber_list.append(fib_num)
        #####################
    
    # Set NaN to 1
    sf_image[np.isnan(sf_image)] = 1.

    return sf_image


# TEST DIAGNOSTIC PLOT
# ---------------------------------------------------------------------------- #
def image_mask_diagnostic_plot(work, xarr, mask):
# ---------------------------------------------------------------------------- #
    '''
    plotting the image mask, which masks out zero flux pixels on the edges of the rectified image
    '''
    plt.figure(1, figsize=(16, 9), tight_layout=True)
    ext = (xarr.min(), xarr.max(), 0, mask.shape[0])
    plt.imshow(mask, extent=ext, origin='lower', aspect='auto', cmap='plasma')

    plt.xlabel('wavelength [A]', fontsize=12, labelpad=15)
    plt.ylabel('fiber #', fontsize=12, labelpad=15)
    
    # Set png file
    plot_dir = os.path.join(work['output']['dir'],'plots')
    os.makedirs(plot_dir, exist_ok=True)
    png_file = '{0}_image_mask.png'.format(work['file'])
    # Add output directory path to png file
    filepath = os.path.join(plot_dir, png_file)
    # Save plot as png
    plt.savefig(filepath, dpi=180, format='png', bbox_inches="tight")
    plt.close()

    return


# TEST DIAGNOSTIC PLOT
# ---------------------------------------------------------------------------- #
def spec_channel_fit_diagnostic_plot(work, input_flux, fit_flux, mask, col):
# ---------------------------------------------------------------------------- #
    '''
    plotting flux vs fiber for a given spectral channel and its resulting 1D fit (with sigma clipping)
    '''
    fibre_index = work['s']

    fibers = fibre_index[mask]
    fluxes = input_flux[mask]
    fluxes_fit = fit_flux[mask]
    residuals = fluxes - fluxes_fit

    # compute sigma of fit
    # (using MAD instead of stdev to compute sigma to be more robust to outliers)
    median_residuals = np.nanmedian(residuals)
    abs_residual_difference = np.abs(residuals - median_residuals)
    MAD = np.nanmedian(abs_residual_difference)
    sigma = 1.4826 * MAD

    sigma_threshold = 5.0
    outlier_mask = abs_residual_difference > sigma_threshold * sigma
    outlier_fibers = fibers[outlier_mask]

    # # get a list of fibers that are outliers
    # outlier_fibers = []
    # for i, fiber_input_flux in enumerate(input_flux[mask]):
    #     fiber_fit_flux = fit_flux[mask][i]

    #     # if input flux values are 20 * the fit (roughly an outlier? need a better quantification)
    #     if (fiber_input_flux > 15+fiber_fit_flux): #or (fiber_input_flux < 20/fiber_fit_flux):
    #         outlier_fibers.append(fibre_index[mask][i])

    plt.figure(figsize=(8, 5))

    plt.plot(fluxes, fibers, '.', label='input')
    plt.plot(fluxes_fit, fibers, '-', label='spectral channel fit')

    wav = work['we'][col]

    plt.title(f'column {col}, wavelength {wav:.2f} A')
    plt.xlabel('flux', fontsize=12, labelpad=15)
    plt.ylabel('fiber #', fontsize=12, labelpad=15)
    plt.legend(fontsize=12)

    # Set png file
    plot_dir = os.path.join(work['output']['dir'],'plots')
    os.makedirs(plot_dir, exist_ok=True)
    png_file = '{0}_sky_fit_column_{1:04d}.png'.format(work['file'], col)
    # Add output directory path to png file
    filepath = os.path.join(plot_dir, png_file)
    # Save plot as png
    plt.savefig(filepath, dpi=180, format='png', bbox_inches="tight")
    plt.close()

    return outlier_fibers


# ---------------------------------------------------------------------------- #
def subtract_sky(hdu, sci_cf_image, sci_cs_image, sci_sf_image, work, log):
# ---------------------------------------------------------------------------- #

    # Add message to log
    msg = ' - subtract sky'
    log.message(msg, with_header=False)

    # Set target name (OBJECT)
    trg_name = hdu[PRIMARY].header['OBJECT']

    # Check sky exposures
    if 'sky' not in work['exposures'].keys():
        # Add message to log
        msg = '   - no sky images!'
        log.message(msg, with_header=False)
        # Get outa here!
        return None, None

    # Check sky continuum fit image
    if work['wrk_config'] not in work['exposures']['sky'].keys():
        # Add message to log
        msg = '   - {0}: sky image not found!'.format(work['wrk_config'])
        log.message(msg, with_header=False)
        # Get outa here!
        return None, None

    # Initialise sky continuum fit image
    sky_cf_image = None

    # Loop for sky continuum fit entries...
    for sky_cf in work['exposures']['sky'][work['wrk_config']]['cf']:

        # Open sky continuum fit image file
        with fits.open(sky_cf['file'], mode='readonly') as skyhdu:

            # Check target name of sky continuum fit image file
            if skyhdu[PRIMARY].header['OBJECT'] == trg_name:
                # Set sky continuum fit image
                sky_cf_image = skyhdu[SCI].data.copy()
                break

    # Check sky continuum fit image
    if sky_cf_image is None:
        # Add message to log
        msg = ('   - {0}: sky continuum fit image not found!'
               ).format(work['wrk_config'])
        log.message(msg, with_header=False)
        # Get outa here!
        return None, None

    # Initialise sky continuum subtracted image
    sky_cs_image = None

    # Loop for sky continuum subtracted entries...
    for sky_cs in work['exposures']['sky'][work['wrk_config']]['cs']:

        # Open sky continuum subtracted image file
        with fits.open(sky_cs['file'], mode='readonly') as skyhdu:

            # Check target name of sky continuum subtracted image file
            if skyhdu[PRIMARY].header['OBJECT'] == trg_name:
                # Set sky continuum subtracted image
                sky_cs_image = skyhdu[SCI].data.copy()
                break

    # Check sky continuum subtracted image
    if sky_cs_image is None:
        # Add message to log
        msg = ('   - {0}: sky continuum subtracted image not found!'
               ).format(work['wrk_config'])
        log.message(msg, with_header=False)
        # Get outa here!
        return None, None

    # Initialise sky spectral channels fit image
    sky_sf_image = None

    # Loop for sky spectral channels fit entries...
    for sky_sf in work['exposures']['sky'][work['wrk_config']]['sf']:

        # Open sky spectral channels fit image file
        with fits.open(sky_sf['file'], mode='readonly') as skyhdu:

            # Check target name of sky spectral channels fit image file
            if skyhdu[PRIMARY].header['OBJECT'] == trg_name:
                # Set sky spectral channels fit image
                sky_sf_image = skyhdu[SCI].data.copy()

    # Check sky spectral channels fit image
    if sky_sf_image is None:
        # Add message to log
        msg = ('   - {0}: sky spectral channels fit image not found!'
               ).format(work['wrk_config'])
        log.message(msg, with_header=False)
        # Get outa here!
        return None, None

    # Initialise object to sky spectral channels fit ratio image
    rat_sf_image = np.ones(sci_sf_image.shape, dtype=np.float32)  # all values initialized to 1
    # Set non-zero mask from sky spectral channels fit image
    nz = sky_sf_image != 0.
    # Set object to sky spectral channels fit ratio
    rat_sf_image[nz] = sci_sf_image[nz] / sky_sf_image[nz]  # wavelength-dependent scaling of sky emission in obj and sky frames
    # Initialise gpm image array (copy of sky spectral channels fit)
    gpm_image = sky_sf_image.copy()
    # Set gpm threshold dictionary for masking sky spectral channels fit
    gpm_thresh = work['spectral_channels']['gpm']['threshold']

    # Check if threshold type for masking is 'auto'
    if gpm_thresh['type'] == 'auto':
        # Clip sky spectral channels fit
        tmp = sigma_clip(sky_sf_image, **gpm_thresh['auto']['sigma_clip'])
        # Set threshold: stddev of sigma clipped sky spectral channels fit
        thresh = gpm_thresh['auto']['scale'] * np.std(tmp)

    # .. else check if threshold type for masking is 'value'
    elif gpm_thresh['type'] == 'value':
        # Set threshold value
        thresh = gpm_thresh['value']

    # masking out interline regions (currently, thresh=0.1)
    # Set zeros
    gpm_image[sky_sf_image <= thresh] = 0.
    # Set ones
    gpm_image[sky_sf_image > thresh] = 1.

    # Clean object continuum subtracted image
    sci_cs_2d = gpm_image * sci_cs_image        # interline regions set to 0, in obj cs image 
    # Clean sky continuum subtracted image
    sky_cs_2d = gpm_image * sky_cs_image        # interline regions set to 0, in sky cs image
    # Collapse object continuum subtracted image
    sci_cs_1d = np.sum(sci_cs_2d, axis=1)       # summing up spectral axis across fibers: (n columns, 1)
    # Some numpy voodoo... set n rows, 1 column
    sci_cs_1d = np.array([sci_cs_1d]).transpose()  # transpose: (n rows, 1)
    # Collapse sky continuum subtracted image
    sky_cs_1d = np.sum(sky_cs_2d, axis=1)        # ''
    # Some numpy voodoo... set n rows, 1 column
    sky_cs_1d = np.array([sky_cs_1d]).transpose()  # ''
    # Initialise object to sky continuum subtracted ratio array (1D)
    rat_cs_1d = np.ones(sci_cs_1d.shape, dtype=np.float32)
    # Set non-zero mask from sky continuum subtracted array
    nz = sky_cs_1d != 0.
    # Set object to sky continuum subtracted ratio array (1D)
    rat_cs_1d[nz] = sci_cs_1d[nz] / sky_cs_1d[nz]
    # Normalise object to sky continuum subtracted ratio array (1D)
    # rat_cs_1d /= rat_cs_1d.mean()     # renaming variables for plotting !!!!
    rat_cs_1d_norm = rat_cs_1d / np.median(rat_cs_1d)
    # Repeat 1D object to sky ratio to same column dimension as 2D images
    # rat_cs_2d = np.repeat(rat_cs_1d, rat_sf_image.shape[1], axis=1)   # renaming variables for plotting !!!
    rat_cs_2d_norm = np.repeat(rat_cs_1d_norm, rat_sf_image.shape[1], axis=1)

    sky_line_sum_diagnostic_plot(
        work,
        sky_spectral_sum_arr=sky_cs_1d,
        object_spectral_sum_arr=sci_cs_1d,
        )
    
    # Combine normalised wavelength and position scalings
    # rat_sf_image *= rat_cs_2d_norm                   # renaming variables for plotting !!!!        
    scaling_image = rat_sf_image * rat_cs_2d_norm    # combining wavelength-dependent and fiber-dependent ratios
    # Clean interline regions
    # rat_sf_image *= gpm_image              # renaming variables for plotting !!!!        
    scaling_image_masked = scaling_image * gpm_image         # interline regions set to 0
    # Set interline regions to unity
    # rat_sf_image[rat_sf_image < 1e-2] = 1.                   # using renamed variable for plotting !!!!       
    scaling_image_masked[scaling_image_masked < 1e-2] = 1.      # interline regions (value of zero) do not contribute to later scaling (factor=1)

    # Scale sky continuum subtracted image
    # sky_scaled = sky_cs_image * rat_sf_image            # using renamed variable for plotting !!!!       
    sky_scaled = sky_cs_image * scaling_image_masked      # scaling the sky lines in the sky frame to the sky lines in the object frame
    # Subtract scaled image from object continuum subtracted image
    sci_image = sci_cs_image - sky_scaled

    sky_line_scaling_diagnostic_plot(
        work,
        sci_cs_image=sci_cs_image,
        sky_cs_image=sky_cs_image,
        fiber_dep_scaling_ratio_2d=rat_cs_2d_norm,
        sky_combined_scaling_factor_2d=scaling_image_masked,
        sky_scaled=sky_scaled, 
        sci_image=sci_image,
        skyline_mask=gpm_image
        )

    # Add continuum back in
    sci_image_with_cont = sci_image + (sci_cf_image - sky_cf_image)

    # skyline_residuals_plot(
    #     work,
    #     flattened_obj_cs_2d=sci_image,
    #     skyline_mask=gpm_image
    #     )

    # Add sky subtracted header key
    value = time.asctime(time.localtime())
    comment = 'Image has been wavelength calibrated'
    hdu['Primary'].header['SKYSUB'] = (value, comment)

    return sci_image, sci_image_with_cont


# TEST DIAGNOSTIC PLOT
# ---------------------------------------------------------------------------- #
def skyline_residuals_plot(work, flattened_obj_cs_2d, skyline_mask):
# ---------------------------------------------------------------------------- #
    '''
    plotting the summed residual intensity of all skylines for each fiber in the object frame after subtraction.
    These skylines are identified with the sky-fit images and meet the set threshold value.
    '''

    # keep only skyline regions
    masked = flattened_obj_cs_2d * skyline_mask

    n_fibres = masked.shape[0]

    mean_resid = np.zeros(n_fibres, dtype=float)
    median_resid = np.zeros(n_fibres, dtype=float)

    for i in range(n_fibres):
        # sky sub intensity for i-th fiber
        #       for all skyline spectral regions 
        vals = masked[i, skyline_mask[i, :] > 0]

        mean_resid[i] = np.mean(vals)
        median_resid[i] = np.median(vals)

    fibre_idx = np.arange(n_fibres)

    plt.figure(figsize=(8, 5))
    plt.plot(fibre_idx, mean_resid, color='black', alpha=0.3, label='mean')
    plt.plot(fibre_idx, median_resid, color='red', alpha=0.7, label='median')
    plt.axhline(0, color='grey', linestyle='--', alpha=0.4)
    plt.ylim(np.min(mean_resid) - 0.2, np.max(mean_resid) + 0.2)
    plt.xlabel('fiber #', fontsize=12, labelpad=15)
    plt.ylabel('residual sky-line intensity', fontsize=12, labelpad=15)
    plt.legend(fontsize=12)
    # Set png file
    plot_dir = os.path.join(work['output']['dir'],'plots')
    os.makedirs(plot_dir, exist_ok=True)
    png_file = '{0}_residual_sky_vs_fiber_threshold_0.5.png'.format(work['file'])
    # Add output directory path to png file
    filepath = os.path.join(plot_dir, png_file)
    # Save plot as png
    plt.savefig(filepath, dpi=500, format='png', bbox_inches="tight")
    plt.close()

    return


# TEST DIAGNOSTIC PLOT
# ---------------------------------------------------------------------------- #
def sky_line_sum_diagnostic_plot(work, sky_spectral_sum_arr, object_spectral_sum_arr):
# ---------------------------------------------------------------------------- #
    '''
    plotting the summed skylines vs. fiber # for both object and sky cs frames.
    the obj-to-sky ratio of these summed skylines is used as a fiber-dependent scale factor in the sky-subtraction step.
    '''

    n_fibres = sky_spectral_sum_arr.shape[0]
    fibre_idx = np.arange(n_fibres)

    plt.figure(figsize=(8,5))
    plt.plot(fibre_idx, sky_spectral_sum_arr, label='sky frame')
    plt.plot(fibre_idx, object_spectral_sum_arr, label='obj frame')

    plt.xlabel('fiber #', fontsize=12, labelpad=15)
    plt.ylabel('summed skyline flux', fontsize=12, labelpad=15)
    plt.legend(fontsize=12)

    # Set png file
    plot_dir = os.path.join(work['output']['dir'],'plots')
    os.makedirs(plot_dir, exist_ok=True)
    png_file = '{0}_summed_sky_lines_vs_fiber.png'.format(work['file'])
    # Add output directory path to png file
    filepath = os.path.join(plot_dir, png_file)
    # Save plot as png
    plt.savefig(filepath, dpi=180, format='png', bbox_inches="tight")
    plt.close()

    return

# TEST DIAGNOSTIC PLOT
# ---------------------------------------------------------------------------- #
def sky_line_scaling_diagnostic_plot(work, sci_cs_image, sky_cs_image, fiber_dep_scaling_ratio_2d, sky_combined_scaling_factor_2d, sky_scaled, sci_image, skyline_mask=None):
# ---------------------------------------------------------------------------- #
    '''
    For each diagnostic fiber, plots:
        Row 1: sci_cs, sky_cs, and scaled_sky spectra at sky-line wavelengths
                --> shows what is being subtracted and whether sky_scaled is too small/large
        Row 2: combined scaling factor applied to sky frame
                --> shows whether the scaling is anomalous for bright fibers
        Row 3: cumulative flux budget: sci_cs - sky_scaled = sci_image
                --> shows the net effect of sky subtraction on the flux
    '''
    # after rectification, all fibers share the same wavelength grid
    wav_grid = work['we']
    offset = 10   #[A]

    diagnostic_fibers = [0, 7, 39, 55, 188, 198, 206, 207, 208, 209]
    for fiber_num in diagnostic_fibers:

        # skip fibers that don't exist in this exposure or have no skyline pixels
        if fiber_num >= sci_cs_image.shape[0]:
            continue
        m = (skyline_mask[fiber_num, :] == 1)  # sky-line wavelength mask
        if m.sum() == 0:

            wav_grid_masked = wav_grid[m]

            # --- quantities at sky-line wavelengths ---
            sci_cs_spec = sci_cs_image[fiber_num, m]
            sky_cs_spec = sky_cs_image[fiber_num, m]
            sky_scl_spec = sky_scaled[fiber_num, m]
            sci_img_spec = sci_image[fiber_num, m]
            combined = sky_combined_scaling_factor_2d[fiber_num, m]
            fib_ratio = fiber_dep_scaling_ratio_2d[fiber_num, 0]   # scalar per fiber

            # --- summed quantities (the flux budget) ---
            sum_sci_cs = sci_cs_image[fiber_num, :].sum()
            sum_sky_scl = sky_scaled[fiber_num, :].sum()
            sum_sci_img = sci_image[fiber_num, :].sum()

            fig, axes = plt.subplots(4, 1, figsize=(10, 13), tight_layout=True)
            fig.suptitle(f'sky-subtraction diagnostic — fiber #{fiber_num}', fontsize=14)

            # --- Row 1: spectra at sky-line wavelengths ---
            # y_2, y_98 = np.percentile(sci_cs_spec, [2, 98])
            # y_lower = y_2 - 10  # 10 counts/s padding
            # y_upper = y_98 + 30 # 30 counts/s padding

            ax = axes[0]
            ax.scatter(wav_grid_masked, sci_cs_spec, s=3, color='black', label='obj cs')
            ax.scatter(wav_grid_masked - offset*2, sky_cs_spec, s=3, color='orange', label='sky cs')
            ax.scatter(wav_grid_masked - offset, sky_scl_spec, s=3, color='blue', label='sky scaled')
            # ax.step(wav_grid_masked, sci_cs_spec,  where='mid', color='black',  lw=1.2, label='obj cs')
            # ax.step(wav_grid_masked - offset*2, sky_cs_spec,  where='mid', color='orange', lw=1.2, alpha=0.7, label='sky cs')
            # ax.step(wav_grid_masked - offset, sky_scl_spec, where='mid', color='blue',   lw=1.2, alpha=0.7, label='sky scaled')
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.6, alpha=0.5)
            ax.set_ylabel('counts / s', fontsize=11)
            ax.set_title('spectra at sky-line wavelengths', fontsize=10)
            ax.legend(fontsize=10, loc='upper right')
            ax.set_ylim(np.min(sci_cs_spec) - 5, np.max(sci_cs_spec) + 5)

            # --- Row 2: combined scaling factor ---
            ax = axes[1]
            ax.scatter(wav_grid_masked, combined,  color='black', s=3, label='wavelength * fiber scale factor')
            ax.axhline(fib_ratio, color='grey', linestyle='--', linewidth=1.2, alpha=0.9, label=f'fiber scale factor ({fib_ratio:.2f})')
            ax.set_ylabel('scale factor', fontsize=11)
            ax.set_title('scaling for sky frame: wavelength scaling (obj/sky skyfit for skylines) & fiber scaling (wavelength-avg obj/sky cs for skylines)', fontsize=10)
            ax.legend(fontsize=10, loc='upper right')

            # --- Row 3: result of sky subtraction at sky-line wavelengths ---
            # y_2, y_98 = np.percentile(sci_img_spec, [2, 98])
            # y_lower = y_2 - 10  # 10 counts/s padding
            # y_upper = y_98 + 20 # ''

            ax = axes[2]
            ax.scatter(wav_grid_masked, sci_cs_spec, s=3, color='black', label='obj cs (before sky-sub)')
            ax.scatter(wav_grid_masked - offset, sci_img_spec, s=3, color='red', label='sci image (after sky-sub)')
            # ax.step(wav_grid_masked, sci_cs_spec,  where='mid', color='black', lw=1.2, alpha=0.9, linestyle='--', label='obj cs (before sky-sub)')
            # ax.step(wav_grid_masked, sci_img_spec, where='mid', color='red', alpha=0.6, lw=1.2, label='sci image (after sky-sub)')
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.6, alpha=0.5)
            ax.set_xlabel('wavelength [Å]', fontsize=11)
            ax.set_ylabel('counts / s', fontsize=11)
            ax.set_title('sky subtraction result at sky-line wavelengths', fontsize=10)
            ax.legend(fontsize=10, loc='upper right')
            ax.set_ylim(np.min(sci_img_spec) - 5, np.max(sci_img_spec) + 5)


            # --- Row 4: flux budget (summed over all wavelengths) ---
            ax = axes[3]
            labels = ['sci_cs\n(before sub)', 'sky_scaled\n(subtracted)', 'sci_image\n(after sub)']
            values = [sum_sci_cs, sum_sky_scl, sum_sci_img]
            colors = ['black', 'blue', 'red']
            bars = ax.bar(labels, values, color=colors, alpha=0.5)
            # annotate bar values
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.0f}', ha='center', va='bottom', fontsize=9)
            ax.set_ylabel('summed flux (counts / s)', fontsize=11)
            ax.set_title('flux budget: sci_cs - sky_scaled = sci_image', fontsize=10)

            # Set png file
            plot_dir = os.path.join(work['output']['dir'],'plots')
            os.makedirs(plot_dir, exist_ok=True)
            png_file = '{0}_sky_scaled_spectrum_of_fiber_{1}.png'.format(work['file'],fiber_num)
            # Add output directory path to png file
            filepath = os.path.join(plot_dir, png_file)
            # Save plot as png
            plt.savefig(filepath, dpi=500, format='png', bbox_inches="tight")
            plt.close()

    return



# ---------------------------------------------------------------------------- #
def set_curve_mask(image, work):
# ---------------------------------------------------------------------------- #

    # Set nr of image rows and columns
    rows, cols = image.shape
    # Initialise mask array
    mask = np.ones((rows, cols), dtype=bool)

    # Loop for image rows...
    for i in range(rows):

        # Set row flux
        farr = image[i, :].copy()
        # Set mask to exclude 'empty' curved edges of image
        mask = set_curve_mask_for_row(mask, i, farr, work)

    # Set mask in work dictionary
    work['image_mask'] = mask

    return

# ---------------------------------------------------------------------------- #
def set_curve_mask_for_row(mask, i, farr, work):
# ---------------------------------------------------------------------------- #

    # Set row zero flux pixel array
    zeros = work['p'][farr == 0.].astype(dtype=int)
    # Initialise row sequences list
    sequences = []

    # Loop through zero flux pixels...
    for k, j in enumerate(zeros):

        # Add to applicable sequence in sequences list
        if not k or j - 1 != sequences[-1][-1]:
            sequences.append([j])

        else:
            sequences[-1].append(j)

    # Loop for sequences...
    for sequence in sequences:

        # Check if sequence for start or end of row
        if sequence[0] == 0 or sequence[-1] == work['cols'] - 1:
            # Set mask for sequence
            mask[i, sequence] = False

    return mask

# ---------------------------------------------------------------------------- #
def stack_fibre_image(traces, fibres, fibre_type='all', dtype=np.float32):
# ---------------------------------------------------------------------------- #

    # Initialise fibres image
    fibre_image = None

    # Loop for extracted fibres...
    for fibre_id, f in fibres.items():

        # Check fibre type
        if traces[fibre_id]['type'] != fibre_type and fibre_type !='all':
            # Ignore! Move on..
            continue

        # Check if fibre image is None
        if fibre_image is None:
            # Set fibre image as 1st 1D flux array
            fibre_image = f

        else:
            # Add current 1D flux array to fibre image
            fibre_image = np.vstack((fibre_image, f))

    # Check fibre image
    if fibre_image is not None:
        # Set image array as specified type
        fibre_image = np.array(fibre_image, dtype=dtype)

    return fibre_image

# ---------------------------------------------------------------------------- #
def write_new_fits(hdu, new_image, gp_image, prefix, tag, work, log):
# ---------------------------------------------------------------------------- #

    # Check new image
    if new_image is None: return hdu

    # Initialise new hdu
    hdu_new = []

    # Create new Primary extension (copy header from original fits)
    hdu_primary = fits.PrimaryHDU(header=hdu[PRIMARY].header)
    # Set nr of extensions in Primary extension header
    hdu_primary.header['NSCIEXT'] = (1, 'Number of science extensions')
    hdu_primary.header['NEXTEND'] = (1, 'Number of data extensions')
    # Add new Primary extension to new hdu
    hdu_new.append(hdu_primary)

    # Create science hdu
    hdu_sci = fits.ImageHDU(data=new_image, name=SCI)
    # Set header items
    set_header_items(hdu_sci.header, work)
    # Set extension name in science header
    hdu_sci.header['EXTNAME'] = (SCI, 'Extension name')
    # Set extension nr in science header
    hdu_sci.header['EXTVER'] = (1, 'Extension number')
    # Add new science extension to new hdu
    hdu_new.append(hdu_sci)

    # Check good pixel count image
    if gp_image is not None:
        # Create science hdu
        hdu_sci = fits.ImageHDU(data=gp_image, name=GPCNT)
        # Set header items
        set_header_items(hdu_sci.header, work)
        # Set extension name in science header
        hdu_sci.header['EXTNAME'] = (GPCNT, 'Extension name')
        # Set extension nr in science header
        hdu_sci.header['EXTVER'] = (2, 'Extension number')
        # Add new science extension to new hdu
        hdu_new.append(hdu_sci)

    # Create new fits
    hdu = fits.HDUList(hdus=hdu_new)
    # Check tag
    if tag:
        # Set new fits file name with tag
        new_name = '{0}{1}.{2}.fits'.format(prefix, work['file'], tag)

    else:
        # Set new fits file name without tag
        new_name = '{0}{1}.fits'.format(prefix, work['file'])
        # Set tag for exposure dictionary
        tag = 'a'

    # Add output directory to new fits file name
    new_file = os.path.join(work['output']['dir'], new_name)
    # Add message to log
    msg = ' - write new fits: {0}'.format(new_name)
    log.message(msg, with_header=False)
    # Write new fits file
    hdu.writeto(new_file, overwrite=True, output_verify='ignore')

    # Set exposures dictionary entry for work config id
    exposures = work['exposures'][work['exp_type']][work['wrk_config']]
    # Check if tag doesn't exist in exposures
    if tag not in exposures.keys():
        # Initialise exposures list for tag
        exposures[tag] = []

    # Set exposures list entry dictionary
    entry = {
        'file': new_file,
        'proposal': hdu[PRIMARY].header['PROPID'],
        'object': hdu[PRIMARY].header['OBJECT']
    }
    # Add entry dictionary to exposures list
    exposures[tag].append(entry)

    return hdu

# ---------------------------------------------------------------------------- #
def set_header_items(header, work):
# ---------------------------------------------------------------------------- #

    # Set fits header keywords
    header['DISPAXIS'] = (1, 'Dispersion axis (0=vertical, 1=horizontal)')
    # Check wavelength ref and dispersion
    if 'w1' in work and 'dw' in work:
        header['CRPIX1'] = (1, 'Reference pixel')
        header['CRVAL1'] = (work['w1'], 'Coordinate at reference pixel')
        header['CDELT1'] = (work['dw'], 'Coordinate increment per pixel')
        header['CTYPE1'] = ('WAVELENGTH', 'Coordinate type')
        header['W_RMS'] = (work['ws']['w_rms'], 'Wavelength solution rms')
        header['Z_RMS'] = (work['ws']['z_rms'], 'Zero point solution rms')

        # NOTE: The wavelength array is set as:
        # warr = CRVAL1 - (CRPIX1 - 1.) * CDELT1 + np.arange(NAXIS1) * CDELT1

    return

# ---------------------------------------------------------------------------- #
def link_exposure_files(work, log):
# ---------------------------------------------------------------------------- #

    # Set message for log
    msg = ' Link output files:\n'
    log.message(msg, with_header=False)

    # Initialise linked indicator
    linked = False

    # Loop for exposure type exposure groups...
    for tag_exposures in work['exposures'][work['exp_type']].values():

        # Loop for tagged exposures...
        for exposures in tag_exposures.values():

            # Loop for exposures...
            for exposure in exposures:

                # Set source file name
                src_name = os.path.basename(exposure['file'])
                # Set source file name with instrument product directory path
                src_file = exposure['file']
                # Set destination directory
                dst_dir = os.path.join(exposure['proposal'], 'product')

                # if no destination directory, make one
                os.makedirs(dst_dir, exist_ok=True)

                # Add message to log
                msg = ' - {0} -> {1}'.format(src_name, dst_dir)
                log.message(msg, with_header=False)
                # Link source file to destination directory
                link_file(src_name, src_file, dst_dir)
                # Set linked indicator
                linked = True

                # Check if exposure type is arc, i.e., reference spectrum
                if work['exp_type'] == 'arc':
                    # Set exposure file without ramp and file extension
                    exposure_file = exposure['file'].split('.')[0]
                    # Set wildcard for wavelength and zero points fit plot files
                    wildcard = '{0}*.png'.format(exposure_file)
                    # Get wavelength and zero points fit plot source files
                    src_files = sorted(glob.glob(wildcard))

                    # Loop for source (plot) files...
                    for src_file in src_files:

                        # Set source file name
                        src_name = os.path.basename(src_file)
                        # Add message to log
                        msg = ' - {0} -> {1}'.format(src_name, dst_dir)
                        log.message(msg, with_header=False)
                        # Link source file to destination directory
                        link_file(src_name, src_file, dst_dir)

    # Check if nothing was linked
    if not linked:
        # Set message for log
        msg = ' - Nothing linked!'
        log.message(msg, with_header=False)

    # Beautify log
    log.message('', with_header=False)

    return

# ---------------------------------------------------------------------------- #
def link_file(src_name, src_file, dst_dir):
# ---------------------------------------------------------------------------- #

    # Set link (destination) file
    dst_file = os.path.join(dst_dir, src_name)
    # Check if symbolic link exists
    if os.path.isfile(dst_file):
        # Remove symbolic link
        os.remove(dst_file)

    # Make symbolic link
    os.symlink(os.path.join('../..', src_file), dst_file)

    return

# ---------------------------------------------------------------------------- #
def dump_exposures(work, log):
# ---------------------------------------------------------------------------- #

    # Add message to log
    msg = ' Dump exposures dictionary'
    log.message(msg, with_header=False)
    # Dump exposures dictionary to file
    dump_json_file(work['exposures'], work['exposures_wrk'])

    return

# ---------------------------------------------------------------------------- #
def dump_wavelength_solutions(solutions, work, log):
# ---------------------------------------------------------------------------- #

    # Check work wavelength solutions
    if solutions['wrk']['solutions']:
        # Add message to log
        msg = ' Dump work wavelength solutions config'
        log.message(msg, with_header=False)
        # Dump work wavelength solutions config to file
        dump_json_file(solutions['wrk'], work['solutions_wrk'])

    # Set existing db wavelength solutions
    solutions_db = solutions['db']
    # Initialise db update indicator: False
    db_update = False

    # Loop for work wavelength solutions...
    for wrk_config in solutions['wrk']['solutions'].keys():

        # Set db config id (remove BVISITID)
        db_config = wrk_config.split('BV')[0]

        # Check if db config id is not in existing db wavelength solutions
        if db_config not in solutions_db['solutions'].keys():
            # Initialise 'new' config entry in db wavelength solutions
            solutions_db['solutions'][db_config] = []

        # Check if observation date is not in existing date list
        if work['obs_date'] not in solutions_db['solutions'][db_config]:
            # Add observation date to date list
            solutions_db['solutions'][db_config].append(work['obs_date'])
            # Sort date list (in place)
            solutions_db['solutions'][db_config].sort()
            # Set db update indicator: True
            db_update = True

    # Check if db wavelength solutions config dump is necessary
    if db_update:
        # Add message to log
        msg = ' Dump db wavelength solutions config'
        log.message(msg, with_header=False)
        # Dump db wavelength solutions config to file
        dump_json_file(solutions['db'], work['solutions_db'])

    return

# ---------------------------------------------------------------------------- #
def load_line_list(work, log):
# ---------------------------------------------------------------------------- #

    # Add message to log
    msg = ' - load line list: {lamp} {grating}'.format(**work['arc'])
    log.message(msg, with_header=False)

    # Set lamp key
    lamp = work['arc']['lamp'].lower()

    # Set line list file name (with camera and grating angles)
    ll_name = work['line_lists'][lamp].format(**work['arc'])
    # Format line list file name (with camera and grating angles)
    ll_file = os.path.join(work['config_dir'], ll_name)
    # Check if line list file (with camera and grating angles) doesn't exist
    if not os.path.isfile(ll_file):
        # Set general line list file name (without camera and grating angles)
        ll_name = work['line_lists_base'][lamp].format(**work['arc'])
        # Format line list file name (with camera and grating angles)
        ll_file = os.path.join(work['config_dir'], ll_name)

    # Add message to log
    msg = '   - line list file: {0}'.format(ll_name)
    log.message(msg, with_header=False)

    # Load line list: wavelength and flux
    swarr, sfarr = np.loadtxt(ll_file, unpack=True, usecols=(0, 1))
    # Load line list: tag
    starr = np.loadtxt(ll_file, dtype=str, unpack=True, usecols=(2))

    return swarr, sfarr, starr

# ---------------------------------------------------------------------------- #
def load_sky_lines(work, log):
# ---------------------------------------------------------------------------- #

    # Add message to log
    msg = ' - load line list: sky'
    log.message(msg, with_header=False)
    # Load line list
    swarr = np.loadtxt(work['line_lists']['sky'], unpack=True, usecols=(0))

    return swarr

# ---------------------------------------------------------------------------- #
def get_fibre_centre(fibre, m, work):
# ---------------------------------------------------------------------------- #

    # Set lower trace fit
    lt = Fit1D([], [], coef=fibre['lower_trace'], **work['trace']['fit'])
    # Set upper trace fit
    ut = Fit1D([], [], coef=fibre['upper_trace'], **work['trace']['fit'])

    # Set j: fibre centre (row pixel) at column m
    j = (ut(m) + lt(m)) / 2.

    return j

# ---------------------------------------------------------------------------- #
def smooth_flux_array(farr, **par):
# ---------------------------------------------------------------------------- #

    # Check if smoothing is required
    if (par and par['window_length'] > 0 and par['polyorder'] > 0):
        # Smooth 1D flux array
        farr = savgol_filter(farr, **par)

    return farr

# ---------------------------------------------------------------------------- #
def orient_fibre_flux(f):
# DEBUGGING
# Due to the detector geometry, the extracted fibre wavelength increases right-to-left (red-to-blue)
# ---------------------------------------------------------------------------- #
    return f[::-1]

# ---------------------------------------------------------------------------- #
def centroid(parr, farr, guess=None, diff=None, kern=KERN, mode='same'):
# ---------------------------------------------------------------------------- #

    '''
    Find the centroid of a line following a similar algorithm as the centroid
    algorithm in IRAF. The input arrays should be an area around the feature
    to be centroided. The default kernel is used if one is not specified.

    The algorithm solves for the solution to the equation

    ..math:: \int (I-I_0) f(x-x_0) dx = 0

    parr: <array> 1D pixel array: observed spectrum
    farr: <array> 1D flux array: observed spectrum
    guess: <int> initial guess (pixel)
    diff: <int> nr of pixels around guess to use for convolution
    kern: <list> kernel to convolve array
    mode: <str> mode of convolution
        full: default, output shape of (N+M-1,)
        same: output of length max(M, N)
        valid: output of length max(M, N) - min(M, N) + 1

    return c: <float> centroid
    '''

    if diff < len(kern):
        diff = len(kern)

    if guess is not None and diff:
        mask = (abs(parr - guess) < diff)

    else:
        mask = np.ones(len(parr), dtype=bool)

    # Convolve flux array with the kernel
    carr = np.convolve(farr[mask], kern, mode=mode)

    # cmask is used to make sure only centre pixels are selected
    cmask = (abs(parr[mask] - parr[mask].mean()) < 3)
    # Interpolate to get centroid
    c = np.interp(0, carr[cmask], parr[mask][cmask])

    return c

# ---------------------------------------------------------------------------- #
def plot_wavelength_fit(ws, work, log):
# ---------------------------------------------------------------------------- #

    if not work['output']['plot_wavelength_fit']: return

    # Add message to log
    msg = ' - plot wavelength fit'
    log.message(msg, with_header=False)

    # Set plotting dictionary: arc
    plot_dict = work['plotting']['arc']

    # Set figure size and layout
    plt.figure(1, figsize=plot_dict['figsize'], tight_layout=True)
    # Set height ratios for subplots
    gs = grid.GridSpec(2, 1, height_ratios=[2, 1]) 
    # Set subplots
    ax0 = plt.subplot(gs[0]); ax1 = plt.subplot(gs[1], sharex=ax0)

    # Plot arc spectrum with line list over lay on ax0
    plot_arc_spectrum(ax0, plot_dict, ws, work, log)
    # Set legend
    set_legend(ax0, plot_dict)

    # Set plotting dictionary: residuals
    plot_dict = work['plotting']['residuals']
    # Update xy limits in plotting dictionary:
    # - xlim
    plot_dict['xlim']['min'] = work['we'].min()
    plot_dict['xlim']['max'] = work['we'].max()
    # - ylim
    plot_dict['ylim']['min'] = abs(work['w_res']).max()
    plot_dict['ylim']['max'] = abs(work['w_res']).max()
    # - set plot axis properties
    set_plot_axis_properties(ax1, **plot_dict)

    # Plot residuals as scatter plot on ax1
    ax1.scatter(ws['wm'], work['w_res'], **plot_dict['residuals'])
    # Plot horizontal zero point line
    ax1.axhline(y=0, **plot_dict['zero_line'])
    # Set legend
    set_legend(ax1, plot_dict)

    # Set title dictionary
    title_dict = {'file': work['file'],
                  **work['arc'],
                  **ws['fit'],
                  'rms': ws['w_rms'],
                  'fibre': work['centre_id']}

    # Set title
    title = plot_dict['title_frmt'].format(**title_dict)
    plt.title(title, fontsize=11, y=plot_dict['title_ypos'])
    # Hide xtick labels on ax0 
    plt.setp(ax0.get_xticklabels(), visible=False)

    # Save figure
    if plot_dict['save']:
        # Set png file
        plot_dir = os.path.join(work['output']['dir'],'plots')
        os.makedirs(plot_dir, exist_ok=True)
        png_file = '{0}_wavelength_fit.png'.format(work['file'])
        # Add output directory path to png file
        filepath = os.path.join(plot_dir, png_file)
        # Save plot as png
        plt.savefig(filepath, dpi=180, format='png')

    # Show figure
    if plot_dict['show']: plt.show()
    # Clear current figure
    plt.clf()
    # Close current figure window (if any)
    plt.close()

    return

# ---------------------------------------------------------------------------- #
def plot_arc_spectrum(ax, plot_dict, ws, work, log):
# ---------------------------------------------------------------------------- #

    # Set variables for artificial spectrum
    asw = work['line_list']['aswarr']
    asf = work['line_list']['asfarr']
    # Mask artificial spectrum for observed wavelength range
    masw = asw[(asw > work['we'].min()) * (asw < work['we'].max())]
    masf = asf[(asw > work['we'].min()) * (asw < work['we'].max())]
    # Set line list wavelengths, fluxes and tags
    sw = work['line_list']['swarr']
    sf = work['line_list']['sfarr']
    st = work['line_list']['starr']
    # Mask line list wavelengths, fluxes and tags for observed wavelength range
    msw = sw[(sw > work['we'].min()) * (sw < work['we'].max())]
    msf = sf[(sw > work['we'].min()) * (sw < work['we'].max())]
    mst = st[(sw > work['we'].min()) * (sw < work['we'].max())]
    # Scale artificial spectrum flux for observed flux
    masf *= work['f'].max() / masf.max()

    # Update limits in plotting dictionary:
    # - xlim
    plot_dict['xlim']['min'] = work['we'].min()
    plot_dict['xlim']['max'] = work['we'].max()
    # - ylim
    plot_dict['ylim']['min'] = work['f'].max()
    plot_dict['ylim']['max'] = work['f'].max()
    # Set plot axis properties
    set_plot_axis_properties(ax, **plot_dict)

    # Plot arc spectrum
    ax.plot(work['we'], work['f'], **plot_dict['arc_spectrum'])
####>
    # # Plot artificial spectrum
    # ax.plot(masw, masf, **plot_dict['artificial_spectrum'])
####<
    # Loop for masked line list wavelengths...
    for w, t in zip(msw, mst):

        # Check if line list wavelength must NOT be displayed
        if '*' in t:
            # Don't display... Move on!
            continue

        # Set label for line list wavelength
        lbl = '{0:.2f} ({1})'.format(w, t)

        # Set mask for flux around matched wavelength
        mask = (work['we'] > w - 2.5) * (work['we'] < w + 2.5)

        # Check annotate position
        if plot_dict['annotate_pos'] == 'top':
            # Set xy position of label
            x, y = w, ax.get_ylim()[1]

        else:
            # Set xy position of label
            x, y = w, work['f'][mask].max() * 1.010

        # Add label for line list wavelength
        ax.annotate(lbl, (x, y), **plot_dict['annotate'])

        # Check matched line list wavelengths
        if ws and ws['wm']:
            # Check if masked line list wavelength was matched
            if w in ws['wm']:
                # Plot vertical line for matched wavelength
                ax.axvline(x=w, **plot_dict['matched_lines'])

            else:
                # Plot vertical line for masked wavelength
                ax.axvline(x=w, **plot_dict['masked_lines'])

    return

# ---------------------------------------------------------------------------- #
def plot_zero_points_fit(ws, work, log):
# ---------------------------------------------------------------------------- #

    if not work['output']['plot_zero_points_fit']: return

    # Add message to log
    msg = ' - plot zero points fit'
    log.message(msg, with_header=False)

    # Set zero points fit
    zf = Fit1D([], [], coef=ws['z_coef'], **work['zero_points']['fit'])
    # Set mask for non-zero zero points
    nz = work['zps'] != 0.

    # Set plotting dictionary: zero_points_fit
    plot_dict = work['plotting']['zero_points_fit'].copy()

    # Set figure size and layout
    plt.figure(1, figsize=plot_dict['figsize'], tight_layout=True)
    # Set subplot
    ax = plt.subplot(1, 1, 1)

    # Update xy limits in plotting dictionary:
    # - xlim
    plot_dict['xlim']['min'] = work['zps'][nz].min()
    plot_dict['xlim']['max'] = work['zps'][nz].max()
    # - ylim
    plot_dict['ylim']['min'] = work['r'].max()
    plot_dict['ylim']['max'] = work['r'].max()
    # Set plot axis properties
    set_plot_axis_properties(ax, **plot_dict)

    # Plot zero points
    ax.scatter(work['zps'][nz], work['r'][nz], **plot_dict['zero_points'])
    # Plot zero points fit
    ax.plot(zf(work['r']), work['r'], **plot_dict['fit'])
    # Set legend
    set_legend(ax, plot_dict)

    # Set title dictionary
    title_dict = {'file': work['file'],
                  **work['zero_points']['fit'],
                  'rms': ws['z_rms']}
    # Set title
    title = plot_dict['title_frmt'].format(**title_dict)
    plt.title(title, fontsize=11, y=plot_dict['title_ypos'])

    # Save figure
    if plot_dict['save']:
        # Set png file
        plot_dir = os.path.join(work['output']['dir'],'plots')
        os.makedirs(plot_dir, exist_ok=True)
        png_file = '{0}_zps_fit.png'.format(work['file'])
        # Add output directory path to png file
        filepath = os.path.join(plot_dir, png_file)
        # Save plot as png
        plt.savefig(filepath, dpi=180, format='png')

    # Show figure
    if plot_dict['show']: plt.show()
    # Clear current figure
    plt.clf()
    # Close current figure window (if any)
    plt.close()

    return

# ---------------------------------------------------------------------------- #
def plot_continuum_fit(key, tag, id, xarr, yarr, ysig, cf, work, log):
# ---------------------------------------------------------------------------- #

    if not work['output']['plot_continuum_fit']: return

    # Add message to log
    msg = ' - plot continuum fit: {0} {1}'.format(tag.capitalize(), id)
    log.message(msg, with_header=False)

    # Set plotting dictionary: continuum_fit
    plot_dict = work['plotting']['continuum_fit'].copy()

    # Set figure size and layout
    plt.figure(1, figsize=plot_dict['figsize'], tight_layout=True)
    # Set subplot
    ax = plt.subplot(1, 1, 1)

    # Update xy limits in plotting dictionary:
    # - xlim
    plot_dict['xlim']['min'] = xarr.min()
    plot_dict['xlim']['max'] = xarr.max()
    # - ylim
    plot_dict['ylim']['min'] = yarr.max()
    plot_dict['ylim']['max'] = yarr.max()
    # Set plot axis properties
    set_plot_axis_properties(ax, **plot_dict)

    # Plot spectrum - full
    ax.plot(xarr, yarr, **plot_dict['spectrum'])
    # Plot spectrum - clipped
    ax.plot(xarr, ysig, **plot_dict['clip'])
    # Plot continuum fit
    ax.plot(xarr, cf(xarr), **plot_dict['fit'])
    # Plot spectrum: continuum fit subtracted
    ax.plot(xarr, yarr - cf(xarr), **plot_dict['subtract'])
    # Plot horizontal zero point line
    ax.axhline(y=0, **plot_dict['zero_line'])
    # Set legend
    set_legend(ax, plot_dict)

    # Set title dictionary
    title_dict = {'file': work['file'],
                  **work['continuum'][key]['fit'],
                  'tag': tag.capitalize(),
                  'id': id}
    # Set title
    title = plot_dict['title_frmt'].format(**title_dict)
    plt.title(title, fontsize=11, y=plot_dict['title_ypos'])

    # Show figure (always)
    plt.show()
    # Clear current figure
    plt.clf()
    # Close current figure window (if any)
    plt.close()

    return

# ---------------------------------------------------------------------------- #
def set_plot_axis_properties(ax, **plot_dict):
# ---------------------------------------------------------------------------- #

    # Set x label
    if plot_dict['xlabel']: ax.set_xlabel(plot_dict['xlabel'])
    # Set y label
    if plot_dict['ylabel']: ax.set_ylabel(plot_dict['ylabel'])
    # Set x limits
    xmin = plot_dict['xlim']['min'] * plot_dict['xlim']['min_factor']
    xmax = plot_dict['xlim']['max'] * plot_dict['xlim']['max_factor']
    ax.set_xlim([xmin, xmax])
    # Set y limits
    ymin = plot_dict['ylim']['min'] * plot_dict['ylim']['min_factor']
    ymax = plot_dict['ylim']['max'] * plot_dict['ylim']['max_factor']
    ax.set_ylim([ymin, ymax])
    # Set minor ticks on
    ax.minorticks_on()
    # Set ticks position
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    return

# ---------------------------------------------------------------------------- #
def set_legend(ax, plot_dict):
# ---------------------------------------------------------------------------- #

    # Get legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Check if legend order is a list
    if isinstance(plot_dict['legend_order'], list):
        # Set legend order list
        legend_order = plot_dict['legend_order']

    # ... Else if alphabetic
    elif plot_dict['legend_order'] == 'alphabetic':
        # Set alphabetic legend order list
        legend_order = [labels.index(l) for l in sorted(set(labels))]

    # Re-order display sequence
    handles, labels = ([handles[i] for i in legend_order],
                       [labels[i] for i in legend_order])

    # Set legend
    ax.legend(handles, labels, **plot_dict['legend'])

    return

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
class SALTError(Exception):
# ---------------------------------------------------------------------------- #

    '''Basic exception'''
    pass

# ---------------------------------------------------------------------------- #