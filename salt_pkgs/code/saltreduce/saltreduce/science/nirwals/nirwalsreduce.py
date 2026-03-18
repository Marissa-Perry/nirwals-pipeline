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
from astropy.stats import sigma_clip
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
    sci = hdu[SCI].data.copy()

    # Set bad pixel image
    bpm = hdu[BPM].data.copy()
    # Initialise good pixel image (same shape as bad pixel image)
    gpm = np.ones(bpm.shape, dtype=np.float32)

    # Check if good pixel mask must be applied
    if work['apply_gpm']:
        # Set good pixel image: good = 0 where bad = 1
        gpm[bpm == 1] = 0.

    # Set science image with gpm applied
    sci *= gpm

    # Set tag in work dictionary
    work['tag'] = 'main image'
    # Extract fibres for science image
    fibres_sci = extract_fibres(sci, traces, work, log)

    # Set tag in work dictionary
    work['tag'] = 'good pixel image'
    # Extract fibres for gpm image
    fibres_gpm = extract_fibres(gpm, traces, work, log)

    # Initialise flat field / continuum fit image as None
    flt = None
    # Check if flat field type (for exposure type) is 'flat' or 'fit'
    if work['flat']['type'][work['exp_type']] in ['flat', 'fit']:

        # Open flat field image file
        with fits.open(work['flat']['reference'], mode='readonly') as fltlist:

            # Set flat field and continuum fit images as needed
            if work['flat']['type'][work['exp_type']] == 'flat':
                # Set flat field image
                flt = fltlist[SCI].data.copy()

            elif work['flat']['type'][work['exp_type']] == 'fit':
                # Set flat field continuum fit image
                flt = fltlist[FIT].data.copy()

        # Set flat field or flat field continuum fit with gpm applied
        flt *= gpm

        # Set tag in work dictionary
        work['tag'] = 'flat field image'
        # Extract fibres for flat field image
        fibres_flt = extract_fibres(flt, traces, work, log)

    # Initialise extracted fibres and good pixels dictionaries
    fibres, good_pixels = {}, {}

    # Loop for science image fibre arrays...
    for id, sciarr in fibres_sci.items():

        # Set non NaN science mask
        non_nan = ~np.isnan(sciarr)
        # Set good pixel array
        gpmarr = fibres_gpm[id]

        # Check flat field image array
        if flt is not None:
            # Set flat field array
            fltarr = fibres_flt[id]
            # Linearly transform intensity scale (bscale)
            fltarr[non_nan] /= fltarr[non_nan].mean()
            # Set combined non NaN science and non zero flat mask
            mask = (non_nan) * (fltarr != 0)
            # Flat field science image array
            sciarr[mask] /= fltarr[mask]

        # Set combined non NaN science and non zero good pixel mask
        mask = (non_nan) * (gpmarr != 0)
        # Scale fibre flux for 'nr' of good pixels:
        # - divide 'sci' by 'gpm'
        sciarr[mask] /= gpmarr[mask]
        # - multiply 'sci' by 'gpm' mean
        sciarr[non_nan] *= gpmarr[non_nan].mean()

        # Add fibre flux to extracted fibres dictionary
        fibres[id] = sciarr
        # Add good pixel array to good pixels dictionary
        good_pixels[id] = gpmarr

    # Set good pixels dictionary in work dictionary
    work['good_pixels'] = good_pixels

    # Check if debug
    if work['debug']:
        # Dump extracted fibres to file
        dump_extracted_fibres(fibres, work)

    return fibres

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

    ##### DEBUGGING ####### 
    # flip flux arr
    work['f'] = work['f'][::-1]
    #######################

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


    ########## DEBUG PLOT 03/14/2026 ##############
    print('\ngenerating arc vs model diagnostic plot...\n\n')
    import matplotlib.pyplot as plt
    from scipy.signal import correlate

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # re-sample model (artificial spectrum from line list)
    # lines are narrower than observed arc resolution
    model_flux = np.interp(w, work['line_list']['aswarr'], work['line_list']['asfarr'])

    # normalize spectra
    obs = work['f'] / np.max(work['f'])
    model = model_flux / np.max(model_flux)

    ax.plot(w, obs, color='black', label='observed arc (wavelength from grating model)')
    ax.plot(w, model, color='red', linewidth=0.8, label='model')

    # lines from line list
    ax.vlines(work['line_list']['swarr'],
              ymin=min(work['f']),
              ymax=max(work['f']),
              color='grey',
              linestyles='--',
              linewidth=0.5,
              label='line list')

    ax.set_xlim(min(w),max(w))
    ax.set_ylim(-0.1,1.5)
    ax.set_xlabel('Wavelength (A)', labelpad=15)
    ax.set_ylabel('Normalized Flux', labelpad=15)
    ax.set_title(f'Arc Lamp: {lamp}', pad=15)
    ax.legend(loc='upper right')
    plot_path = os.path.join(work['output']['dir'], f"{work['file']}_arc_model_debug.png")  
    plt.savefig(plot_path, dpi=120)
    plt.close()
###############################################

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
    sf_image = fit_spectral_channels(cs_image, work, log)
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

            ##### DEBUGGING ##### 
            # flip flux arr here as well...
            f = f[::-1]
            #####################

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
        # Set continuum fit in continuum fit image array
        cf_image[i][rm] = cf(xarr)[rm]
        # Set row flux minus continuum fit in continuum subtracted image array
        cs_image[i][rm] = yarr[rm] - cf(xarr)[rm]

    # Set mask in work dictionary
    work['image_mask'] = mask
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
def fit_spectral_channels(image, work, log):
# ---------------------------------------------------------------------------- #

    # Add message to log
    msg = ' - fit spectral channels'
    log.message(msg, with_header=False)

    # Initialise new spectral fit image array
    sf_image = np.zeros(image.shape, dtype=np.float32)
    # Set 1D fitting dictionary
    fit = work['spectral_channels']['fit']

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

    # Set NaN to 1
    sf_image[np.isnan(sf_image)] = 1.

    return sf_image

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
    rat_sf_image = np.ones(sci_sf_image.shape, dtype=np.float32)
    # Set non-zero mask from sky spectral channels fit image
    nz = sky_sf_image != 0.
    # Set object to sky spectral channels fit ratio
    rat_sf_image[nz] = sci_sf_image[nz] / sky_sf_image[nz]

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

    # Set zeros
    gpm_image[sky_sf_image <= thresh] = 0.
    # Set ones
    gpm_image[sky_sf_image > thresh] = 1.

    # Clean object continuum subtracted image
    sci_cs_2d = gpm_image * sci_cs_image
    # Clean sky continuum subtracted image
    sky_cs_2d = gpm_image * sky_cs_image
    # Collapse object continuum subtracted image
    sci_cs_1d = np.sum(sci_cs_2d, axis=1)
    # Some numpy voodoo... set n rows, 1 column
    sci_cs_1d = np.array([sci_cs_1d]).transpose()
    # Collapse sky continuum subtracted image
    sky_cs_1d = np.sum(sky_cs_2d, axis=1)
    # Some numpy voodoo... set n rows, 1 column
    sky_cs_1d = np.array([sky_cs_1d]).transpose()

    # Initialise object to sky continuum subtracted ratio array (1D)
    rat_cs_1d = np.ones(sci_cs_1d.shape, dtype=np.float32)
    # Set non-zero mask from sky continuum subtracted array
    nz = sky_cs_1d != 0.
    # Set object to sky continuum subtracted ratio array (1D)
    rat_cs_1d[nz] = sci_cs_1d[nz] / sky_cs_1d[nz]
    # Normalise object to sky continuum subtracted ratio array (1D)
    rat_cs_1d /= rat_cs_1d.mean()
    # Repeat 1D object to sky ratio to same column dimension as 2D images
    rat_cs_2d = np.repeat(rat_cs_1d, rat_sf_image.shape[1], axis=1)
    # Combine normalised wavelength and position scalings
    rat_sf_image *= rat_cs_2d
    # Clean interline regions
    rat_sf_image *= gpm_image
    # Set interline regions to unity
    rat_sf_image[rat_sf_image < 1e-2] = 1.

    # Scale sky continuum subtracted image
    sky_scaled = sky_cs_image * rat_sf_image
    # Subtract scaled image from object continuum subtracted image
    sci_image = sci_cs_image - sky_scaled
    # Add continuum back in
    sci_image_with_cont = sci_image + (sci_cf_image - sky_cf_image)

    # Add sky subtracted header key
    value = time.asctime(time.localtime())
    comment = 'Image has been wavelength calibrated'
    hdu['Primary'].header['SKYSUB'] = (value, comment)

    return sci_image, sci_image_with_cont

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
        png_file = '{0}_wavelength_fit.png'.format(work['file'])
        # Add output directory path to png file
        png_file = os.path.join(work['output']['dir'], png_file)
        # Save plot as png
        plt.savefig(png_file, dpi=180, format='png')

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
        png_file = '{0}_zps_fit.png'.format(work['file'])
        # Add output directory path to png file
        png_file = os.path.join(work['output']['dir'], png_file)
        # Save plot as png
        plt.savefig(png_file, dpi=180, format='png')

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