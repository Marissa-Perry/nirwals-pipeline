# ---------------------------------------------------------------------------- #
"""
SALT NIRWALS science pipeline:
- nirwalsprepare provides functionality for:
  - preparing and adding continuum fit to combined flat exposures
  - preparing / updating NIRWALS fibres config from combined flat exposures
"""
# ---------------------------------------------------------------------------- #

# Standard library imports
import os
import sys
import glob
from datetime import datetime

# numpy import
import numpy as np
# astropy import
from astropy.io import fits
# matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Local application imports
# - saltutility.keys
from .....saltutility.keys import key_values
from .....saltutility.keys import key_values_id
# - saltutility.files
from .....saltutility.files import load_json_file
from .....saltutility.files import dump_json_file
# - saltutility.logging
from .....saltutility.logging import logging

# Application imports
# - spectroscopy.ifu
from ...spectroscopy.ifu import find_fibres
from ...spectroscopy.ifu import set_fibre_traces
from ...spectroscopy.ifu import trace_fibres
# - functions
from ...functions import Fit1D

# ---------------------------------------------------------------------------- #

MYNAME = 'nirwalsprepare'

# ---------------------------------------------------------------------------- #

# Primary fits extension
PRIMARY = 'Primary'
# Science fits extension
SCI = 'SCI'
# Continuum Fit fits extension
FIT = 'FIT'
# Observation log file
OBSLOG = '{0}{1}OBSLOG.fits'

# ---------------------------------------------------------------------------- #
def prepare_data(obs_date, log_file, **kwargs):
# ---------------------------------------------------------------------------- #

    """
    Do pipeline preparation of flat field exposures for the specified
    observation date.

    The preparation includes (only if flat field exposures exist):

    - Prepare continuum fit image.

    - Prepare / update NIRWALS fibres config:
      - find fibre positions in images;
      - update fibres config.

    obs_date    : Observation date [CCYYMMDD] <str>
    log_file    : Log file <str>
    kwargs      : Various keyword arguments <dict>
    """

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

    # Set with stdout indicator
    with_ = kwargs['with_stdout']
    # Set only stdout indicator
    only_ = kwargs['only_stdout']

    # Start log
    with logging(log_file, with_stdout=with_, only_stdout=only_) as log:

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

            # Call action procedure dynamically
            getattr(me, action['name'])(obs_date, log, **kwargs)

    return

# ---------------------------------------------------------------------------- #
def generate_bpm(obs_date, log, **kwargs):
# ---------------------------------------------------------------------------- #
    '''
    Generate BPM from current night of observation.
    Add BPM to each product file header.
    Check that "AR-ANGLE" header exists in product files.
    '''
    # Set local variables for keyword arguments:
    # - work directory
    work_dir = kwargs['work_dir']
    # Set instrument parameters:
    # - data work folder
    folder = kwargs['params']['data_dir']
    # Set product data work directory for instrument
    prd_dir = os.path.join(work_dir, '{0}/product'.format(folder))
    # Get primary-reduced data files with wildcard in product data directory
    wildcard = '*{0}*reduced.fits'.format(obs_date)
    data_files = sorted(glob.glob(os.path.join(prd_dir, wildcard)))

    # Set instrument work directory
    wrk_dir = kwargs['params']['wrk_dir']
    # Set bpm work subdirectory
    sub_dir = kwargs['params']['sub_dirs']['bpm']
    # Set bpm directory
    bpm_dir = os.path.join(work_dir, folder, wrk_dir, sub_dir)
    # Make bpm directory
    os.makedirs(bpm_dir, exist_ok=True)

    dark_files = []
    dark_data = []
    for file in data_files:
        with fits.open(file) as hdul:
            exp_type = hdul['PRIMARY'].header['EXPTYPE']

            if exp_type == "Dark":
                dark_files.append(file)
                dark_data.append(hdul['SCI'].data)

    dark_data = np.array(dark_data)
    if len(dark_data) == 0:
        raise RuntimeError(f'No dark files found for {obs_date}')
    # average the darks
    master_dark = np.median(dark_data, axis=0)  

    # Set preparation config file
    config_file = os.path.join(kwargs['config_dir'], kwargs['params']['config'])
    # Load preparation config (JSON input file)
    config = load_json_file(config_file)
    # Set bpm threshold from config
    bpm_thresh = config['bpm']['threshold']['value']

    # initialise bpm 
    bpm = np.zeros(master_dark.shape, dtype=np.float32)
    # set good = 0, bad = 1 based on threshold
    bpm[master_dark > bpm_thresh] = 1.
    bad = len(bpm[bpm == 1])
    perc_bad = (bad / bpm.size) * 100

    # diagnostic plot for bpm threshold value
    master_dark_arr = master_dark.flatten()
    zoom_master_dark_arr = master_dark_arr[master_dark_arr < 30]
    plt.figure(figsize=(8,5))
    plt.hist(zoom_master_dark_arr, bins=200, color='black', alpha=0.9)
    plt.axvline(bpm_thresh, color='red', linestyle='dotted', zorder=5, label=f'{perc_bad:.0f}% bad pixels, threshold={bpm_thresh}')
    plt.yscale('log') 
    plt.xlim(-10,30)
    plt.xlabel('counts / s', fontsize=14, labelpad=15)
    plt.ylabel('# of pixels', fontsize=14, labelpad=15)
    plt.legend(fontsize=14, loc='upper right')
    # Set png file
    png_file = os.path.join('plots','bpm_threshold.png')
    # Add bpm directory path to png file
    filepath = os.path.join(bpm_dir, png_file)
    # Save plot as png
    plt.savefig(filepath, dpi=500, format='png', bbox_inches="tight")
    plt.close()

    # diagnostic plot for bpm image
    plt.figure(figsize=(10,5))
    plt.title(f'bpm, {perc_bad:.0f}% bad, threshold={bpm_thresh}', fontsize=12, pad=15)
    plt.imshow(bpm, origin='lower', cmap='Greys_r', vmin=0, vmax=1) 
    # Set png file
    png_file = os.path.join('plots','bpm_image.png')
    # Add bpm directory path to png file
    filepath = os.path.join(bpm_dir, png_file)
    # Save plot as png
    plt.savefig(filepath, dpi=500, format='png', bbox_inches="tight")
    plt.close()

    filename = f'NIRWALSBpm1x1_thresh_{bpm_thresh}.fits'
    filepath = os.path.join(bpm_dir, filename)
    print(f"\nWriting out BPM to: {filepath}", end='\n\n')
    # set fits extensions
    primary = fits.PrimaryHDU()
    bpm_hdu = fits.ImageHDU(data=bpm.astype(np.float32), name='BPM')
    # Add header keywords
    bpm_hdu.header['COMMENT'] = f'Bad-pixel mask generated from median dark frame of {obs_date} observation. Threshold value of {bpm_thresh} results in {perc_bad:.0f}% bad pixels.'
    # write out
    hdul = fits.HDUList([primary, bpm_hdu])
    hdul.writeto(filepath, overwrite=True)

    # Get all fits file(s) in product data directory
    wildcard = '*{0}*.fits'.format(obs_date)
    prd_files = sorted(glob.glob(os.path.join(prd_dir, wildcard)))
    for prd_file in prd_files:
        basename = os.path.basename(prd_file)
        try:
            
            with fits.open(prd_file, mode='update') as hdul:
                # ------- add "BPM" hdu -------
                hdu_names = [hdu.name for hdu in hdul]
                # remove existing BPM extension if present
                if 'BPM' in hdu_names:
                    bpm_index = hdu_names.index('BPM')
                    hdul.pop(bpm_index)
                # create new BPM extension
                bpm_hdu = fits.ImageHDU(data=bpm, name='BPM')
                bpm_hdu.header['COMMENT'] = f'Bad-pixel mask generated from median dark frame of {obs_date} observation. Threshold value of {bpm_thresh} results in {perc_bad:.0f}% bad pixels.'
                # append updated BPM
                hdul.append(bpm_hdu)
                # -----------------------------
                # ----- check AR-ANGLE hdu -------
                header = hdul['PRIMARY'].header
                header_keys = list(header.keys())
                # add AR-ANGLE if missing (use CAMANG value)
                if 'AR-ANGLE' not in header_keys and 'CAMANG' in header_keys:
                    header['AR-ANGLE'] = (header['CAMANG'], 'Articulation angle [degrees] (copied from CAMANG)')
                # --------------------------------

        except Exception as e:
            print(f"Error processing {basename}: {e}")

    return


# ---------------------------------------------------------------------------- #
def set_dark_file(hdulist, prd_dir, prefix, obs_date, subtract=True):
# ---------------------------------------------------------------------------- #
    """
    Find master dark frame with the same exposure time as current image.
    """

    # check if dark subtraction
    if not subtract:
        return None

    # get exposure time of current frame
    exp_time = hdulist[PRIMARY].header['EXPTIME']
    # Set wildcard for master dark file(s)
    wildcard = '{0}{1}Dark*.fits'.format(prefix, obs_date)
    # Add product data directory to wildcard
    wildcard = os.path.join(prd_dir, wildcard)
    # Get master dark file(s)
    dark_files = sorted(glob.glob(wildcard))

    if not dark_files:
        raise FileNotFoundError('No master dark files found with wildcard: {0}'.format(wildcard))

    for dark_file in dark_files:
        with fits.open(dark_file, mode='readonly') as dark_hdu:
            # get exposure time of dark frame
            dark_exp_time = dark_hdu[PRIMARY].header['EXPTIME']
            # match by exposure time
            if dark_exp_time == exp_time:
                return dark_file

    raise ValueError('No matching master dark found for EXPTIME={0} using wildcard {1}'.format(exp_time, wildcard))


# ---------------------------------------------------------------------------- #
def dark_subtract_flat(hdulist, dark_file, log=None):
# ---------------------------------------------------------------------------- #
    """
    Subtract master dark from the master flat
    """

    # ensure we don't subtract twice
    if hdulist[PRIMARY].header.get('DARKSUB', False):
        log.message(' - already dark subtracted', with_header=False)
        # 'Beautify' log
        log.message('', with_header=False)
        return

    with fits.open(dark_file, mode='readonly') as dark_hdu:
        dark = dark_hdu[SCI].data.copy()

    image = hdulist[SCI].data.copy()

    hdulist[SCI].data = image - dark

    hdulist[PRIMARY].header['DARKSUB'] = (True, 'Master dark subtracted')
    hdulist[PRIMARY].header['DARKFILE'] = (os.path.basename(dark_file),'Master dark file')

    log.message(' - subtract dark: {0}'.format(os.path.basename(dark_file)), with_header=False)
    # 'Beautify' log
    log.message('', with_header=False)
    return

# ---------------------------------------------------------------------------- #
def prepare_flat_fit(obs_date, log, **kwargs):
# ---------------------------------------------------------------------------- #

    # Set local variables for keyword arguments:
    # - work directory
    work_dir = kwargs['work_dir']
    # Set instrument parameters:
    # - data work folder
    folder = kwargs['params']['data_dir']
    # - instrument file (raw) prefix
    prefix = kwargs['params']['raw_prefix']
    # Set product data work directory for instrument
    prd_dir = os.path.join(work_dir, '{0}/product'.format(folder))
    # Get combined flat field file(s) with wildcard in product data directory
    wildcard = '{0}{1}Flat*.fits'.format(prefix, obs_date)
    flat_files = sorted(glob.glob(os.path.join(prd_dir, wildcard)))

    # Add message to log
    msg = '{0} -- Prepare flat field continuum fit:\n'
    log.message(msg.format(MYNAME.upper()))

    # If no flat field files exist...
    if len(flat_files) == 0:
        # Add message to log
        log.message(' No combined flat(s)!\n')
        return

    # Set fibre preparation config file
    config_file = os.path.join(kwargs['config_dir'], kwargs['params']['config'])
    # Load fibre preparation config (JSON input file)
    config = load_json_file(config_file)

    # Loop for flat field files...
    for flat_file in flat_files:

        # Set flat field base name (without full path)
        flat = os.path.basename(flat_file)
        # Add message to log
        msg = ' Combined flat: {0}\n'.format(flat)
        log.message(msg, with_header=False)

        # Open flat field file
        with fits.open(flat_file, mode='append') as hdulist:
        
            ########### dark subtraction of master flat #########
            # check if dark subtraction set in config
            dark_cfg = config.get('dark', {})
            dark_subtract = dark_cfg.get('subtract', False)

            dark_file = set_dark_file(hdulist, prd_dir, prefix, obs_date, subtract=dark_subtract)
            if dark_file is None:
                dark_subtract_flat(hdulist, dark_file, log)
            #####################################################

            # Check if FITEXT already exists
            if 'FITEXT' in hdulist[SCI].header:
                # Check if recalc is not needed
                if not kwargs['params']['recalc']:
                    # Add message to log
                    msg = ' - already prepared flat fit\n'
                    log.message(msg, with_header=False)
                    # Get outa here, bro!
                    return

            try:
                # Delete Continuum Fit extension (if it exists)
                del hdulist[FIT]; del hdulist[SCI].header['FITEXT']

            except:
                pass

            # Set 2D image array
            image = hdulist[SCI].data

            # Add message to log
            msg = ' - fit continuum'
            log.message(msg, with_header=False)
            # Fit continuum
            continuum = fit_continuum(image, config)
            # Copy SCI hdu header
            header = hdulist[SCI].header.copy()
            # Create continuum fit hdu
            hdu = fits.ImageHDU(data=continuum, header=header, name=FIT)
            # Add continuum fit hdu to hdu list
            hdulist.append(hdu)
            # Set FITEXT in SCI hdu header
            hdulist[SCI].header['FITEXT'] = len(hdulist) - 1

            # Add message to log
            msg = ' - write updated fits to file'
            log.message(msg, with_header=False)
            # Write fits file
            hdulist.writeto(flat_file, overwrite=True)

        # 'Beautify' log
        log.message('', with_header=False)

    return

# ---------------------------------------------------------------------------- #
def fit_continuum(image, config):
# ---------------------------------------------------------------------------- #

    # Set nr of image rows and columns
    rows, cols = image.shape
    # Initialise continuum array
    continuum = np.zeros((rows, cols), dtype=np.float32)
    # Set 1D columns array
    p = np.arange(cols, dtype=np.float32)
    # Set continuum fit dictionary
    continuum_fit = config['flat']['continuum']['fit']
    # Set continuum columns range
    continuum_cols = config['flat']['continuum']['cols']
    # Set columns mask
    cm = (p > continuum_cols[0]) * (p < continuum_cols[1])

    # Loop for image rows...
    for i in range(rows):

        # Set row flux
        f = image[i, :].copy()

        # Check if zero row flux
        if f[f == 0].size == cols:
            # Move on!
            continue

        # Fit row flux
        c_fit = Fit1D(p[cm], f[cm], **continuum_fit)
        # Set row continuum in continuum array
        continuum[i] = c_fit(p)
####>
        # plt.figure(1, figsize=(16, 9), tight_layout=True)
        # plt.plot(p, f, lw=1, ls='-', c='black')
        # plt.plot(p[mask], f[mask], lw=0.75, ls='-', c='cyan')
        # plt.plot(p, c_fit(p), lw=0.75, ls='--', c='red')
        # plt.xlabel('Pixel')
        # plt.ylabel('Flux')
        # plt.title('Row: {0}'.format(i+1))
        # plt.show()
        # plt.close()
####<
    return continuum

# ---------------------------------------------------------------------------- #
def prepare_fibres_config(obs_date, log, **kwargs):
# ---------------------------------------------------------------------------- #

    # Set local variables for keyword arguments:
    # - work directory
    work_dir = kwargs['work_dir']
    # Set instrument parameters:
    # - data work folder
    folder = kwargs['params']['data_dir']
    # - instrument file (raw) prefix
    prefix = kwargs['params']['raw_prefix']
    # Set product data work directory for instrument
    prd_dir = os.path.join(work_dir, '{0}/product'.format(folder))
    # Get combined flat field file(s) with wildcard in product data directory
    wildcard = '{0}{1}Flat*.fits'.format(prefix, obs_date)
    flat_files = sorted(glob.glob(os.path.join(prd_dir, wildcard)))

    # Add message to log
    msg = '{0} -- Prepare fibre config entries:\n'
    log.message(msg.format(MYNAME.upper()))

    # If no flat field files exist...
    if len(flat_files) == 0:
        # Add message to log
        log.message(' No combined flat(s)!\n')
        return

    # Set local variable(s) from keyword arguments:
    # - config directory
    config_dir = kwargs['config_dir']

    # Set fibre preparation config file
    config_file_p = os.path.join(config_dir, kwargs['params']['config'])
    # Load fibre preparation config (JSON input file)
    config_p = load_json_file(config_file_p)

    # Set observation date in prepare config
    config_p['obs_date'] = obs_date
    # Set recalc indicator in prepare config
    config_p['recalc'] = kwargs['params']['recalc']

    # Set instrument work directory
    wrk_dir = kwargs['params']['wrk_dir']
    # Set fibres work subdirectory
    sub_dir = kwargs['params']['sub_dirs']['fibres']
    # Set output directory
    output_dir = os.path.join(work_dir, folder, wrk_dir, sub_dir)
    # Set output directory in work dictionary
    config_p['output']['dir'] = output_dir
    # Make output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set db fibre traces config file
    config_file = os.path.join(config_dir, config_p['fibre_traces_db'])
    # Update db fibre traces config file in prepare config
    config_p['fibre_traces_db'] = config_file
    # Set work fibre traces config file
    config_file = os.path.join(folder, wrk_dir, config_p['fibre_traces_wrk'])
    # Update work fibre traces config file in prepare config
    config_p['fibre_traces_wrk'] = config_file

    # Prepare / update fibres config entries
    config_f, updated = prepare_fibres_config_entries(config_p, flat_files, log)

    # Check if updated
    if updated:
        # Dump updated config to file
        dump_fibres_config(config_p, config_f, log)

    # 'Beautify' log
    log.message('', with_header=False)

    return

# ---------------------------------------------------------------------------- #
def prepare_fibres_config_entries(config_p, flat_files, log):
# ---------------------------------------------------------------------------- #

    # Initialise fibres config dictionary
    config_f = {}
    # Load db fibre_traces config (JSON input file) dictionary
    config_f['db'] = load_json_file(config_p['fibre_traces_db'])
    # Initialise work fibre_traces config dictionary
    config_f['wrk'] = {
        'obs_date': config_p['obs_date'],
        'traces': {},
        'added': ''
    }

    # Set observation date
    obs_date = config_p['obs_date']
    # Set recalc indicator
    recalc = config_p['recalc']

    # Initialise updated indicator
    updated = False
    # Set trace point tolerance
    tpt = config_p['trace']['point_tolerance']

    # Loop for flat field files...
    for flat_file in flat_files:

        # Set work dictionary for preparing fibres config
        work = config_p.copy()
        # Set flat field base name (without full path) in work dictionary
        work['flat']['reference'] = os.path.basename(flat_file)
        # Set output base name
        out_name = os.path.splitext(work['flat']['reference'])[0]
        # Add message to log
        msg = ' Combined flat: {0}\n'.format(work['flat']['reference'])
        log.message(msg, with_header=False)

        # Open flat field file
        with fits.open(flat_file, mode='readonly') as hdulist:

            # Set dispersion axis (0=vertical, 1=horizontal)
            axis = hdulist[PRIMARY].header['DISPAXIS']
            # Set 2D image array (flat field fit)
            image = hdulist[FIT].data # image = hdulist[SCI].data
            # Set nr of image rows (m) and columns (n) in work dictionary
            work['rows'], work['cols'] = image.shape
            # Set exposure config id in work dictionary
            work['config'] = set_exposure_config_id(hdulist, work)

        # Set latest db traces dictionary
        db_traces = config_f['db']['traces']

        # Check if work config id is in existing fibre traces
        if work['config'] in db_traces.keys():
            # Check if already updated for observation date
            if (not recalc and obs_date in db_traces.values()):
                # Add message to log
                msg = (' - {0} already updated for observation date: {1}'
                       ).format(work['config'], obs_date)
                log.message(msg, with_header=False)
                # Move on!
                continue

        # Find fibres in image
        fibres, windows, profiles = find_fibres(image, axis, work, log)
        # Plot fibre profiles
        plot_fibre_profiles(out_name, windows, profiles, work, log)

        # Loop for fibres...
        for fibre in fibres.values():

            # Get median of fibre ranges
            med_range = list(np.median(np.array(fibre['ranges']), axis=0))

            # Loop for fibre ranges...
            for i, r in enumerate(fibre['ranges']):

                # Check if lower range is within trace point tolerance
                if med_range[0] - tpt < r[0] < med_range[0] + tpt:
                    # Okidoki! All hunky-dori...
                    pass 

                else:
                    # Override fibre ranges entry as a negative entry
                    # NOTE: Only GREATER THAN zero values are used in trace fit
                    fibre['ranges'][i] = [-1., -1.]

        try:
            # Set fibre traces dictionary
            traces = set_fibre_traces(config_f, work, log)
            # Trace fibres
            success, traces = trace_fibres(traces, fibres, windows, work, log)

        except Exception as error:
            # Raise error
            raise error

        # Check trace success indicator
        if success:
            # Update fibre traces in fibres config
            updated = update_fibres_config(updated, config_f, traces, work, log)
            # Plot fibre traces
            plot_fibre_traces(out_name, fibres, windows, traces, work, log)
            # # Dump output
            # dump_output(fibres, windows, profiles, traces, work, log)

        # 'Beautify' log
        log.message('', with_header=False)

    return config_f, updated

# ---------------------------------------------------------------------------- #
def set_exposure_config_id(hdulist, work):
# ---------------------------------------------------------------------------- #

    # Get exposure config key value dictionary
    key_dict = key_values(hdulist, work['config_key']['list'])
    # Set exposure config id in work dictionary
    config_id = key_values_id(work['config_key']['frmts'], key_dict)

    return config_id

# ---------------------------------------------------------------------------- #
def update_fibres_config(updated, config_f, traces, work, log):
# ---------------------------------------------------------------------------- #

    # Check if exposure config id is in existing fibre traces
    if work['config'] in config_f['wrk']['traces'].keys():
        # Nothing to do... Return updated indicator as is.
        return updated

    else:
        # Add message to log
        log.message(' - update fibres config', with_header=False)
        # Add current fibre traces to existing traces
        config_f['wrk']['traces'][work['config']] = traces
        # Set time added
        config_f['added'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        # Return True for updated indicator
        return True

# ---------------------------------------------------------------------------- #
def dump_output(fibres, windows, profiles, traces, work, log):
# ---------------------------------------------------------------------------- #

    # Check output directory
    if not work['output']['dir']: return

    # Add message to log
    msg = ' - dump output'
    log.message(msg, with_header=False)

    # Set output base name
    out_name = os.path.splitext(work['flat']['reference'])[0]
    # Set fibres output file name (JSON file)
    out_file = '{0}.json'.format(out_name)
    # Add output directory to fibres output file name
    out_file = os.path.join(work['output']['dir'], out_file)
    # Add fibres, windows, profiles and traces to 1 dictionary
    out = {
        'fibres': fibres,
        'windows': windows,
        'profiles': profiles,
        'traces': traces
    }
    # Dump fibres output to file
    dump_json_file(out, out_file)

    return

# ---------------------------------------------------------------------------- #
def dump_fibres_config(config_p, config_f, log):
# ---------------------------------------------------------------------------- #

    # Check work fibre traces config
    if config_f['wrk']['traces']:
        # Add message to log
        msg = ' Dump work fibre traces config'
        log.message(msg, with_header=False)
        # Dump work fibre traces config to file
        dump_json_file(config_f['wrk'], config_p['fibre_traces_wrk'])

    # Set existing db fibre traces
    db_traces = config_f['db']['traces']
    # Initialise db update indicator: False
    db_update = False

    # Loop for work fibre traces...
    for config in config_f['wrk']['traces'].keys():

        # Check if config id is not in existing db 
        if config not in db_traces.keys():
            # Initialise 'new' config entry in db 
            db_traces[config] = []

        # Check if observation date is not in existing date list
        if config_p['obs_date'] not in db_traces[config]:
            # Add observation date to date list
            db_traces[config].append(config_p['obs_date'])
            # Sort date list (in place)
            db_traces[config].sort()
            # Set db update indicator: True
            db_update = True

    # Check if db fibre traces config dump is necessary
    if db_update:
        # Add message to log
        msg = ' Dump db fibre traces config'
        log.message(msg, with_header=False)
        # Dump db wavelength solutions config to file
        dump_json_file(config_f['db'], config_p['fibre_traces_db'])

    return

# ---------------------------------------------------------------------------- #
def dump_fibres_config_(obs_date, config_file, config, configs, config_dir, log):
# ---------------------------------------------------------------------------- #

    # Set config file name for current observation date
    config_file = os.path.join(config_dir, configs['fibres'].format(obs_date))
    # Set current observation date
    config['obs_date'] = obs_date
    # Set time added
    added = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    config['added'] = added

    # Add message to log
    msg = ' Dump updated fibres config'
    log.message(msg, with_header=False)
    # Dump fibres config to file
    dump_json_file(config, config_file)

    # Check if db file update is necessary:
    # - set fibres db file name
    db_file = configs['fibres_db']
    # - add config directory path to file name
    db_file = os.path.join(config_dir, db_file)
    # - load wavelength solutions db (JSON input file)
    db = load_json_file(db_file)
    # - filter db entries for == observation date
    if list(filter(lambda e: e['date'] == obs_date, db['entries'])):
        # - already in db file! Get outa here!
        pass

    else:
        # - set db entry
        entry = {
            'date': obs_date,
            'file': configs['fibres'].format(obs_date),
            'note': 'Added: {0}'.format(added)
        }
        # - add to db entries
        db['entries'].append(entry)

        # - add message to log
        msg = ' Dump updated fibres db'
        log.message(msg, with_header=False)
        # - dump fibres db to file
        dump_json_file(db, db_file)

    return

# ---------------------------------------------------------------------------- #
def plot_fibre_profiles(out_name, windows, profiles, work, log):
# ---------------------------------------------------------------------------- #

    # Check output indicator
    if not work['output']['plot_fibre_profiles']: return

    # Add message to log
    msg = '   - plot fibre profiles'
    log.message(msg, with_header=False)

    # NOTE: Plot fibre profiles

    # Initialise 1D row pixel array
    rows = np.arange(work['rows'], dtype=np.float32)

    # Loop for windows and profiles...
    for window, profile in zip(windows, profiles):

        # Set window range
        r0, r1 = window[0] + 1, window[1]
        # Set x and y labels dictionary
        labels = {'x': 'row', 'y': 'columns: {0} to {1}'.format(r0, r1)}

        # Set plotting dictionary: fibre_profiles
        plot_dict = work['plotting']['fibre_profiles'].copy()

        # Set figure size and layout
        plt.figure(1, figsize=plot_dict['figsize'], tight_layout=True)
        # Set subplot
        ax = plt.subplot(1, 1, 1)

        # Update plotting dictionary:
        # - xlabel
        plot_dict['xlabel'] = plot_dict['xlabel'].format(**labels)
        # - ylabel
        plot_dict['ylabel'] = plot_dict['ylabel'].format(**labels)
        # - xlim
        plot_dict['xlim']['min'] = rows.min()
        plot_dict['xlim']['max'] = rows.max()
        # - ylim
        plot_dict['ylim']['min'] = np.array(profile).max()
        plot_dict['ylim']['max'] = np.array(profile).max()
        # Set plot axis properties
        set_plot_axis_properties(ax, **plot_dict)

        # Plot profile
        ax.plot(rows, profile, **plot_dict['profile'])
        # Set legend
        set_legend(ax, **plot_dict)

        # Set title
        title = plot_dict['title_frmt'].format(**labels)
        plt.title(title, fontsize=11, y=plot_dict['title_ypos'])

        # Save figure
        if plot_dict['save']:
            # Set visualisation output file name (PNG file)
            out_file = '{0}_profile_{1}_{2}.png'.format(out_name, r0, r1)
            # Add output directory to output file name
            out_file = os.path.join(work['output']['dir'], out_file)
            # Save plot as png
            plt.savefig(out_file, dpi=180, format='png')

        # Show figure
        if plot_dict['show']: plt.show()
        # Clear current figure
        plt.clf()
        # Close current figure window (if any)
        plt.close()

    return

# ---------------------------------------------------------------------------- #
def plot_fibre_traces(out_name, fibres, windows, traces, work, log):
# ---------------------------------------------------------------------------- #

    # Check output indicator
    if not work['output']['plot_fibre_traces']: return

    # Add message to log
    msg = '   - plot fibre traces'
    log.message(msg, with_header=False)

    # NOTE: Plot fibre traces

    # Initialise mid-points list
    mid_points = []

    # Loop for windows...
    for window in windows:

        # Add window mid-point to mid-points list
        mid_points.append(0.5 * (window[0] + window[1]))

    # Convert mid-points list to array
    mid_points = np.array(mid_points)

    # Set labels dictionary
    lbls = {
        'actual': ['Lower (actual)', 'Upper (actual)'],
        'fitted': ['Lower (fitted)', 'Upper (fitted)']
    }
    # Set colours list (red, blue)
    clrs = ['r', 'b']

    # Initialise 1D column pixel array
    cols = np.arange(work['cols'], dtype=np.float32)

    # Loop for nr of fibre blocks (start at 1)...
    for block in range(1, work['nr_of_blocks'] + 1):

        # Set plotting dictionary: fibre_traces
        plot_dict = work['plotting']['fibre_traces'].copy()

        # Set figure size and layout
        plt.figure(1, figsize=plot_dict['figsize'], tight_layout=True)
        # Set subplot
        ax = plt.subplot(1, 1, 1)

        # Update plotting dictionary:
        # - xlim
        plot_dict['xlim']['min'] = cols.min()
        plot_dict['xlim']['max'] = cols.max()
        # Set plot axis properties
        set_plot_axis_properties(ax, **plot_dict)

        # Set x-axis major and minor ticks
        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        # Set y-axis major and minor ticks
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.yaxis.set_minor_locator(MultipleLocator(5))

        # Add secondary y-axis
        axy2 = ax.secondary_yaxis('right')
        # Set secondary y-axis label
        axy2.set_ylabel('Slit Id')

        # Initialise legend labels list
        labels = []
        # Initialise secondary y-axis ticks and tick labels
        yticks, yticklabels = [], []

        # Loop for fibres and traces...
        for f, t in zip(fibres.values(), traces['fibres'].values()):

            # Check if current trace is not for block... and move on
            if t['block'] != block: continue

            # Set traces (actual):
            # - lower trace
            al = [r[0] for r in f['ranges']]
            # - upper trace
            au = [r[1] for r in f['ranges']]

            # Set traces (fitted):
            # - lower trace fit
            lt = Fit1D([], [], coef=t['lower_trace'], **work['trace']['fit'])
            # - lower trace
            fl = lt(cols)
            # - upper trace fit
            ut = Fit1D([], [], coef=t['upper_trace'], **work['trace']['fit'])
            # - upper trace
            fu = ut(cols)

            # Plot actual trace:
            for at, l, c in zip([al, au], lbls['actual'], clrs):
                # - check if label is in legend labels list
                if l in labels:
                    # - override label
                    l = ''
                else:                    
                    # - add label to labels list
                    labels.append(l)
                # - update plotting dictionary
                plot_dict['actual']['label'] = l
                plot_dict['actual']['color'] = c
                # - convert trace points list to array
                at = np.array(at)
                # - set mask for greater than zero trace points
                gz = at > 0.
                # - plot
                ax.scatter(mid_points[gz], at[gz], **plot_dict['actual'])

            # Plot fitted trace:
            for ft, l, c in zip([fl, fu], lbls['fitted'], clrs):
                # - check if label is in legend labels list
                if l in labels:
                    # - override label
                    l = ''
                else:                    
                    # - add label to labels list
                    labels.append(l)
                # - update plotting dictionary
                plot_dict['fit']['label'] = l
                plot_dict['fit']['color'] = c
                # - plot
                ax.plot(cols, ft, **plot_dict['fit'])

            # Fill area between lower and upper trace
            ax.fill_between(cols, fl, fu, **plot_dict['fill'])
            # Add secondary y-axis tick and tick label
            yticks.append((fl[-1]+fu[-1])/2.); yticklabels.append(t['slit_id'])

        # Set secondary y-ticks
        axy2.set_yticks(yticks)
        # Set secondary y-tick labels
        axy2.set_yticklabels(yticklabels)

        # Update legend[ncol] in plotting dictionary
        plot_dict['legend']['ncol'] = len(labels)
        # Set legend
        set_legend(ax, **plot_dict)

        # Set title
        title = plot_dict['title_frmt'].format(block)
        plt.title(title, fontsize=11, y=plot_dict['title_ypos'])

        # Save figure
        if plot_dict['save']:
            # Set visualisation output file name (PNG file)
            out_file = '{0}_traces_block_{1:d}.png'.format(out_name, block)
            # Add output directory to output file name
            out_file = os.path.join(work['output']['dir'], out_file)
            # Save plot as png
            plt.savefig(out_file, dpi=180, format='png')

        # Show figure
        if plot_dict['show']: plt.show()
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
    if xmin != 0 or xmax != 0: ax.set_xlim([xmin, xmax])
    # Set y limits
    ymin = plot_dict['ylim']['min'] * plot_dict['ylim']['min_factor']
    ymax = plot_dict['ylim']['max'] * plot_dict['ylim']['max_factor']
    if ymin != 0 or ymax != 0: ax.set_ylim([ymin, ymax])
    # Set minor ticks on
    ax.minorticks_on()
    # Set ticks position
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    return

# ---------------------------------------------------------------------------- #
def set_legend(ax, **plot_dict):
# ---------------------------------------------------------------------------- #

    # Get legend handles and labels
    hands, labs = ax.get_legend_handles_labels()
    # Check legend order
    if plot_dict['legend_order']:
        # Re-order display sequence
        hands, labs = ([hands[i] for i in plot_dict['legend_order']],
                       [labs[i] for i in plot_dict['legend_order']])

    # Set legend
    ax.legend(hands, labs, **plot_dict['legend'])

    return

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
class SALTError(Exception):
# ---------------------------------------------------------------------------- #

    """Basic exception"""
    pass

# ---------------------------------------------------------------------------- #