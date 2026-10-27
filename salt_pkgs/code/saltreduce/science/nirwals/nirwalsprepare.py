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
from scipy.ndimage import median_filter
from astropy.stats import mad_std
# matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import LogLocator

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
from ...functions import ipc_correction_global
from ...functions import isolated_hot_pixels
from ...functions import build_alpha_maps
from ...primary.primary_reductions import duplicate

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
    data_files = sorted(f for f in glob.glob(os.path.join(prd_dir, wildcard)) if not duplicate(f))

    # Set instrument work directory
    wrk_dir = kwargs['params']['wrk_dir']
    # Set bpm work subdirectory
    sub_dir = kwargs['params']['sub_dirs']['bpm']
    # Set bpm directory
    bpm_dir = os.path.join(work_dir, folder, wrk_dir, sub_dir)
    # Make bpm directory
    os.makedirs(bpm_dir, exist_ok=True)

    # Set preparation config file
    config_file = os.path.join(kwargs['config_dir'], kwargs['params']['config'])
    # Load preparation config (JSON input file)
    config = load_json_file(config_file)
    # Set bpm sigma threshold factor from config
    bpm_thresh_sigma = config['bpm']['threshold']['sigma']  # using threshold set in config file
    generate = config['bpm']['generate']

    # set once the BPM is generated below, for the correction loop further down
    correct_neighbors = False
    alpha_asym_3 = None
    alpha_asym_1 = None
    measured_3 = None

    # check if BPM generation was overrided for this observation night and retrieve BPM in prd dir
    if not generate:
        log.message('   - config set to skip BPM generation.')

        try:
            bpm_file = sorted(f for f in glob.glob(os.path.join(prd_dir, f'*Bpm*{bpm_thresh_sigma}sigma.fits')) if not duplicate(f))[0]
            log.message('   - using file {0}'.format(bpm_file), with_header=False)

            with fits.open(bpm_file) as hdul:
                bpm = hdul['BPM'].data  # save pre-generated BPM data 
                comment = ' '.join(str(hdul['BPM'].header['COMMENT']).split())  # save comment from header to use in new fits
            # compute the percentage of bad pixels in this BPM
            bad = len(bpm[bpm == 1])
            perc_bad = (bad / bpm.size) * 100

        except Exception as e:
            log.message(f"No BPM found in product directory: {os.path.join(prd_dir, '*Bpm*.fits')}. \nAdd one or set config to generate one.", with_header=False)
            return None
    
    else:

        log.message('   - generating BPM.')

        # instrument (raw) file prefix — needed to locate master dark/flat files
        prefix = kwargs['params']['raw_prefix']

        # ---------- FLAGS union: from every individual reduced exposure ----------
        # per-exposure ramp-fit flags live only on the individual frames, so read
        # them here to preserve "flagged in ANY exposure that night".
        # Some observations lack a FLAGS extension entirely; those are skipped.
        if not data_files:
            raise RuntimeError(f'No reduced files found for {obs_date}')

        flag_union = None
        for file in data_files:
            with fits.open(file) as hdul:
                if 'FLAGS' not in hdul:
                    continue
                flags = hdul['FLAGS'].data
                if flag_union is None:
                    flag_union = np.zeros_like(flags)
                flag_union[flags == 1] = 1

        # ---------- locate combined masters ----------
        # dark product paths only, for bookkeeping -- their SCI data isn't read here since the
        # per-file correction loop below overwrites it in place on every run
        dark_wildcard = os.path.join(prd_dir, '{0}{1}Dark*.fits'.format(prefix, obs_date))
        dark_files = sorted(f for f in glob.glob(dark_wildcard) if not duplicate(f))
        if len(dark_files) == 0:
            raise RuntimeError(f'No combined dark files found for {obs_date}')
        product_file_by_exptime = {fits.getheader(f, PRIMARY)['EXPTIME']: f for f in dark_files}

        flat_wildcard = os.path.join(prd_dir, '{0}{1}Flat*.fits'.format(prefix, obs_date))
        flat_files = sorted(f for f in glob.glob(flat_wildcard) if not duplicate(f))
        if len(flat_files) == 0:
            raise RuntimeError(f'No combined flat files found for {obs_date}')

        flat_stack = []
        for flat_file in flat_files:
            with fits.open(flat_file) as hdulist:
                flat_stack.append(hdulist[SCI].data.copy())
        master_flat = np.median(np.array(flat_stack), axis=0)

        # initialise BPM
        bpm = np.zeros(master_flat.shape, dtype=np.float32)

        # ---------- component 1: up-the-ramp-fit flags ----------
        if flag_union is None:
            log.message('   - no FLAGS extension found in any reduced file; skipping flag component', with_header=False)
        else:
            bpm[flag_union == 1] = 1.
            bad_flag_perc = (flag_union.sum() / flag_union.size) * 100

            plt.figure(figsize=(10,5))
            plt.title(f'flagged pixels, {bad_flag_perc:.1f}% bad', fontsize=12, pad=15)
            plt.imshow(flag_union, origin='lower', cmap='Greys_r', vmin=0, vmax=1)
            # Set png file
            plot_dir = os.path.join(bpm_dir,'plots')
            os.makedirs(plot_dir, exist_ok=True)
            png_file = 'flag_image.png'
            # Add bpm directory path to png file
            filepath = os.path.join(plot_dir, png_file)
            # Save plot as png
            plt.savefig(filepath, dpi=150, format='png', bbox_inches="tight")
            plt.close()

        # ---------- component 2: master darks ------------
        # rebuilt fresh from raw exposures, grouped by EXPTIME and mean-combined -- the product
        # *Dark*.fits files get IPC-corrected in place later in this function, so a second run
        # can't use them as calibration input without compounding that correction
        raw_wildcard = os.path.join(work_dir, '{0}/raw'.format(folder), '*reduced.fits')
        raw_files = sorted(f for f in glob.glob(raw_wildcard) if not duplicate(f))
        dark_groups = {}
        for raw_file in raw_files:
            hdr = fits.getheader(raw_file, PRIMARY)
            if hdr.get('OBSTYPE') != 'Dark':
                continue
            dark_groups.setdefault(hdr['EXPTIME'], []).append(raw_file)
        if len(dark_groups) == 0:
            raise RuntimeError(f'No raw dark exposures found for {obs_date}')

        bpm_dark = np.zeros(bpm.shape, dtype=bool)
        # raw data + background stats from the longest-exptime dark (component 4 input)
        longest_dark_data = None
        longest_dark_file = None
        longest_dark_med = None
        longest_dark_sig = None
        longest_exptime = -1
        longest_n_frames = None
        plot_dark = None   # arrays kept for the longest-exposure diagnostic only

        for exptime, group_files in dark_groups.items():
            d = np.mean([fits.getdata(f, SCI) for f in group_files], axis=0)
            finite = np.isfinite(d)

            med = np.median(d[finite])
            p84 = np.percentile(d[finite], 84)   # median + 1 sigma
            sig = p84 - med
            upper = med + bpm_thresh_sigma * sig
            lower = med - bpm_thresh_sigma * sig

            flagged = finite & (d > upper) #| (d < lower))  # only upper since dark current accumulates, don't want to cut on noise floor
            bpm_dark |= flagged

            bad_dark_perc = 100 * flagged.sum() / finite.sum()
            log.message('   - dark BPM: exptime={0}s, {1:.1f}% flagged'.format(exptime, bad_dark_perc), with_header=False)

            # track the longest exposure's own data/stats separately from the union
            if exptime > longest_exptime:
                longest_exptime = exptime
                longest_dark_data = d
                longest_dark_med = med
                longest_dark_sig = sig
                longest_n_frames = len(group_files)

            # retain only the longest exposure (the most stressing case) for plotting
            if plot_dark is None or exptime > plot_dark['exptime']:
                plot_dark = dict(exptime=round(exptime), data=d[finite], med=med, upper=upper, lower=lower, perc=bad_dark_perc)

        longest_dark_file = product_file_by_exptime.get(longest_exptime)
        bpm[bpm_dark] = 1.

        # ---------- component 4: inter-pixel crosstalk correction ----------
        log.message('   - crosstalk:', with_header=False)
        correct_neighbors = config['bpm'].get('crosstalk', {}).get('correct_neighbors', {}).get('apply', True)
        alpha_asym_3 = config['bpm']['crosstalk']['correct_neighbors']['alpha_asym_3']
        alpha_asym_1 = config['bpm']['crosstalk']['correct_neighbors']['alpha_asym_1']
        if correct_neighbors:
            measured_3 = np.zeros(bpm.shape, dtype=bool)
            isolation_radius = config['bpm']['crosstalk'].get('isolation_radius', 5)
            # a pixel's own alpha is measured only where it's an isolated hot pixel -- the default alpha covers everywhere else
            per_pixel_cfg = config['bpm']['crosstalk']['correct_neighbors'].get('per_pixel_alpha', {})
            if per_pixel_cfg.get('apply', True):
                min_measured = per_pixel_cfg['min_measured']
                # isolated hot pixels for calibration are flagged at the same threshold as the BPM's own dark-current component
                alpha_threshold = longest_dark_med + bpm_thresh_sigma * longest_dark_sig
                hot_pixel, alpha_candidates, nearby_count = isolated_hot_pixels(longest_dark_data, alpha_threshold, radius=isolation_radius)
                alpha_asym_3, alpha_asym_1, alpha_sym, measured_3, measured_1, used_asym_3, used_asym_1 = build_alpha_maps(longest_dark_data, longest_dark_med, hot_pixel, alpha_candidates, nearby_count, alpha_asym_3, alpha_asym_1, min_measured=min_measured)
                if measured_3.sum() >= min_measured:
                    log.message('      - measured alpha for {0} isolated pixels ({1}-sigma hot-pixel thresh, {2}-pixel isolation radius): alpha_asym_3={3:.4f}, alpha_asym_1={4:.4f}'.format(measured_3.sum(), bpm_thresh_sigma, isolation_radius, used_asym_3, used_asym_1), with_header=False)
                else:
                    log.message('      - {0} isolated pixels ({1}-sigma hot-pixel thresh, {2}-pixel isolation radius) below config threshold of >{3} pixels. Using default config values alpha_asym_3={4:.4f}, alpha_asym_1={5:.4f}'.format(measured_3.sum(), bpm_thresh_sigma, isolation_radius, min_measured, used_asym_3, used_asym_1), with_header=False)

                alpha_calibration_diagnostic_plot(longest_dark_data, longest_dark_med, longest_dark_sig, bpm_thresh_sigma,
                                                   alpha_asym_3, alpha_asym_1, alpha_sym, measured_3, measured_1, used_asym_3, used_asym_1, longest_exptime, bpm_dir)

        # ---------- diagnostic: longest-exposure master dark only ----------
        if plot_dark is not None:
            zoom = plot_dark['data'][plot_dark['data'] < 30]
            if zoom.size:
                bin_width = 0.1
                custom_bins = np.arange(zoom.min(), zoom.max() + bin_width, bin_width)
                plt.figure(figsize=(8, 5))
                plt.title(fr"{plot_dark['exptime']}s master dark, {plot_dark['perc']:.1f}% bad pixels", fontsize=13, pad=15)
                plt.hist(zoom, bins=custom_bins, color='black', alpha=0.9)
                # plt.axvline(plot_dark['lower'], color='grey', linestyle='dotted')
                plt.axvline(plot_dark['med'], color='red', linestyle='dotted', label='median')
                plt.axvline(plot_dark['upper'], color='grey', linestyle='dotted', label=fr'{bpm_thresh_sigma}$\sigma$ threshold')
                plt.yscale('log')
                plt.xlim(-10, 30)
                plt.xlabel('counts / s', fontsize=14, labelpad=15)
                plt.ylabel('# of pixels', fontsize=14, labelpad=15)
                plt.legend(fontsize=13, loc='upper right')
                plot_dir = os.path.join(bpm_dir, 'plots')
                os.makedirs(plot_dir, exist_ok=True)
                filepath = os.path.join(plot_dir, f"master_dark_threshold_exptime{plot_dark['exptime']}.png")
                plt.savefig(filepath, dpi=150, format='png', bbox_inches='tight')
                plt.close()

        # -------------- component 3: master flat --------------
        # Flag pixels by their deviation from the smoothed structure of the master flat
        smooth_k = config['bpm']['threshold']['flat_smooth_k']
        illum_frac = config['bpm']['threshold']['flat_illum_frac']

        finite_flat = np.isfinite(master_flat)
        fill = np.median(master_flat[finite_flat])  # fill non-finite so the median filter stays defined
        smooth = median_filter(np.where(finite_flat, master_flat, fill), size=(1, smooth_k))
        core = np.percentile(smooth[np.isfinite(smooth)], 90)  # estimating the core illumination level of fibers as the 90th percentile
        illum = finite_flat & (smooth > illum_frac * core)     # estimating the rest of the fiber profile as being > 20% of this core illumination level
        # response of detector: ratio between a smoothed master flat pixels and raw master flat
        response = np.full(master_flat.shape, np.nan, dtype=float)
        response[illum] = master_flat[illum] / smooth[illum]  
        # noise-normalised deviation: (response - 1) * sqrt(counts) 
        z = np.full(master_flat.shape, np.nan, dtype=float)
        z[illum] = (response[illum] - np.median(response[illum])) * np.sqrt(np.clip(smooth[illum], 1.0, None))  #
        z_scatter = mad_std(z[illum])

        bpm_flat = np.zeros(master_flat.shape, dtype=bool)
        bpm_flat[illum] = (np.abs(z[illum]) > bpm_thresh_sigma * z_scatter)  # identifying bad pixels
        bpm[bpm_flat] = 1.
        flat_bpm_perc = 100 * bpm_flat.sum() / illum.sum()
        log.message('   - flat BPM: {0:.2f}% flagged', with_header=False)

        # ---------- diagnostic: noise-normalised flat deviation ----------
        zvals = z[illum]
        zvals = zvals[np.isfinite(zvals)]
        if zvals.size:
            lim = np.percentile(np.abs(zvals), 99.9)
            plt.figure(figsize=(8, 5))
            plt.title(fr'master flat response, {flat_bpm_perc:.2f}% bad pixels', fontsize=13, pad=15)
            plt.hist(zvals, bins=np.linspace(-lim, lim, 201), color='black', alpha=0.9)
            plt.axvline(bpm_thresh_sigma * z_scatter, color='grey', linestyle='dotted', label=fr'{bpm_thresh_sigma}$\sigma$ threshold')
            plt.axvline(-bpm_thresh_sigma * z_scatter, color='grey', linestyle='dotted')
            plt.axvline(0.0, color='red', linestyle='dotted', label='median')
            plt.yscale('log')
            plt.xlabel(r'noise-normalised deviation', fontsize=13, labelpad=15)
            plt.ylabel('# of pixels', fontsize=14, labelpad=15)
            plt.legend(fontsize=13, loc='upper right')
            plot_dir = os.path.join(bpm_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            filepath = os.path.join(plot_dir, 'master_flat_threshold.png')
            plt.savefig(filepath, dpi=150, format='png', bbox_inches='tight')
            plt.close()
        # ---------- bad-pixel percentage + header comment ----------
        bad = len(bpm[bpm == 1])
        perc_bad = (bad / bpm.size) * 100
        crosstalk_comment = f'correct_neighbors={correct_neighbors}' if correct_neighbors else 'not applied'
        comment = (f'Bad-pixel mask generated from combined dark and flat frames of {obs_date} observation. Threshold value of {bpm_thresh_sigma} sigma results in {perc_bad:.1f}% bad pixels. Crosstalk correction (longest-exptime hot pixels): {crosstalk_comment}.')

    # diagnostic plot for bpm image
    plt.figure(figsize=(10,5))
    plt.title(fr'{perc_bad:.1f}% bad pixels, threshold={bpm_thresh_sigma}$\sigma$', fontsize=12, pad=15)
    plt.imshow(bpm, origin='lower', cmap='Greys_r', vmin=0, vmax=1) 
    # Set png file
    plot_dir = os.path.join(bpm_dir,'plots')
    os.makedirs(plot_dir, exist_ok=True)
    png_file = 'bpm_image.png'
    # Add bpm directory path to png file
    filepath = os.path.join(plot_dir, png_file)
    # Save plot as png
    plt.savefig(filepath, dpi=500, format='png', bbox_inches="tight")
    plt.close()

    filename = f'NIRWALSBpm1x1_{bpm_thresh_sigma}sigma.fits'
    filepath = os.path.join(bpm_dir, filename)
    print(f"\nWriting out BPM to: {filepath}", end='\n\n')
    # set fits extensions
    primary = fits.PrimaryHDU()
    bpm_hdu = fits.ImageHDU(data=bpm.astype(np.float32), name='BPM')
    # Add header keywords
    bpm_hdu.header['COMMENT'] = comment
    # write out
    hdul = fits.HDUList([primary, bpm_hdu])
    hdul.writeto(filepath, overwrite=True)

    # Get all fits file(s) in product data directory
    wildcard = '*{0}*.fits'.format(obs_date)
    prd_files = sorted(f for f in glob.glob(os.path.join(prd_dir, wildcard)) if not duplicate(f))
    for prd_file in prd_files:
        basename = os.path.basename(prd_file)
        try:
            
            with fits.open(prd_file, mode='update') as hdul:
                # ------- add "BPM" hdu -------
                hdu_names = [hdu.name for hdu in hdul]
                # correct IPC crosstalk across this file's own SCI data; BPM-flagged pixels can't leak into their neighbors unless individually measured
                if correct_neighbors and SCI in hdu_names:
                    before = hdul[SCI].data
                    after = ipc_correction_global(before, alpha_asym_3, alpha_asym_1, bpm=(bpm == 1), calibrated_pixel=measured_3)
                    hdul[SCI].data = after
                    if prd_file == longest_dark_file:
                        ipc_correction_diagnostic_plot(before, after, bpm, measured_3, longest_dark_med, longest_exptime, bpm_dir)
                # remove existing BPM extension if present
                if 'BPM' in hdu_names:
                    bpm_index = hdu_names.index('BPM')
                    hdul.pop(bpm_index)
                # create new BPM extension
                bpm_hdu = fits.ImageHDU(data=bpm, name='BPM')
                bpm_hdu.header['COMMENT'] = comment
                # append updated BPM
                hdul.append(bpm_hdu)
                # -----------------------------
                # ----- fix header keys missing from raw NIRWALS data -----
                header = hdul['PRIMARY'].header
                header_keys = list(header.keys())
                # GR-ANGLE, GRRANGLE, and GRTILT are the same grating angle value
                # AR-ANGLE and CAMANG are the same camera angle value
                # for NIRWALS the camera angle is always 2x the grating angle. 
                # Take whichever value is present and copy it into any of the keys that are missing.
                gr_angle = next((header[k] for k in ('GR-ANGLE', 'GRRANGLE', 'GRTILT') if k in header_keys), None)
                if gr_angle is not None:
                    for k in ('GR-ANGLE', 'GRTILT', 'GRRANGLE'):
                        if k not in header_keys:
                            header[k] = (gr_angle, 'Grating angle (copied from another grating angle key)')

                cam_angle = next((header[k] for k in ('AR-ANGLE', 'CAMANG') if k in header_keys), None)
                if cam_angle is None and gr_angle is not None:
                    cam_angle = 2 * gr_angle
                if cam_angle is not None:
                    for k in ('AR-ANGLE', 'CAMANG'):
                        if k not in header_keys:
                            header[k] = (cam_angle, 'Camera angle (copied/derived as 2x grating angle)')
                if 'GRATING' not in header_keys:
                    header['GRATING'] = ('NG950', 'Grating name (NIRWALS has a single fixed grating)')
                if 'DISPAXIS' not in header_keys:
                    header['DISPAXIS'] = (1, 'Dispersion axis (0=vertical, 1=horizontal)')
                if 'PIXSUM' not in header_keys:
                    header['PIXSUM'] = ('1 1', 'CCD pixel summing (x y)')
                # --------------------------------

        except Exception as e:
            print(f"Error processing {basename}: {e}")

    return

# ---------------------------------------------------------------------------- #  # for full detector: zoom_centre=(1024, 1024), zoom_size=2048 
def ipc_correction_diagnostic_plot(before, after, bpm, measured, median_dark, exptime, bpm_dir, zoom_centre=(360, 1400), zoom_size=100):
# ---------------------------------------------------------------------------- #
    rows, cols = before.shape
    cx, cy = zoom_centre
    half = zoom_size // 2
    r0, r1 = max(0, cy - half), min(rows, cy + half)
    c0, c1 = max(0, cx - half), min(cols, cx + half)
    before = before[r0:r1, c0:c1]
    after = after[r0:r1, c0:c1]
    bpm = bpm[r0:r1, c0:c1]
    measured = measured[r0:r1, c0:c1]
    extent = [c0, c1, r0, r1]
    # compute each pixel's excess over the frame's own dark-current background
    raw_residual = before - median_dark 
    residual = after - median_dark

    good = bpm == 0
    vmin, vmax = np.nanpercentile(before[good], [1, 99])
    # shared scale for before and after correction
    rlim = max(np.nanpercentile(np.abs(raw_residual[good]), 99), np.nanpercentile(np.abs(residual[good]), 99))
    bpm_overlay = np.zeros((*before.shape, 4))
    bpm_overlay[bpm == 1] = [0, 0, 0, 1]
    # individually-measured pixels
    circle_rows, circle_cols = np.where(measured)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [(before, 'raw', 'gray', vmin, vmax, bpm_overlay),
              (raw_residual, r'raw $-$ background', 'berlin', -rlim, rlim, bpm_overlay),
              (after, 'corrected', 'gray', vmin, vmax, bpm_overlay),
              (residual, r'corrected $-$ background', 'berlin', -rlim, rlim, bpm_overlay)]
    for ax, (data, title, cmap, lo, hi, overlay) in zip(axes.flat, panels):
        im = ax.imshow(data, origin='lower', cmap=cmap, vmin=lo, vmax=hi, extent=extent)
        ax.imshow(overlay, origin='lower', extent=extent)
        for rm, cm in zip(circle_rows, circle_cols):
            ax.add_patch(plt.Circle((c0 + cm + 0.5, r0 + rm + 0.5), radius=3, fill=False, edgecolor='white', linewidth=0.4))
        ax.set_title(title, fontsize=15)
        fig.colorbar(im, ax=ax, label='counts / s', shrink=1.0)
    fig.suptitle(fr'{round(exptime,-2)}s master dark', fontsize=18)
    plt.tight_layout()
    plot_dir = os.path.join(bpm_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    filepath = os.path.join(plot_dir, 'IPC_correction.png')
    plt.savefig(filepath, dpi=500, format='png', bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------- #
def alpha_calibration_diagnostic_plot(dark, med, sig, bpm_thresh_sigma, alpha_asym_3_map, alpha_asym_1_map,
                                       alpha_sym_map, measured_3, measured_1, used_asym_3, used_asym_1, exptime, bpm_dir):
# ---------------------------------------------------------------------------- #
    central_values = dark[measured_3] - med
    asym_3_values = alpha_asym_3_map[measured_3]
    central_values_1 = dark[measured_1] - med
    asym_1_values = alpha_asym_1_map[measured_1]
    sym_values = alpha_sym_map[measured_1]   # alpha_sym depends on the same read-before neighbor as alpha_asym_1, so it shares that mask

    asym_3_lo, asym_3_hi = np.percentile(asym_3_values, [16, 84])
    asym_1_lo, asym_1_hi = np.percentile(asym_1_values, [16, 84])
    sym_lo, sym_hi = np.percentile(sym_values, [16, 84])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5), sharey=True, gridspec_kw={'width_ratios': [3, 1], 'wspace': 0})

    ax1.axvline(med, color='grey', ls='dotted', lw=1.2)
    ax1.axvspan(med - sig, med + sig, color='grey', alpha=0.3)
    ax1.text(med + 0.005, 0.062, 'dark-current\nbackground', color='grey', ha='center', fontsize=13)

    # 16th-84th percentile spread of each measured distribution
    xlim_min, xlim_max = 10**-(2.8), 10**(4.8)   # x-axis limits 
    log_xmin_data = np.log10(med + bpm_thresh_sigma * sig)  # 5-sigma cutoff
    log_xlim_min = np.log10(xlim_min)
    log_xlim_max = np.log10(xlim_max)
    xmin_fraction = (log_xmin_data - log_xlim_min) / (log_xlim_max - log_xlim_min)
    ax1.axhspan(sym_lo, sym_hi, xmin=xmin_fraction, color='#9a8c98', alpha=0.12)
    ax1.axhspan(asym_3_lo, asym_3_hi, xmin=xmin_fraction, color='#FD8721', alpha=0.12)
    ax1.axhspan(asym_1_lo, asym_1_hi, xmin=xmin_fraction, color='#577562', alpha=0.12)
    ax1.axhline(used_asym_3, xmin=xmin_fraction, color='#FD8721', ls='--', lw=1.2, zorder=5)
    ax1.axhline(used_asym_1, xmin=xmin_fraction, color='#577562', ls='--', lw=1.2, zorder=6)
    ax1.axhline(np.median(sym_values), xmin=xmin_fraction, color='#9a8c98', ls='--', lw=1.2, zorder=4)

    ax1.scatter(central_values_1, sym_values, s=8, alpha=0.1, color='#9a8c98')
    ax1.scatter(central_values, asym_3_values, s=8, alpha=0.05, color='#fec89a')
    ax1.scatter(central_values_1, asym_1_values, s=8, alpha=0.15, color='#B9CBC0')
    ax1.axvline(med + bpm_thresh_sigma * sig, color='black', lw=1.2)
    ax1.text(med + bpm_thresh_sigma * sig + 0.3, 0.062, fr'{bpm_thresh_sigma}$\sigma$'+'\n'+'hot pixel', color='black', ha='center', fontsize=13)
    ax1.set_xscale('log')
    ax1.set_xlabel('pixel value [counts/s]', fontsize=16, labelpad=15)
    ax1.set_ylabel(r'$\alpha$', fontsize=20, labelpad=15)
    ax1.set_ylim(-0.0001, 0.06)
    ax1.set_xlim(10**-(2.8), 10**(4.8)) 
    ax1.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))   # label every power of 10
    ax1.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=15))
    ax1.tick_params(labelsize=14)

    # right panel: alpha distribution
    bins = np.linspace(-0.005, 0.08, 100)
    sym_counts, _, _ = ax2.hist(sym_values, bins=bins, orientation='horizontal', color='#9a8c98', alpha=0.5)
    asym_3_counts, _, _ = ax2.hist(asym_3_values, bins=bins, orientation='horizontal', color='#fec89a', alpha=0.5)
    asym_1_counts, _, _ = ax2.hist(asym_1_values, bins=bins, orientation='horizontal', color='#B9CBC0', alpha=0.5)
    ax2.axhline(np.median(sym_values), color='#9a8c98', ls='--', lw=1.2)
    ax2.axhline(used_asym_3, color='#FD8721', ls='--', lw=1.2)
    ax2.axhline(used_asym_1, color='#577562', ls='--', lw=1.2)
    text_x_pos = max(sym_counts.max(), asym_3_counts.max(), asym_1_counts.max()) * 1.30   # 30% buffer
    ax2.text(text_x_pos, used_asym_3, fr'$\alpha_{{\rm asym,3}}={used_asym_3:.3f}$', color='#FD8721', ha='left', fontsize=12)
    ax2.text(text_x_pos, used_asym_1, fr'$\alpha_{{\rm asym,1}}={used_asym_1:.3f}$', color='#577562', ha='left', fontsize=12)
    ax2.text(text_x_pos, np.median(sym_values), fr'$\alpha_{{\rm sym}}={np.median(sym_values):.3f}$', color='#9a8c98', ha='left', fontsize=12)
    ax2.set_xlabel('# of pixels', fontsize=16, labelpad=15)
    ax2.set_xticks([0, 1500, 3000])
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.tick_params(labelsize=14)

    plt.tight_layout()
    plot_dir = os.path.join(bpm_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    filepath = os.path.join(plot_dir, 'alpha_calibration.png')
    plt.savefig(filepath, dpi=500, format='png', bbox_inches='tight')
    plt.close()

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
    flat_files = sorted(f for f in glob.glob(os.path.join(prd_dir, wildcard)) if not duplicate(f))

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

            # using raw master flat, not dark subtracted, for fiber tracing

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

        valid = cm

        # too little real signal for this polynomial order (e.g. an unilluminated edge row) -- leave continuum at 0
        if np.count_nonzero(f[valid]) <= 2 * continuum_fit['order']:
            continue

        # Fit row flux
        c_fit = Fit1D(p[valid], f[valid], **continuum_fit)
        # Set row continuum in continuum array
        continuum[i] = c_fit(p)
####>
#         plt.figure(1, figsize=(16, 9), tight_layout=True)
#         plt.plot(p, f, lw=1, ls='-', c='black')
#         plt.plot(p[cm], f[cm], lw=0.75, ls='-', c='cyan')
#         plt.plot(p, c_fit(p), lw=0.75, ls='--', c='red')
#         plt.xlabel('Pixel')
#         plt.ylabel('Flux')
#         plt.title('Row: {0}'.format(i+1))
#         plt.show()
#         plt.close()
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
    flat_files = sorted(f for f in glob.glob(os.path.join(prd_dir, wildcard)) if not duplicate(f))

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
            med_range = list(np.nanmedian(np.array(fibre['ranges']), axis=0))

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
        # Merge onto whatever is already on disk, so a config that failed this run doesn't lose a previously-cached trace
        existing_wrk = load_json_file(config_p['fibre_traces_wrk'])
        if existing_wrk and existing_wrk.get('traces'):
            merged_traces = dict(existing_wrk['traces'])
            merged_traces.update(config_f['wrk']['traces'])
            config_f['wrk']['traces'] = merged_traces

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