# ---------------------------------------------------------------------------- #
"""
SALT IFU utilities:
- saltifu provides general utilities for processing of ifu spectral data
"""
# ---------------------------------------------------------------------------- #

# Standard library imports
from math import ceil

# numpy import
import numpy as np
# scipy imports
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

# for plotting
import matplotlib.pyplot as plt
import os

# Application imports
# - saltreduce.functions
from ...functions import Fit1D

# ---------------------------------------------------------------------------- #

MYNAME = 'saltifu'

# ---------------------------------------------------------------------------- #
def find_fibres(image, axis, work, log):
# ---------------------------------------------------------------------------- #

    """
    image: <numpy array> 2D image array
    axis: <int> dispersion axis (0=vertical, 1=horizontal)
    work: <dict> work dictionary with parameters for finding fibres
        - nr_of_fibres: <int> expected nr of fibres
        - trace: <dict> dictionary for fibre tracing
          - size: <int> (pixels)
          - step: <int> (pixels)
        - profile: <dict> dictionary for data profile prep
          - compress: <str> median, mean or sum
          - smooth: <dict> (savgol_filter parameters)
            - window_length: <int>
            - polyorder: <int>
        - flat: <dict> 
          - reference: <str> flat file to use 
          - find_peaks: <dict> dictionary for finding peaks
            - height: <float> minimum peak height
            - width: <int> minimum peak width (pixels)
            - wlen: <int> maximum peak width (pixels)
            - rel_height: <float> relative height for peak width

    return: <list> fibres list
        - fibre: <dict> fibre dictionary
            - id: <str> fibre 'number'
            - type: <str> object or sky
            - range: <list> feature start and end rows
            - width: <int> feature width (end row minus start row)
    """

    # Add message to log
    msg = ' - find fibres'
    log.message(msg, with_header=False)

    # Initialise fibres dictionary
    fibres = {}
    # Initialise windows list
    windows = []
    # Initialise profiles list
    profiles = []

    # Loop for nr of fibres...
    for i in range(work['nr_of_fibres']):

        # Set fibre id
        id = '{0:03d}'.format(i + 1)
        # Initialise fibres dictionary entry
        # fibres[id] = {'centres': [], 'ranges': [], 'widths': []}
        fibres[id] = {'ranges': [], 'widths': []}

    # Set trace window dictionary from parameters
    window = work['trace']['window']
    # Set nr of image columns
    cols = image.shape[1]
    # Set nr of windows
    n_win = ceil(cols / window['step'])
    # Initialise window range
    r0 = 0; r1 = window['size']

    # Loop for nr of windows...
    for _ in range(n_win + 1):

        # Horizontal => window columns
        w_image = image[:, r0:r1]
        # Find features in windowed image
        features, profile = find_fibre_features(w_image, axis, work)

        # Loop for nr of fibres...
        for i in range(work['nr_of_fibres']):

            # Set fibre id
            id = '{0:03d}'.format(i + 1)
            # Check if fibre id is in features dictionary
            if id in features:
                # Add feature details to fibres dictionary
                # fibres[id]['centres'].append(features[id]['centre'])
                fibres[id]['ranges'].append(features[id]['range'])
                fibres[id]['widths'].append(features[id]['width'])

            else:
                # Add NaN to fibres dictionary
                # fibres[id]['centres'].append(np.nan)
                fibres[id]['ranges'].append([np.nan, np.nan])
                fibres[id]['widths'].append(np.nan)

        # Add window range to windows list
        windows.append([r0, r1])
        # Add fibre profile to profiles list
        profiles.append(profile.tolist())

        # Check if window range end is at end of image
        if r1 == cols:
            # This is the end, my friend...
            break

        else:
            # Increment window range
            r0 += window['step']; r1 = min(r1 + window['step'], cols)

    # Add message to log
    msg = '   - nr of fibres: {0}'.format(len(fibres))
    log.message(msg, with_header=False)

    return fibres, windows, profiles

# ---------------------------------------------------------------------------- #
def find_fibre_features(image, axis, work):
# ---------------------------------------------------------------------------- #

    # Set nr of image columns
    cols = image.shape[1]

    # Compress data profile (along dispersion axis):
    # - call method dynamically as attribute of numpy (np)
    profile = getattr(np, work['profile']['compress'])(image, axis=axis)
    # - check if method is 'sum'
    if work['profile']['compress'] == 'sum':
        # - average profile with nr of columns
        profile /= float(cols)
    # Check if smoothing is required
    if work['profile']['smooth']['window_length'] > 0:
        # Smooth 1D profile array
        profile = savgol_filter(profile, **work['profile']['smooth'])

    # Find all peaks / features: scipy.signal.find_peaks
    _, properties = find_peaks(profile, **work['flat']['find_peaks'])
    # Set property arrays
    left_ips = properties['left_ips']
    right_ips = properties['right_ips']
    widths = properties['widths']

    # Initialise features (fibres) dictionary
    features = {}

    # Round property arrays
    left_ips = np.around(left_ips, decimals=2)
    right_ips = np.around(right_ips, decimals=2)
    widths = np.around(widths, decimals=2)
    # Set features list
    features_list = list(zip(list(left_ips), list(right_ips), list(widths)))

    # Loop for features...
    for i, f in enumerate(features_list):

        # Check feature width
        if f[2] >= work['flat']['find_peaks']['width']:
            # Set id
            id = '{:03d}'.format(i+1)
            # Set feature dictionary entry
            feature = {'type': 'dummy',
                       'range': [f[0], f[1]],
                       'width': f[2]}
            # Set features dictionary entry
            features[id] = feature

    return features, profile

# ---------------------------------------------------------------------------- #
def set_fibre_traces(fibres_config, work, log):
# ---------------------------------------------------------------------------- #

    # Check if exposure config is in existing fibre traces
    if work['config'] in fibres_config.keys():
        # Add message to log
        msg = ' - set existing fibre traces dictionary'
        log.message(msg, with_header=False)
        # Set existing fibre traces dictionary for exposure config
        traces = fibres_config[work['config']]

    else:
        # Add message to log
        msg = ' - initialise new fibre traces dictionary'
        log.message(msg, with_header=False)
        # Initialise new fibre traces dictionary
        traces = {
            'reference': work['flat']['reference'],
            'fibres': work['fibres']
        }

    return traces

# ---------------------------------------------------------------------------- #
def trace_fibres(traces, fibres, windows, work, log):
# ---------------------------------------------------------------------------- #

    # Add message to log
    msg = ' - trace fibres'
    log.message(msg, with_header=False)

    # Initialise mid-points list
    mid_points = []

    # Loop for windows...
    for window in windows:

        # Add window mid-point to mid-points list
        mid_points.append(0.5 * (window[0] + window[1]))

    # Convert mid-points list to array
    mid_points = np.array(mid_points)

    # Initialise fibre trace lists
    lower_trace = [[] for _ in range(work['nr_of_fibres'])]
    upper_trace = [[] for _ in range(work['nr_of_fibres'])]

    # Loop for nr of fibres...
    for i in range(work['nr_of_fibres']):

        # Set fibre id
        id = '{0:03d}'.format(i + 1)
        # Check if fibre id is in fibres dictionary
        if id in fibres:
            # Set traces:
            # - lower
            lower_trace[i] = [r[0] for r in fibres[id]['ranges']]
            # - upper
            upper_trace[i] = [r[1] for r in fibres[id]['ranges']]

    # Set minimum nr of points required for trace
    min_points = int(len(mid_points) * work['trace']['min_points_pc'] / 100)

    # Loop for traces...
    for i, t in enumerate(list(zip(lower_trace, upper_trace))):

        # Set fibre id
        id = '{0:03d}'.format(i + 1)
        # Convert traces to arrays
        l_trace = np.array(t[0])
        u_trace = np.array(t[1])

        # Set mask for greater than zero lower trace points
        gz = l_trace > 0.
        # Check nr of trace points
        if len(mid_points[gz]) < min_points:
            # Add message to log
            msg = ('   - not enough trace points to determine fibre trace: '
                   'Fibre {0}').format(id)
            log.message(msg, with_header=False)
            return False, traces

        # Fit lower trace
        lower_fit = Fit1D(mid_points[gz], l_trace[gz], **work['trace']['fit'])
        # Check lower trace fit coefficients for NaN
        if np.isnan(lower_fit.coef).any():
            # Add message to log
            msg = ('   - fitting lower trace returned NaN coefficient(s): '
                   'Fibre {0}').format(id)
            log.message(msg, with_header=False)
            return False, traces

        # Fit upper trace
        upper_fit = Fit1D(mid_points[gz], u_trace[gz], **work['trace']['fit'])
        # Check upper trace fit coefficients for NaN
        if np.isnan(lower_fit.coef).any():
            # Add message to log
            msg = ('   - fitting upper trace returned NaN coefficient(s): '
                   'Fibre {0}').format(id)
            log.message(msg, with_header=False)
            return False, traces

        # Set coefficients in fibres trace dictionary
        traces['fibres'][id]['lower_trace'] = lower_fit.coef.tolist()
        traces['fibres'][id]['upper_trace'] = upper_fit.coef.tolist()

    return True, traces

# ---------------------------------------------------------------------------- #
def extract_fibre_optimal(sci, gpm, flt, gain, read_noise, aperture_weight):
# ---------------------------------------------------------------------------- #
    '''
    Horne 1986 optimal extraction method.
    Default extraction option.
    '''
    # spatial profile P from flat image
    flt_ap = flt * aperture_weight  # down-weight the flat ion the edge rows by how much of each row is really in the fiber
    flt_col_sums = flt_ap.sum(axis=0)   # sum across fibers
    flt_col_sums[flt_col_sums == 0] = 1.0
    P = flt_ap / flt_col_sums   # normalised to sum to 1 per column

    sci_e = sci * gain               # [counts] --> [e-], science in electrons
    V_e = read_noise**2 + np.abs(sci_e)  # variance in electrons
    V_e[V_e <= 0] = 1.0                  # avoid division by zero
    V_e[gpm == 0] = 1e30             # bad pixels get very high variance
    sci_ap = sci_e * aperture_weight  # down-weight the flat ion the edge rows by how much of each row is really in the fiber

    # Horne 1986 Eq. 8
    num = np.sum(P * sci_ap / V_e, axis=0)
    den = np.sum(P**2 / V_e, axis=0)       # ''
    good = den > 0                         # avoid divide by zero
    F_e = np.zeros_like(den, dtype=np.float32)
    F_e[good] = num[good] / den[good]      # [e-]
    F = F_e / gain                         # [e-] --> [counts] 

    # flat-field  using mean-normalised flat spectrum (same as boxcar extraction method)
    flt_col_sums = flt_ap.sum(axis=0)    # sum across fibers
    flt_mean = np.nanmean(flt_col_sums[flt_col_sums != 0])
    flt_mean_norm = flt_col_sums / flt_mean
    good = flt_mean_norm != 0
    F[good] = F[good] / flt_mean_norm[good]

    F[~np.isfinite(F)] = 0.0
    return F

# ---------------------------------------------------------------------------- #
def extract_fibre_boxcar(sciarr, gpmarr, fltarr):
# ---------------------------------------------------------------------------- #
    '''
    Boxcar extraction with flat-fielding and good-pixel renormalization.
    Secondary extraction option.
    '''
    non_nan = ~np.isnan(sciarr)

    # Check flat field image array
    if fltarr is not None:
        # Linearly transform intensity scale (bscale)
        fltarr[non_nan] /= fltarr[non_nan].mean()
        # Set combined non NaN science and non zero flat mask
        mask = (non_nan) * (fltarr != 0)
        # Flat field science image array
        sciarr[mask] /= fltarr[mask]

    # Renormalize by number of good pixels per wavelength bin
    # Set combined non NaN science and non zero good pixel mask
    mask = (non_nan) * (gpmarr != 0)
    # Scale fibre flux for 'nr' of good pixels:
    # - divide 'sci' by 'gpm'
    sciarr[mask] /= gpmarr[mask]
    # - multiply 'sci' by 'gpm' mean
    sciarr[non_nan] *= gpmarr[non_nan].mean()

    return sciarr

# ---------------------------------------------------------------------------- #
def plot_extraction_comparison(method, id, cols_to_debug, sci_opt, gpm, flt, aperture_weight, sciarr, gpmarr, fltarr, gain, read_noise, work):
# ---------------------------------------------------------------------------- #
    '''
    Top: extracted spectrum using both optimal and boxcar methods. 
    Bottom: some diagnostic columns (plotting: raw-flat / optimal / boxcar profile, aperture weight, good-pixel mask).
    '''
    mark_colors = ['green', 'orange', 'purple']

    if len(cols_to_debug) == 0:
        return

    rows = np.arange(flt.shape[0])
    cols = np.arange(flt.shape[1])

    def normalize(a):
        s = a.sum(axis=0)
        s = np.where(s == 0, 1.0, s)
        return a / s

    # extraction weighting from optimal and boxcar
    P_raw = normalize(flt)                        # profile as it sits in the flat data
    P_opt = normalize(flt * aperture_weight)      # optimal profile P
    W_box = normalize(aperture_weight * gpm)      # boxcar effective weight (top-hat over good rows)

    # extracted data from optimal and boxcar for fiber spectrum
    F_opt = extract_fibre_optimal(sci_opt.copy(), gpm.copy(), flt.copy(), gain, read_noise, aperture_weight)
    F_box = extract_fibre_boxcar(sciarr.copy(), gpmarr, fltarr.copy())
    gpm_per_col = gpm.sum(axis=0)

    fig = plt.figure(figsize=(4.6 * len(cols_to_debug), 10))

    outer = fig.add_gridspec(2, 1, height_ratios=[2, 3], hspace=0.30)
    # ===================== TOP: full-width spectrum + gpm count =====================
    top = outer[0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
    axS = fig.add_subplot(top[0])
    axC = fig.add_subplot(top[1], sharex=axS)

    axS.set_title(f'extracted spectrum for fibre #{id} (method set to {method})', fontsize=15, pad=12)
    axS.step(cols, F_box, where='mid', lw=0.8, color='red',  alpha=0.6, label='boxcar')
    axS.step(cols, F_opt, where='mid', lw=0.8, color='blue', alpha=0.6, label='optimal')
    axS.set_ylabel('[counts / s]', fontsize=12, labelpad=12)
    axS.legend(fontsize=12, loc='upper right')
    m = np.isfinite(F_opt) & np.isfinite(F_box)
    if m.any():
        hi = np.nanpercentile(np.concatenate([F_opt[m], F_box[m]]), 99)
        axS.set_ylim(-8, max(hi, 1))
    plt.setp(axS.get_xticklabels(), visible=False)

    axC.fill_between(cols, 0, gpm_per_col, step='mid', color='grey', alpha=0.8)
    axC.set_ylim(0, gpm_per_col.max() + 1)
    axC.set_ylabel('# good pix', fontsize=10, labelpad=12)
    axC.set_xlabel('wavelength column', fontsize=12, labelpad=10)

    for j, col in enumerate(cols_to_debug):
        c = mark_colors[j % len(mark_colors)]
        axS.axvline(col, color=c, ls='dashed', lw=1.8, zorder=5)
        axC.axvline(col, color=c, ls='dashed', lw=1.8, zorder=5)

    # ===================== BOTTOM: one stack per debug column =====================
    bottom = outer[1].subgridspec(1, len(cols_to_debug), wspace=0.05)
    for j, col in enumerate(cols_to_debug):

        good = gpm[:,col] == 1
        F_ap = sci_opt[:,col] * aperture_weight[:,col]  * good  # aperture-weighted data with bad rows masked out
        F_ap_normalized = F_ap / F_ap.sum()

        c = mark_colors[j % len(mark_colors)]
        cell = bottom[j].subgridspec(3, 1, height_ratios=[4, 1, 1], hspace=0.0)
        axP = fig.add_subplot(cell[0])
        axA = fig.add_subplot(cell[1], sharex=axP)
        axM = fig.add_subplot(cell[2], sharex=axP)

        ratio = F_opt[col]/F_box[col]
        axP.set_title(f'col #{col},  opt/boxcar={ratio:.1f}', color=c, fontsize=12, pad=8)
        axP.step(rows, F_ap_normalized, where='mid', lw=1.5, color='black', label='data', zorder=5)
        axP.step(rows, P_raw[:, col], where='mid', lw=1.5, color='lightblue', label='raw flat')
        axP.step(rows, P_opt[:, col], where='mid', lw=1.5, color='blue', alpha=0.8, label='optimal profile')
        axP.step(rows, W_box[:, col], where='mid', lw=1.5, color='red', alpha=0.8, label='boxcar profile')
        axP.set_ylim(-0.05, max(P_raw[:, col].max(), P_opt[:, col].max(), W_box[:, col].max(), F_ap_normalized.max()) * 1.1)
        plt.setp(axP.get_xticklabels(), visible=False)

        axA.fill_between(rows, 0, aperture_weight[:, col], step='mid', color='grey', alpha=0.8)
        axA.set_ylim(-0.09, 1.15)
        plt.setp(axA.get_xticklabels(), visible=False)

        axM.fill_between(rows, 0, gpm[:, col], step='mid', color='grey', alpha=0.8)
        axM.set_ylim(-0.09, 1.15)
        axM.set_xlabel('fibre aperture row', fontsize=11, labelpad=8)

        if j == 0: 
            axP.set_ylabel('norm values', fontsize=11, labelpad=12)
            axA.set_ylabel('weight', fontsize=9, labelpad=15)
            axM.set_ylabel('# good pix', fontsize=9, labelpad=15)
            axP.legend(fontsize=10, loc='upper left')
        else:
            for ax in (axP, axA, axM):
                plt.setp(ax.get_yticklabels(), visible=False)

    plot_dir = os.path.join(work['output']['dir'], 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    out_file = os.path.join(plot_dir, '{0}_extraction_debug_fibre{1}.png'.format(work['file'], id))
    plt.savefig(out_file, dpi=300, format='png', bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------- #
def extract_fibres(sci, sci_unmasked, gpm, flt, traces, gain, read_noise, work, log):
# ---------------------------------------------------------------------------- #
    '''
    Extract all fibres from the 2D images using the configured method (optimal or boxcar).

    work['extract_method']:
      'optimal' : Horne 1986 inverse-variance weighted extraction (default).
                  Uses per-fibre native-resolution science/gpm/flat regions.
      'boxcar'  : sum within aperture with good-pixel renormalization.
                  Uses per-fibre collapsed 1D arrays.

    sci : 2D science image, gpm applied (boxcar, bad pixels contribute mean flux value to sum)
    sci_unmasked : 2D science image, gpm NOT applied (optimal, bad-pixels are given high noise)
    gpm : 2D good-pixel image (1=good, 0=bad)
    flt  2D flat / continuum-fit image (gpm applied), or None
    traces : fibre traces dictionary
    gain : [e-]
    read_noise : [e-]

    Returns (fibres, good_pixels)
    '''
    # Set extraction method (default optimal)
    method = work.get('extract_method', 'optimal')
    flat_type = work['flat']['type'][work['exp_type']]
    apply_flat_boxcar = flat_type in ['flat', 'fit']

    # optimal extraction requires a flat field for the spatial profile
    if method == 'optimal' and flt is None:

        # arc observations should not be flat-fielded, use boxcar method without flat
        if work['exp_type'] == 'arc':
            method = 'boxcar'
        else:
            raise SALTError("Optimal extraction requires a flat field (flat.type must be 'flat' or 'fit').")


    if method not in ('optimal', 'boxcar'):
        raise SALTError("Unknown extract_method: '{0}' (expected 'optimal' or 'boxcar')".format(method))
    
    # Add message to log
    # log.message(' - extract fibres: method = {0}'.format(method), with_header=False)
    print(' - extract fibres: method = {0}'.format(method))

    ###### DEBUGGING ########
    fiber_to_debug = '050'
    #########################

    # Initialise extracted fibres and good pixels dictionaries
    fibres, good_pixels = {}, {}

    # Loop for fibre traces...
    for id, fibre in traces.items():

        ###### DEBUGGING ########
        if id != fiber_to_debug:
            cols_to_debug = []
        else:
            cols_to_debug = [192, 1110, 1800]
        #########################

        # Set fibre row range and oversampled 'valid' aperture mask
        r_min, r_max, valid = set_fibre(fibre, work)

        # Good-pixel count per wavelength bin
        gpmarr = set_fibre_array(gpm, r_min, r_max, valid, work)

        ######################## DEBUG PLOT  #################################
        if len(cols_to_debug) != 0 and flt is not None:
            # aperture weights for sub-pixel precision
            n_rows = r_max - r_min            # number of pixels contributing to the fiber
            n_sub_rows = work['row_repeat']   # number of sub-pixels contributing to the fiber
            n_cols = valid.shape[-1]          # column (wavelength) pixels
            aperture_weight = valid.reshape(n_rows, n_sub_rows, n_cols).mean(axis=1)  # averaging across sub-pixels to get sub-pixel precision on detector pixel resolution

            # extract for optimal
            sci_2D = sci_unmasked[r_min:r_max, :].copy()
            gpm_2D = gpm[r_min:r_max, :].copy()
            flt_2D = flt[r_min:r_max, :].copy()

            # extract for boxcar
            sciarr = set_fibre_array(sci, r_min, r_max, valid, work)
            fltarr = set_fibre_array(flt, r_min, r_max, valid, work)

            plot_extraction_comparison(method, id, cols_to_debug, sci_2D, gpm_2D, flt_2D, aperture_weight, sciarr, gpmarr, fltarr, gain, read_noise, work)
        #######################################################################

        # =========================== BOXCAR ===========================
        if method == 'boxcar':
            # Collapse science (gpm applied) over the oversampled aperture
            sciarr = set_fibre_array(sci, r_min, r_max, valid, work)

            fltarr = None
            if apply_flat_boxcar and flt is not None:
                # Extract fibre flat field array
                fltarr = set_fibre_array(flt, r_min, r_max, valid, work)

            # Add fibre flux to extracted fibres dictionary
            fibres[id] = extract_fibre_boxcar(sciarr, gpmarr, fltarr)

        # =========================== OPTIMAL ==========================
        elif method == 'optimal':
            # computing sub-pixel precision (as with boxcar extraction)
            n_rows = r_max - r_min            # number of pixels contributing to the fiber
            n_sub_rows  = work['row_repeat']  # number of sub-pixels contributing to the fiber
            n_cols = valid.shape[-1]          # column (wavelength) pixels
            aperture_weight = valid.reshape(n_rows, n_sub_rows, n_cols).mean(axis=1)  # averaging across sub-pixels to get sub-pixel precision on detector pixel resolution

            # Extract fibres for all arrays
            sci_2D = sci_unmasked[r_min:r_max, :].copy()
            gpm_2D = gpm[r_min:r_max, :].copy()
            flt_2D = flt[r_min:r_max, :].copy()
        
            # Add fibre flux to extracted fibres dictionary
            fibres[id] = extract_fibre_optimal(sci_2D, gpm_2D, flt_2D, gain, read_noise, aperture_weight)

        # Store good-pixel count for this fibre
        good_pixels[id] = gpmarr

    return fibres, good_pixels

# ---------------------------------------------------------------------------- #
def set_fibre(fibre, work):
# ---------------------------------------------------------------------------- #

    # Set trace (fitted) functions:
    # - upper trace fit
    u = Fit1D([], [], coef=fibre['upper_trace'], **work['trace']['fit'])
    # - lower trace fit
    l = Fit1D([], [], coef=fibre['lower_trace'], **work['trace']['fit'])

    # Set lower and upper rows
    l_rows, u_rows = l(work['p']), u(work['p'])
    # Set minimum and maximum rows (covers full extent of current fibre)
    r_min, r_max = int(l_rows.min()), int(u_rows.max() + 0.5)

    # Set shape for 'repeated' rows and valid arrays
    shape = (work['row_repeat'] * (r_max - r_min), work['cols'])
    # Set rows array: Each row contains an array of the row index
    rows = np.indices(shape)[0]
    # Initialise an array for the 'valid' fibre image array
    valid = np.zeros(shape, dtype=np.uint8)

    # Set upper and lower rows for 'repeated' rows
    u_rows = (work['row_repeat'] * (u_rows - l_rows.min()))
    l_rows = (work['row_repeat'] * (l_rows - l_rows.min()))

    # Loop for rows...
    for i, row in enumerate(rows):

        # Set valid if row falls between the lower and upper rows
        valid[i] = (row > l_rows) * (row < u_rows)

    return r_min, r_max, valid

# ---------------------------------------------------------------------------- #
def set_fibre_array(img, r_min, r_max, valid, work):
# ---------------------------------------------------------------------------- #

    # Initialise fibre flux array
    fibre_arr = np.zeros(work['cols'], dtype=np.float32)

    # Set fibre image (covers full extent of fibre)
    img = img[r_min:r_max, :].copy()
    # Repeat each row N times
    img = np.repeat(np.repeat(img, work['row_repeat'], axis=0), 1, axis=1)
    # Apply valid fibre image array to the full extent of fibre
    img *= valid
    # Set fibre array
    fibre_arr = np.sum(img, axis=0) / work['row_repeat']

    return fibre_arr

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
class SALTError(Exception):
# ---------------------------------------------------------------------------- #

    """Basic exception"""
    pass

# ---------------------------------------------------------------------------- #