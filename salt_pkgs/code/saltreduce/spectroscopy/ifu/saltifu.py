# ---------------------------------------------------------------------------- #
"""
SALT IFU utilities:
- saltifu provides general utilities for processing of ifu spectral data
"""
# ---------------------------------------------------------------------------- #

# Standard library imports
import copy
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
from ...functions import AMP_BOUNDARIES

# ---------------------------------------------------------------------------- #

MYNAME = 'saltifu'

# ---------------------------------------------------------------------------- #
def find_fibres(image, axis, work, log):
# ---------------------------------------------------------------------------- #

    """
    Fibres are tracked across windows by nearest row position.

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

    # Set trace window dictionary from parameters
    window = work['trace']['window']
    # Set cross-window position-matching tolerance
    match_tolerance = work['trace']['window_match_tolerance']
    # Set nr of image columns
    cols = image.shape[1]
    # Set nr of windows
    n_win = ceil(cols / window['step'])
    # Initialise window range
    r0 = 0; r1 = window['size']

    # Initialise tracks list (of fiber positions across windows)
    tracks = []
    # Initialise windows list
    windows = []
    # Initialise profiles list
    profiles = []

    # Loop for nr of windows...
    for win_idx in range(n_win + 1):

        # Horizontal => window columns
        w_image = image[:, r0:r1]
        # Find features in windowed image
        features, profile = find_fibre_features(w_image, axis, work)

        # Set this window's features, in row-position order
        feature_ids = sorted(features.keys())
        feature_ranges = [features[fid]['range'] for fid in feature_ids]
        feature_widths = [features[fid]['width'] for fid in feature_ids]
        feature_positions = [0.5 * (r[0] + r[1]) for r in feature_ranges]
        claimed_feature = [False] * len(feature_ids)

        if win_idx == 0:
            # Each feature starts its own track
            for pos, rng, wid in zip(feature_positions, feature_ranges, feature_widths):
                tracks.append({'ranges': [rng], 'widths': [wid], 'last_pos': pos})

        else:
            # Match each existing track to the nearest unclaimed feature in window
            for track in tracks:
                # Nearest unclaimed feature
                nearest_f = -1
                nearest_dist = None
                for f_idx, pos in enumerate(feature_positions):
                    if claimed_feature[f_idx]:
                        continue
                    dist = abs(pos - track['last_pos'])
                    if nearest_dist is None or dist < nearest_dist:
                        nearest_dist = dist
                        nearest_f = f_idx

                # Check if nearest feature is within tolerance
                if nearest_f >= 0 and nearest_dist <= match_tolerance:
                    # Extend fiber track with matched feature
                    track['ranges'].append(feature_ranges[nearest_f])
                    track['widths'].append(feature_widths[nearest_f])
                    track['last_pos'] = feature_positions[nearest_f]
                    claimed_feature[nearest_f] = True

                else:
                    # No match this window
                    track['ranges'].append([np.nan, np.nan])
                    track['widths'].append(np.nan)

            # Unclaimed features get a new track in window
            for f_idx, pos in enumerate(feature_positions):
                if claimed_feature[f_idx]:
                    continue
                # No matches in previous windows
                new_ranges = [[np.nan, np.nan]] * win_idx + [feature_ranges[f_idx]]
                new_widths = [np.nan] * win_idx + [feature_widths[f_idx]]
                tracks.append({'ranges': new_ranges, 'widths': new_widths, 'last_pos': pos})

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

    # Set nr of processed windows
    n_windows = len(windows)

    # Loop tracks...
    for track in tracks:

        # Pad any track shorter than the full window count
        while len(track['ranges']) < n_windows:
            track['ranges'].append([np.nan, np.nan])
            track['widths'].append(np.nan)

    # Set each track's first valid position
    sort_keys = []
    for track in tracks:
        first_pos = np.inf
        for r in track['ranges']:
            if not np.isnan(r[0]):
                first_pos = 0.5 * (r[0] + r[1])
                break
        sort_keys.append(first_pos)

    # Sort tracks by first valid position
    order = np.argsort(sort_keys)
    tracks = [tracks[i] for i in order]

    # Build fibres dictionary, keyed by row-ascending ID
    fibres = {}
    for i, track in enumerate(tracks):
        id = '{0:03d}'.format(i + 1)
        fibres[id] = {'ranges': track['ranges'], 'widths': track['widths']}

    # Add message to log
    msg = '   - nr of fibres: {0}'.format(len(fibres))
    log.message(msg, with_header=False)
    if len(fibres) != work['nr_of_fibres']:
        # Add message to log
        msg = '   - WARNING: nr of fibres found ({0}) does not match nr_of_fibres ({1})'.format(len(fibres), work['nr_of_fibres'])
        log.message(msg, with_header=False)

    return fibres, windows, profiles

# ---------------------------------------------------------------------------- #
def find_fibre_features(image, axis, work):
# ---------------------------------------------------------------------------- #

    # Set nr of image columns
    cols = image.shape[1]

    # Compress data profile (along dispersion axis):
    # - call nan-aware method dynamically as attribute of numpy (np)
    profile = getattr(np, 'nan' + work['profile']['compress'])(image, axis=axis)
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
def set_fibre_traces(fibres_config, fibres, work, log):
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
        # deep copy: each config needs its own independent fibres dict
        traces = {
            'reference': work['flat']['reference'],
            'fibres': copy.deepcopy(work['fibres'])
        }

    # Add a placeholder entry for any detected fibre id beyond the static catalog
    for id in fibres.keys():
        if id not in traces['fibres']:
            traces['fibres'][id] = {
                'ifu_id': None, 'type': None, 'slit_id': None, 'block': None,
                'lower_trace': [], 'upper_trace': []
            }

    return traces

# ---------------------------------------------------------------------------- #
def fit_trace(x, y, fit_kwargs):
# ---------------------------------------------------------------------------- #
    order = fit_kwargs['order']
    kwargs = dict(fit_kwargs)
    fit = Fit1D(x, y, **kwargs)
    # near-flat trace
    while np.isnan(fit.coef).any() and kwargs['order'] > 0:
        # drop the order by one and retry until the fit is finite or order reaches 0.
        kwargs['order'] -= 1
        fit = Fit1D(x, y, **kwargs)
    # pad back out to the requested order's coefficient count (callers reconstruct with a fixed order)
    fit.coef = np.pad(fit.coef, (0, order + 1 - len(fit.coef)))
    return fit

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

    # Set fibre ids, from the detected fibres dictionary (may differ from nr_of_fibres)
    ids = list(fibres.keys())

    # Set minimum nr of points required for trace
    min_points = int(len(mid_points) * work['trace']['min_points_pc'] / 100)
    # Set flat-line placeholder coefficients for an untraceable fibre id
    zero_coef = [0.0] * (work['trace']['fit']['order'] + 1)

    # Loop for fibre ids...
    for id in ids:

        # Set traces:
        # - lower
        l_trace = np.array([r[0] for r in fibres[id]['ranges']])
        # - upper
        u_trace = np.array([r[1] for r in fibres[id]['ranges']])

        # Set mask for greater than zero lower trace points
        gz = l_trace > 0.
        # Check nr of trace points
        if len(mid_points[gz]) < min_points:
            # Add message to log
            msg = ('   - not enough trace points to determine fibre trace: '
                   'Fibre {0}').format(id)
            log.message(msg, with_header=False)
            # Flag fibre id as untraceable
            traces['fibres'][id]['lower_trace'] = zero_coef
            traces['fibres'][id]['upper_trace'] = zero_coef
            traces['fibres'][id]['trace_failed'] = True
            continue

        # Fit lower trace
        lower_fit = fit_trace(mid_points[gz], l_trace[gz], work['trace']['fit'])
        # Check lower trace fit coefficients for NaN
        if np.isnan(lower_fit.coef).any():
            # Add message to log
            msg = ('   - fitting lower trace returned NaN coefficient(s): '
                   'Fibre {0}').format(id)
            log.message(msg, with_header=False)
            # Flag fibre id as untraceable
            traces['fibres'][id]['lower_trace'] = zero_coef
            traces['fibres'][id]['upper_trace'] = zero_coef
            traces['fibres'][id]['trace_failed'] = True
            continue

        # Fit upper trace
        upper_fit = fit_trace(mid_points[gz], u_trace[gz], work['trace']['fit'])
        # Check upper trace fit coefficients for NaN
        if np.isnan(upper_fit.coef).any():
            # Add message to log
            msg = ('   - fitting upper trace returned NaN coefficient(s): '
                   'Fibre {0}').format(id)
            log.message(msg, with_header=False)
            # Flag fibre id as untraceable
            traces['fibres'][id]['lower_trace'] = zero_coef
            traces['fibres'][id]['upper_trace'] = zero_coef
            traces['fibres'][id]['trace_failed'] = True
            continue

        # Set coefficients in fibres trace dictionary
        traces['fibres'][id]['lower_trace'] = lower_fit.coef.tolist()
        traces['fibres'][id]['upper_trace'] = upper_fit.coef.tolist()

    return True, traces

# ---------------------------------------------------------------------------- #
def realign_fibre_identities(traces, work, log):
# ---------------------------------------------------------------------------- #

    # Set this flat's own FIBERMAP data
    fibre_map = work.get('fibre_map')
    if fibre_map is None:
        # Add message to log
        msg = ' - no FIBERMAP extension in flat: skipping fibre identity realignment'
        log.message(msg, with_header=False)
        return

    # Set expected row offset and match tolerance
    offset = work['trace'].get('expected_row_offset')
    tolerance = work['trace'].get('row_offset_tolerance')
    if offset is None or tolerance is None:
        # Add message to log
        msg = ' - trace.expected_row_offset/row_offset_tolerance not configured: skipping fibre identity realignment'
        log.message(msg, with_header=False)
        return

    # Add message to log
    msg = ' - realign fibre identities'
    log.message(msg, with_header=False)

    # Set fibre id list
    ids = list(traces['fibres'].keys())
    n = len(ids)

    # Set static identity for each fibre id (snapshot before any reassignment)
    ifu_ids = [traces['fibres'][id]['ifu_id'] for id in ids]
    types = [traces['fibres'][id]['type'] for id in ids]
    blocks = [traces['fibres'][id]['block'] for id in ids]
    slit_ids = [traces['fibres'][id]['slit_id'] for id in ids]

    # Set detected trace centre (column 0) for each fibre id
    detected = np.full(n, np.nan)
    for i, id in enumerate(ids):
        if traces['fibres'][id].get('trace_failed'):
            continue
        lower = traces['fibres'][id]['lower_trace']
        upper = traces['fibres'][id]['upper_trace']
        if lower and upper:
            detected[i] = 0.5 * (lower[0] + upper[0])

    # Set expected trace centre for each fibre id, from this flat's own FIBERMAP
    expected = np.full(n, np.nan)
    for i in range(n):
        fm_type = 'object' if types[i] == 'obj' else 'sky'
        match = fibre_map[(fibre_map['IFU_ID'] == ifu_ids[i]) & (fibre_map['TYPE'] == fm_type)]
        if len(match):
            expected[i] = match['ROW'][0] + offset

    # Match each detected trace to the nearest, not-yet-claimed expected position
    assigned_to = np.full(n, -1, dtype=int)
    claimed = np.zeros(n, dtype=bool)

    # Pass 1: nearest match within tolerance, ids taken in slit order
    for i in range(n):
        if not np.isfinite(detected[i]):
            continue
        available = np.where(np.isfinite(expected) & ~claimed)[0]
        if len(available) == 0:
            continue
        j = available[np.argmin(np.abs(expected[available] - detected[i]))]
        if abs(expected[j] - detected[i]) <= tolerance:
            # fibre id i is re-identified as expected slot j
            assigned_to[i] = j
            claimed[j] = True

    # Set forced match tolerance
    forced_tolerance = work['trace'].get('forced_row_offset_tolerance')

    # Pass 2: forced match for anything left over, within forced_row_offset_tolerance
    n_forced = 0
    if forced_tolerance is not None:
        for i in range(n):
            if assigned_to[i] >= 0 or not np.isfinite(detected[i]):
                continue
            available = np.where(np.isfinite(expected) & ~claimed)[0]
            if len(available) == 0:
                continue
            j = available[np.argmin(np.abs(expected[available] - detected[i]))]
            if abs(expected[j] - detected[i]) <= forced_tolerance:
                assigned_to[i] = j
                claimed[j] = True
                n_forced += 1

    # Apply matched identities; mark anything whose own identity was claimed elsewhere invalid
    n_relabeled = 0
    n_invalidated = 0
    for i in range(n):
        j = assigned_to[i]
        if j < 0:
            # Own original identity was claimed by another fibre id
            if claimed[i]:
                n_invalidated += 1
                traces['fibres'][ids[i]]['ifu_id'] = -n_invalidated
            continue
        if (ifu_ids[j], types[j]) != (ifu_ids[i], types[i]):
            n_relabeled += 1
        traces['fibres'][ids[i]]['ifu_id'] = ifu_ids[j]
        traces['fibres'][ids[i]]['type'] = types[j]
        traces['fibres'][ids[i]]['block'] = blocks[j]
        traces['fibres'][ids[i]]['slit_id'] = slit_ids[j]

    # Set nr of fibre ids left with a safe, unclaimed, best-effort original identity
    n_unresolved = int(np.sum((assigned_to < 0) & ~claimed))

    # Add message to log
    msg = '   - relabeled: {0}, forced: {1}, unresolved: {2}, invalidated: {3}'.format(n_relabeled, n_forced, n_unresolved, n_invalidated)
    log.message(msg, with_header=False)
    if n_forced:
        # Add message to log
        msg = '   - WARNING: {0} fibre id(s) left with forced, lower-confidence identity'.format(n_forced)
        log.message(msg, with_header=False)
    if n_unresolved:
        # Add message to log
        msg = '   - WARNING: {0} fibre id(s) left with unverified identity'.format(n_unresolved)
        log.message(msg, with_header=False)
    if n_invalidated:
        # Add message to log
        msg = '   - WARNING: {0} fibre id(s) had their IDs claimed by another fibre id: IFU_ID set to a unique negative sentinel'.format(n_invalidated)
        log.message(msg, with_header=False)

    return

# ---------------------------------------------------------------------------- #
def renormalize_good_pixels(arr, good_count, expected_count, valid):
# ---------------------------------------------------------------------------- #
    '''
    Rescale for signal lost by masked pixels.
    Divide by good-pixel count, then rescale back up by the expected count.
    '''
    expected_count = np.broadcast_to(expected_count, arr.shape)
    out = arr.copy()
    out[valid] = out[valid] / good_count[valid] * expected_count[valid]
    return out

# ---------------------------------------------------------------------------- #
def extract_fibre_optimal(sci, gpm, flt, gain, read_noise, aperture_weight, dark_rate, exptime, apply_flat_response, n_sigma, n_iter, min_frac_use):
# ---------------------------------------------------------------------------- #
    '''
    Horne 1986 optimal extraction method.
    Default extraction option.
    '''
    # down-weight the flat on the edge rows by how much of each row is really in the fiber
    flt_ap = flt * aperture_weight
    # sum across fibers
    flt_col_sums = flt_ap.sum(axis=0)
    flt_col_sums[flt_col_sums == 0] = 1.0
    # normalised to sum to 1 per column
    P = flt_ap / flt_col_sums

    # variance: source/dark shot noise + read noise
    V = (np.abs(sci) + dark_rate) / (gain * exptime) + read_noise**2 / (gain * exptime)**2
    # avoid division by zero
    V[V <= 0] = 1.0
    # bad pixels get very high variance
    V[gpm == 0] = 1e30
    # down-weight the science data the same way as the flat
    sci_ap = sci * aperture_weight

    good_px = gpm != 0
    weight = 1.0 / V

    # if too few good pixels, set entire column with very high variance
    frac_use = np.sum(np.where(good_px, P, 0.0), axis=0)
    weight[:, frac_use < min_frac_use] = 1e-30

    # run multiple extractions with iterative rejection
    for i in range(n_iter + 1):
        # Horne 1986 Eq. 8
        num = np.sum(P * sci_ap * weight, axis=0)
        den = np.sum(P**2 * weight, axis=0)
        # avoid divide by zero
        good = den > 0
        F = np.zeros_like(den, dtype=np.float32)
        F[good] = num[good] / den[good]  # [counts/s]

        # Horne 1986 Eq. 9: variance of the weighted estimator, [counts/s]^2
        var = np.full_like(den, np.inf, dtype=np.float32)
        var[good] = 1.0 / den[good]

        if i == n_iter:
            break

        # --- rejection: ---
        # model pixels from this iteration's profile
        model = P * F[None, :]
        # compute residuals
        with np.errstate(invalid='ignore', divide='ignore'):
            chi = (sci_ap - model) * np.sqrt(weight)
        outlier = np.abs(chi) > n_sigma
        # if no outliers, stop iterations
        if not outlier.any():
            break
        # test the fragility of the profile if we reject those outliers
        candidate_good = good_px & ~outlier
        candidate_frac_use = np.sum(np.where(candidate_good, P, 0.0), axis=0)
        # unflag any outliers who's rejection would leave a column with too little coverage
        outlier &= (candidate_frac_use >= min_frac_use)[None, :]
        # if all outliers were unflagged, stop iterations
        if not outlier.any():
            break
        # give outliers very high variance in next iteration and remove from good pixels
        weight[outlier] = 1e-30
        good_px &= ~outlier
        # -------------------

    if apply_flat_response:
        # same sum as P's normalisation (these cancel out in flat fielding)
        flt_col_sums = flt_ap.sum(axis=0)
        finite_positive = flt_col_sums[flt_col_sums > 0]
        # exclude non-physical values
        valid_vals = 20 * np.median(finite_positive) if finite_positive.size else np.inf
        valid = (flt_col_sums > 0) & (flt_col_sums < valid_vals)
        flt_norm = np.nanmedian(flt_col_sums[valid]) if valid.any() else 1.0
        response = np.ones_like(flt_col_sums)
        response[valid] = flt_col_sums[valid] / flt_norm
        good_resp = response != 0
        # flat-field the extracted flux and variance
        F[good_resp] /= response[good_resp]
        var[good_resp] /= response[good_resp]**2

    F[~np.isfinite(F)] = 0.0
    var[~np.isfinite(var)] = np.inf
    return F, var

# ---------------------------------------------------------------------------- #
def extract_fibre_boxcar(sciarr, gpmarr, fltarr, min_frac_use):
# ---------------------------------------------------------------------------- #
    '''
    Boxcar extraction with flat-fielding and good-pixel renormalization.
    Secondary extraction option.
    '''
    non_nan = ~np.isnan(sciarr)

    # flat-field extracted spectra
    # Check flat field image array
    if fltarr is not None:
        finite_positive = fltarr[fltarr > 0]
        # exclude non-physical values
        valid_vals = 20 * np.median(finite_positive) if finite_positive.size else np.inf
        valid = (fltarr > 0) & (fltarr < valid_vals)
        flt_norm = np.nanmedian(fltarr[valid]) if valid.any() else 1.0
        response = np.ones_like(fltarr)
        response[valid] = fltarr[valid] / flt_norm
        mask = non_nan & (response != 0)
        # flat-field extracted spectra
        sciarr[mask] /= response[mask]

    # Renormalize by number of good pixels per wavelength bin
    valid = non_nan * (gpmarr != 0)
    sciarr = renormalize_good_pixels(sciarr, gpmarr, gpmarr[non_nan].mean(), valid)

    # too few good pixels to perform renormalization
    low_coverage = gpmarr < min_frac_use * gpmarr[non_nan].mean()
    sciarr[low_coverage] = 0.0

    return sciarr

# ---------------------------------------------------------------------------- #
def get_plot_dir(work, category):
# ---------------------------------------------------------------------------- #
    '''
    Per-exposure, per-category plot directory: <output_dir>/plots/<exposure file stub>/<category>
    '''
    plot_dir = os.path.join(work['output']['dir'], 'plots', work['file'], category)
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

# ---------------------------------------------------------------------------- #
def plot_fibre_extraction(work, id, fibre_type, spectrum, gpm_per_col, variance=None):
# ---------------------------------------------------------------------------- #
    '''
    Top: extracted spectrum for a single fibre, using the exposure's configured extraction method
    (+ 1-sigma band from variance, if given -- boxcar has no noise model, so this is optimal-only).
    Bottom: good-pixel count per wavelength column.
    '''
    cols = np.arange(spectrum.shape[0])

    fig, (axS, axC) = plt.subplots(2, 1, figsize=(10, 4), sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0})

    bundle = 'sky-bundle' if fibre_type == 'sky' else 'object-bundle'
    axS.set_title(f'fibre #{id} ({bundle})', fontsize=15, pad=13)
    if variance is not None:
        sigma = np.sqrt(np.where(np.isfinite(variance) & (variance > 0), variance, np.nan))
        axS.fill_between(cols, spectrum - sigma, spectrum + sigma, step='mid', color='grey', alpha=0.3, linewidth=0.6, zorder=0)
    axS.step(cols, spectrum, where='mid', lw=0.5, color='black')
    axS.axhline(0, linestyle='dashed', lw=0.8, color='grey')
    axS.set_ylabel('[counts / s]', fontsize=12, labelpad=12)
    m = np.isfinite(spectrum)
    axS.set_ylim(-3, np.nanpercentile(spectrum[m], 98))
    plt.setp(axS.get_xticklabels(), visible=False)

    axC.fill_between(cols, 0, gpm_per_col, step='mid', color='grey', alpha=0.8)
    axC.set_ylim(0, gpm_per_col.max() + 1)
    axC.set_ylabel('# good pixels', fontsize=10, labelpad=15)
    axC.set_xlabel('wavelength column', fontsize=14, labelpad=15)

    plot_dir = get_plot_dir(work, 'extracted_spectra')
    out_file = os.path.join(plot_dir, '{0}_extraction_fibre{1}.png'.format(work['file'], id))
    plt.savefig(out_file, dpi=180, format='png', bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------- #
def plot_flat_fielding(work, id, fibre_type, flt_2D, aperture_weight, flux, variance, gpm_per_col):
# ---------------------------------------------------------------------------- #
    '''
    Top: median normalized extracted flat
    Middle: extracted vs. flat fielded flux
    Bottom: good-pixel count per column
    '''
    apply_flat_response = work['flat']['type'][work['exp_type']] == 'flat'

    flt_ap = flt_2D * aperture_weight
    flt_col_sums = flt_ap.sum(axis=0)
    cols = np.arange(flt_col_sums.shape[0])

    fig, (axT, axR, axG) = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={'height_ratios': [3, 3, 1], 'hspace': 0.0})

    bundle = 'sky-bundle' if fibre_type == 'sky' else 'object-bundle'
    axT.set_title(f'fiber #{id} ({bundle})', fontsize=15, pad=15)

    if apply_flat_response:
        finite_positive = flt_col_sums[flt_col_sums > 0]
        # exclude non-physical values
        valid_vals = 20 * np.median(finite_positive) if finite_positive.size else np.inf
        valid = (flt_col_sums > 0) & (flt_col_sums < valid_vals)
        flt_norm = np.nanmedian(flt_col_sums[valid]) if valid.any() else 1.0
        response = np.ones_like(flt_col_sums)
        response[valid] = flt_col_sums[valid] / flt_norm

        axT.step(cols, response, where='mid', lw=0.5, color='black', label='extracted flat')
        # axT.axhline(1, color='red', alpha=0.9, lw=1.0, ls='dashed')
        axT.set_ylabel('median-norm flat', fontsize=12, labelpad=12)
        axT.set_ylim(0.5, 1.5)
    else:
        axT.step(cols, flt_col_sums, where='mid', lw=0.5, color='black', label='extracted flat')
        axT.set_ylabel('flat [counts / s]', fontsize=12, labelpad=12)
        ymin, ymax = axT.get_ylim()
        axT.set_ylim(ymin, ymax + 0.2 * (ymax - ymin))

    for i, amp_edge in enumerate(AMP_BOUNDARIES):
        axT.axvline(amp_edge, color='grey', alpha=0.4, ls='dotted', lw=0.8, label='amp boundary' if i == 0 else None)
    axT.legend(fontsize=10, loc='upper right')
    plt.setp(axT.get_xticklabels(), visible=False)

    sigma = np.sqrt(np.where(np.isfinite(variance) & (variance > 0), variance, np.nan))
    axR.fill_between(cols, flux - sigma, flux + sigma, step='mid', color='grey', alpha=0.4)
    if apply_flat_response:
        extracted = flux * response   # before dividing by the flat
        axR.step(cols, extracted, where='mid', lw=0.8, color='orange', alpha=0.6, zorder=1, label='extracted')
        axR.step(cols, flux, where='mid', lw=0.8, color='blue', alpha=0.5, zorder=2, label='flat-fielded')
        ylim_ref = extracted
    else:
        axR.step(cols, flux, where='mid', lw=0.8, color='blue', alpha=0.5, zorder=2, label='extracted')
        ylim_ref = flux
    axR.set_ylabel('science [counts / s]', fontsize=12, labelpad=12)
    finite_ref = ylim_ref[np.isfinite(ylim_ref)]
    if finite_ref.size:
        axR.set_ylim(-2, np.nanpercentile(finite_ref, 99.9))
    axR.legend(fontsize=10, loc='upper right')
    plt.setp(axR.get_xticklabels(), visible=False)

    axG.fill_between(cols, 0, gpm_per_col, step='mid', color='grey', alpha=0.6)
    axG.set_ylim(0, gpm_per_col.max() + 1)
    axG.set_ylabel('# good pixels', fontsize=8, labelpad=15)
    axG.set_xlabel('wavelength column', fontsize=12, labelpad=10)
    axG.set_xlim(cols[0], cols[-1])

    plot_dir = get_plot_dir(work, 'extracted_spectra')
    out_file = os.path.join(plot_dir, '{0}_flat_response_fibre{1}.png'.format(work['file'], id))
    plt.savefig(out_file, dpi=180, format='png', bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------- #
def extract_fibres(sci, sci_unmasked, gpm, flt, dark_rate, traces, gain, read_noise, exptime, work, log):
# ---------------------------------------------------------------------------- #
    '''
    Extract all fibres from the 2D images using the configured method (optimal or boxcar).

    work['extract_method']:
      'optimal' : Horne 1986 inverse-variance weighted extraction (default).
      'boxcar'  : sum within aperture with good-pixel renormalization.

    Returns (fibres, good_pixels, variances)
    '''
    # Set extraction method (default optimal)
    method = work.get('extract_method', 'optimal')
    flat_type = work['flat']['type'][work['exp_type']]
    apply_flat_response = flat_type == 'flat'
    n_sigma = work['extract']['n_sigma']
    n_iter = work['extract']['n_iter']
    min_frac_use = work['extract']['min_frac_use']

    # optimal extraction requires a flat field for the spatial profile
    if method == 'optimal' and flt is None:

        # arc observations should not be flat-fielded, use boxcar method without flat
        if work['exp_type'] == 'arc':
            method = 'boxcar'
        else:
            raise SALTError("Optimal extraction requires a flat field (flat.type must be 'flat').")


    if method not in ('optimal', 'boxcar'):
        raise SALTError("Unknown extract_method: '{0}' (expected 'optimal' or 'boxcar')".format(method))
    
    # Add message to log
    # log.message(' - extract fibres: method = {0}'.format(method), with_header=False)
    print(' - extract fibres: method = {0}'.format(method))

    # Initialise extracted fibres, good pixels, and variance dictionaries
    fibres, good_pixels, variances = {}, {}, {}

    # Loop for fibre traces...
    for id, fibre in traces.items():

        # Set fibre row range and oversampled 'valid' aperture mask
        r_min, r_max, valid = set_fibre(fibre, work)

        # Good-pixel count per wavelength bin
        gpmarr = set_fibre_array(gpm, r_min, r_max, valid, work)

        # =========================== BOXCAR ===========================
        if method == 'boxcar':
            # Collapse science (gpm applied) over the oversampled aperture
            sciarr = set_fibre_array(sci, r_min, r_max, valid, work)

            fltarr = None
            if apply_flat_response and flt is not None:
                # Extract fibre flat field array
                fltarr = set_fibre_array(flt, r_min, r_max, valid, work)

            # Add fibre flux to extracted fibres dictionary
            fibres[id] = extract_fibre_boxcar(sciarr, gpmarr, fltarr, min_frac_use)
            variances[id] = np.full(fibres[id].shape, np.inf, dtype=np.float32)

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
            dark_2D = dark_rate[r_min:r_max, :].copy()

            # Add fibre flux to extracted fibres dictionary
            fibres[id], variances[id] = extract_fibre_optimal(sci_2D, gpm_2D, flt_2D, gain, read_noise, aperture_weight, dark_2D, exptime, apply_flat_response, n_sigma, n_iter, min_frac_use)

        # Store good-pixel count for this fibre
        good_pixels[id] = gpmarr

    # diagnostic plots for the brightest object-bundle fibres
    if work['output'].get('plot_fibre_extraction') or work['output'].get('plot_flat_response'):
        n_diagnostic_fibers = 10
        obj_ids = [id for id, fibre in traces.items() if fibre.get('type', 'object') != 'sky']
        brightness = {id: np.nanmedian(np.where(fibres[id] > 0, fibres[id], np.nan)) for id in obj_ids}
        diagnostic_ids = sorted(brightness, key=lambda id: brightness[id] if np.isfinite(brightness[id]) else -np.inf)[-n_diagnostic_fibers:]

        if work['output'].get('plot_fibre_extraction'):
            log.message('   - diagnostic plots for fibre extraction', with_header=False)
            for id in diagnostic_ids:
                plot_fibre_extraction(work, id, traces[id].get('type', 'object'), fibres[id], good_pixels[id], variance=variances[id])

        if work['output'].get('plot_flat_response') and method == 'optimal':
            log.message('   - diagnostic plots for flat-fielding', with_header=False)
            for id in diagnostic_ids:
                r_min, r_max, valid = set_fibre(traces[id], work)
                n_rows = r_max - r_min
                n_sub_rows = work['row_repeat']
                n_cols = valid.shape[-1]
                aperture_weight = valid.reshape(n_rows, n_sub_rows, n_cols).mean(axis=1)
                flt_2D = flt[r_min:r_max, :]
                plot_flat_fielding(work, id, traces[id].get('type', 'object'), flt_2D, aperture_weight, fibres[id], variances[id], good_pixels[id])

    return fibres, good_pixels, variances

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