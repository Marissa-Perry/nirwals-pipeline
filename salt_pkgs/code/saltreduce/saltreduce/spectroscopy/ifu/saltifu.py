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
def extract_fibres(image, traces, work, log):
# ---------------------------------------------------------------------------- #

    """
    image: <numpy array> 2D image array
    traces: <dictionary> fibre traces dictionary
    work: <dictionary> work dictionary

    return: <dictionary> extracted fibres dictionary
    """

    # Check tag for log message
    if 'tag' in work and work['tag']:
        # Set log message with tag
        msg = ' - extract fibres: {0}'.format(work['tag'])

    else:
        # Set log message without tag
        msg = ' - extract fibres'

    # Add message to log
    log.message(msg, with_header=False)

    # Initialise extracted fibres dictionary
    fibres = {}

    # Loop for fibre traces...
    for id, fibre in traces.items():

        # Set fibre row range and 'valid' array
        r_min, r_max, valid = set_fibre(fibre, work)

        # Extract fibre science array
        sciarr = set_fibre_array(image, r_min, r_max, valid, work)

        # Add fibre flux and gpm arrays to extracted fibres dictionary
        fibres[id] = sciarr

    return fibres

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