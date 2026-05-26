# ---------------------------------------------------------------------------- #
"""
SALT wavelength calibration utilities:
- saltwavelength provides general utilities for wavelength calibration of
  spectral data:
  - ...
"""
# ---------------------------------------------------------------------------- #

# Standard library imports
import os
import copy

# numpy import
import numpy as np
# scipy imports
from scipy.interpolate import splrep
from scipy.interpolate import splev
from scipy.optimize import minimize
from scipy.signal import find_peaks_cwt
from scipy.signal import find_peaks
from scipy.signal import correlate
# **************************************************************************** >
# matplotlib imports
import matplotlib.pyplot as plt
# **************************************************************************** <
# Local application imports
# Application imports
from .WavelengthFit import WavelengthFit

# ---------------------------------------------------------------------------- #

MYNAME = 'saltwavelength'

# ---------------------------------------------------------------------------- #

# Primary fits extension nr
PRIMARY = 0
# Wavelength decimal precision
DEC = 8
# Default kernel to convolve arrays
KERN = [0, -1, -2, -3, -2, -1, 0, 1, 2, 3, 2, 1, 0]

# ---------------------------------------------------------------------------- #
def calculate_wavelength_fit(parr, farr, line_list, wf, log=None, **kw):
# ---------------------------------------------------------------------------- #

    """
    Determine wavelength fit given an observed spectrum and line list.
    Hopefully an accurate first guess (ws) is provided and relative fluxes
    are provided as well, but if not, then the program is still designed
    to attempt to handle it.

    parr: <array> 1D pixel array: observed spectrum
    farr: <array> 1D flux array: observed spectrum
    line_list: <dictionary> line list arrays:
        - swarr: <array> 1D wavelength array: line list
        - sfarr: <array> 1D flux array: line list
        - aswarr: <array> 1D wavelength array: line list artificial spectrum
        - asfarr: <array> 1D flux array: line list artificial spectrum
    wf: <WavelengthFit> initial / guess wavelength fit
    kw: <dictionary> additional keyword arguments:
        - dc: <int> coefficient modulation for matching line list
        - xcor: <dictionary> find_cross_correlation keyword arguments
            - step: <int> number of steps to sample over
            - best: <int> if 1, return best value
                          if 0, return interpolated value
            - int_type: <str> type of interpolation
                              interp: use numpy interp
                              spline: use scipy splrep and splev
        - width: <int> width threshold for detecting line peaks
        - height: <int> height threshold for detecting line peaks
        - sections: <int> nr of sections for detecting line peaks
        - mtol: <float> matched wavelength difference tolerance
        - wtol: <float> wavelength difference tolerance
        - tol: <float> minimum difference tolerance
        - res: <float> minimum residual allowed

    return nwf: <WavelengthFit> wavelength fit
    """

    # Initialise wavelength fit, pixel and wavelength arrays
    nwf = None; p = None; w = None

    # Set wavelength minimum and maximum
    w_min, w_max = wf.value(parr.min()), wf.value(parr.max())

    # Set variables for line list arrays
    swarr = line_list['swarr']
    sfarr = line_list['sfarr']
    aswarr = line_list['aswarr']
    asfarr = line_list['asfarr']

    # Set wavelength range mask for artificial spectrum arrays
    wmask = (aswarr > w_min - kw['pad']) * (aswarr < w_max + kw['pad'])
    # Set wavelength range masked artificial spectrum arrays
    asw, asf = aswarr[wmask], asfarr[wmask]
    # Scale artificial spectrum flux for observed flux
    asf *= farr.max() / asf.max()
    # Set wavelength range mask for line list arrays
    wmask = (swarr > w_min - kw['pad']) * (swarr < w_max + kw['pad'])
    # Set wavelength range masked line list arrays
    sw, sf = swarr[wmask], sfarr[wmask]
####>
    # # Set optimum parameters for finding flux peaks
    # width, height, snr = set_find_peaks_parameters(asf, log=log, **kw)
    # # Update width and height values in keywords dictionary
    # kw['match_line_list']['width'] = width
    # kw['match_line_list']['height'] = height
    # kw['match_line_list']['snr'] = snr
####<
    # Set parameters for finding flux peaks
    width = kw['match_line_list']['width']
    height = kw['match_line_list']['height']
    snr = kw['match_line_list']['snr']
    # Find arc (emission) line peaks
    lines = find_line_peaks(parr, farr, width=width, height=height, snr=snr)

    # Check to see if there are line peaks
    if len(lines):

        # Fit zero point
        wf = fit_zero_point(
            parr, farr, sw, sf, wf, **kw['fit_zero_point'])

        while 1:

            # Match observed spectrum with line list:
            # - p = 1D matched pixel array
            # - w = 1D matched wavelength array
            p, w = match_line_list(parr, farr, sw, sf, asw, asf, wf,
                                   **kw['match_line_list'], **kw['debug'])

            # Set mask for greater than zero wavelengths
            zmask = (w > 0)
            # Check nr of matched lines
            if zmask.sum() >= wf.order:
                # Set masked matched pixel and wavelength arrays
                p, w = p[zmask], w[zmask]
                # Set the wavelength fit
                nwf = WavelengthFit(p, w, function=wf.function, order=wf.order)
####>
                # break
####<
####>
                # # Set wavelength minimum and maximum
                # w_min, w_max = nwf.value(parr.min()), nwf.value(parr.max())                
                # # Set mask for wavelength range edges
                # emask = (sw < w_min + 3.) | (sw > w_max - 3.)
                # # Check if any 'edge' wavelengths
                # if sw[emask].size > 0:
                #     # Check if log is required
                #     if log:
                #         # Add messages to log
                #         msg = ("   - 'edge' wavelength(s) removed: {0}"
                #                 "").format(sw[emask])
                #         log.message(msg, with_header=False)

                #     # Set mask for wavelength
                #     wmask = (sw >= w_min + 3.) & (sw <= w_max - 3.)
                #     # Set line list arrays with 'edge' wavelength(s) removed
                #     sw, sf = sw[wmask], sf[wmask]
####<
####>
                # Check if minimum residual allowed is zero: Break
                if kw['match_line_list']['res'] == 0.: break

                # Calculate residuals of matched wavelengths
                residuals = nwf.res(p, w)
    ####>
                # print('\n   pixel wavelength residual')
                # # Loop for pixels, wavelengths and residuals...
                # for pr, wr, r in zip(p, w, residuals):
                #     print('{0:>8.3f} {1:>10.3f} {2:>8.3f}'.format(pr, wr, r))
                # print('')
    ####<
                # Check if nr of residuals < minimum nr of lines 'needed': Break
                if residuals.size < kw['match_line_list']['min']: break
                # Set mask for 'largest' residual
                mmask = abs(residuals) == abs(residuals).max()
                # Set mask for 'bad' residuals: residual > minimum allowed
                bmask = abs(residuals) > kw['match_line_list']['res']
                # Set 'bad' wavelengths
                bad_wavelengths = w[mmask & bmask]
                # Set 'bad' residuals
                bad_residuals = residuals[mmask & bmask]
                # Check if 'bad' wavelengths were found
                if bad_wavelengths.size > 0:
                    # Set mask to remove 'baddest' wavelength from line list
                    rmask = sw != bad_wavelengths[0]
                    # Set line list arrays with 'baddest' wavelength removed
                    sw, sf = sw[rmask], sf[rmask]
                    # Check if log is required
                    if log:
                        # Add messages to log
                        msg = ("   - 'bad' fit wavelength removed: {0}"
                               " (res = {1:5.3f})"
                               "").format(bad_wavelengths[0], bad_residuals[0])
                        log.message(msg, with_header=False)

                else:
                    break
####<
            else:
                break

    else:
        # Check if log is required
        if log:
            # Add messages to log
            msg = '   - no lines found!'
            log.message(msg, with_header=False)

    return nwf

# ---------------------------------------------------------------------------- #
def fit_zero_point(parr, farr, sw, sf, wf, **kw):
# ---------------------------------------------------------------------------- #

    """
    Fit best zero point using cross-correlation.

    parr: <array> 1D pixel array: observed spectrum
    farr: <array> 1D flux array: observed spectrum
    sw: <array> 1D wavelength array: line list
    sf: <array> 1D flux array: line list
    wf: <WavelengthFit> current wavelength fit
    kw: <dictionary> additional keyword arguments
        - dc: <int> coefficient modulation for fitting zero point
        - xcor: <dictionary> find_cross_correlation keyword arguments
            - step: <int> number of steps to sample over
            - best: <int> if 1, return best value
                          if 0, return interpolated value
            - int_type: <str> type of interpolation
                              interp: use numpy interp
                              spline: use scipy splrep and splev

    return nwf -- new updated wavelength fit
    """

    # Initialise coefficient modulations
    dcoef = wf.coef * 0.
    # Set coefficient modulation: index 0
    dcoef[0] = kw['dc']

    nwf = find_cross_correlation(
        parr, farr, sw, sf, wf, dcoef=dcoef, **kw['xcor'])

    return nwf

# ---------------------------------------------------------------------------- #
def find_cross_correlation(parr, farr, sw, sf, wf, dcoef=None,
                           step=20, best=0, int_type='interp'):
# ---------------------------------------------------------------------------- #

    """
    Find the fit using cross correlation of the wavelength fit.
    An initial guess needs to be supplied along with the modulations in each
    coefficient and the number of steps to calculate the correlation.
    The input wavelength and flux for the known spectral features should be
    in the format where they have already been convolved with the response
    function of the spectrograph.

    parr: <array> 1D pixel array: observed spectrum
    farr: <array> 1D flux array: observed spectrum
    sw: <array> 1D wavelength array: line list
    sf: <array> 1D flux array: line list
    wf: <WavelengthFit> current wavelength fit
    dcoef: <list> modulation over each coefficient for correlation
    step: <int> number of steps to sample over
    best: <int> if 1, return best value
                if 0, return interpolated value
    int_type: <str> type of interpolation
              interp: use numpy interp
              spline: use scipy splrep and splev

    return nwf -- new wavelength fit with best (interpolated) coefficients
    """

    try:
        # Copy existing wavelength fit
        nwf = copy.deepcopy(wf)

    except:
        # Set new wavelength fit
        nwf = WavelengthFit(
            wf.parr, wf.warr, function=wf.function, order=wf.order)

    # Check coefficient modulations
    if dcoef is None: dcoef = wf.coef * 0. + 1.

    # Get list of coefficient modulations
    dlist = modulate_coefficients(wf.coef, dcoef, 0, step)
    # Initialise cross-correlation array
    ccarr = np.zeros(len(dlist), dtype=float)

    # Loop for coefficient modulations...
    for i in range(len(dlist)):

        # Set coeficient
        nwf.set_coef(dlist[i])
        # Set wavelength coverage
        warr = nwf.value(parr)
        # Resample artificial spectrum at same wavelengths as observed
        afarr = interpolate(warr, sw, sf, type=int_type, left=0., right=0.)
        # Calculate normalised correlation value
        ccarr[i] = normalised_correlation(farr, afarr)

    # Set best coefficients
    i = ccarr.argmax()
    bcoef = dlist[i]

    # Check if best is not required
    if not best:
        # Convert list of coefficient modulations to arrray
        darr = np.array(dlist)

        # Loop for coefficients...
        for j in range(len(nwf.coef)):

            if dcoef[j] != 0.:

                i = ccarr.argsort()[::-1]
                tk = np.polyfit(darr[:, j][i[0:5]], ccarr[i[0:5]], 2)

                # Set interpolated value
                if tk[0] == 0:
                    bval = 0

                else:
                    bval = -0.5 * tk[1] / tk[0]

                # Check that best value is close
                if abs(bval - bcoef[j]) < 2 * dcoef[j] / step:
                    # Update coefficient with interpolated value
                    bcoef[j] = bval

                # coef = np.polyfit(dlist[:][j], ccarr, 2)
                # nwf.coef[j] = -0.5 * coef[1] / coef[0]

    # Set coefficients in wavelength fit
    nwf.set_coef(bcoef)

    return nwf

# ---------------------------------------------------------------------------- #
def match_line_list(parr, farr, sw, sf, asw, asf, wf, **kw):
# ---------------------------------------------------------------------------- #

    """
    Match a observed spectrum with a line list.

    parr: <array> 1D pixel array: observed spectrum
    farr: <array> 1D flux array: observed spectrum
    sw: <array> 1D wavelength array: line list
    sf: <array> 1D flux array: line list
    asw: <array> 1D wavelength array: artificial spectrum of line list
    asf: <array> 1D flux array: artificial spectrum of line list
    wf: <WavelengthFit> current wavelength fit
    kw: <dictionary> additional keyword arguments
        - dc: <int> coefficient modulation for matching line list
        - xcor: <dictionary> find_cross_correlation keyword arguments
            - step: <int> number of steps to sample over
            - best: <int> if 1, return best value
                          if 0, return interpolated value
            - int_type: <str> type of interpolation
                              interp: use numpy interp
                              spline: use scipy splrep and splev
        - width: <int> width threshold for detecting line peaks
        - height: <int> height threshold for detecting line peaks
        - sections: <int> nr of sections for detecting line peaks
        - mtol: <float> matched wavelength difference tolerance
        - wtol: <float> wavelength difference tolerance
        - tol: <float> minimum difference tolerance

    return  ar(p_list), arr(w_list) -- peak pixel position and matched
                                       wavelength lists as arrays
    """

    # Initialise wavelength array
    warr = wf.value(parr)

    # ************************************************************************ >
    # Check debug
    if kw['wavelength_range']:
        # Set wavelength range
        wr = '{0:>9.3f} {1:>9.3f}'.format(warr.min(), warr.max())
        # Print wavelength range to screen
        print('')
        print('   - WAVELENGTH RANGE: {0}'.format(wr))
    # ************************************************************************ <

    # Find peak positions and fluxes
    p, f = find_flux_peaks(parr, farr, kw['width'], kw['height'], kw['snr'],
                           sections=kw['sections'])

    # ************************************************************************ >
    # Check debug
    if kw['flux_peaks']:
        # Print flux peaks to screen
        print('')
        print('   - FLUX PEAKS:\n')
        print('        pixel       flux')
        for pi, fi in zip(p.tolist(), f.tolist()):
            print('    {0:>9.3f} {1:>10.3f}'.format(pi, fi))
        print('')
        # Plot flux peaks
        plt.figure(1, figsize=(16, 9), tight_layout=True)
        plt.plot(parr, farr, lw=1., ls='-', c='blue')
        for pix in p:
            plt.axvline(x=pix, ls='--', lw=0.5, c='black', ymin=0., ymax=1.)
            # Set xy position of label
            x, y = pix, plt.ylim()[1]
            # Add label for line list wavelength
            plt.annotate('{0:.3f}'.format(pix), (x, y),
                         textcoords='offset points', xytext=[0,10],
                         ha='center', fontsize=8, rotation='vertical')

        plt.show()
        plt.close()
    # ************************************************************************ <

    # Initialise coefficient modulations
    dcoef = wf.coef * 0.
    # Set coefficient modulation: index 0
    dcoef[0] = kw['dc']

    # Initialise peak pixel position and matched wavelength lists
    p_list = []
    w_list = []

    # ************************************************************************ >
    # Check debug
    if kw['line_list']:
        # Print line list to screen (sorted descending)
        print('')
        print('   - LINE LIST (sorted descending):\n')
        print('       lambda       flux   i')
        for i in sf.argsort()[::-1]:
            # Skip sf[i] == 0.
            if sf[i] == 0.: continue
            print('    {0:>9.3f} {1:>10.4f} {2:>3d}'.format(sw[i], sf[i], i))
        print('')
    # ************************************************************************ <

    # Loop for line list flux indices (sorted descending)...
    for i in sf.argsort()[::-1]:

        # Skip sf[i] == 0.
        if sf[i] == 0.: continue

        # Exclude lines outside of observed range
        if sw[i] < warr.max() and sw[i] > warr.min():
            # Set wavelength difference mask for observed spectrum
            omask = abs(warr - sw[i]) < kw['wtol']
            # Set wavelength difference mask for artificial spectrum
            smask = abs(asw - sw[i]) < kw['wtol']

            # **************************************************************** >
            # Check debug
            if kw['spectrum_region']:
                # Print observed spectrum region to screen
                print('   - OBSERVED SPECTRUM REGION FOR i={0}:\n'.format(i))
                print('        pixel       flux')
                for pi, fi in zip(parr[omask].tolist(), farr[omask].tolist()):
                    print('    {0:>9d} {1:>10.3f}'.format(int(pi), fi))
                print('')
                # Print artificial spectrum region to screen
                print('   - ARTIFICIAL SPECTRUM REGION FOR i={0}:\n'.format(i))
                print('       lambda       flux')
                for wi, fi in zip(asw[smask].tolist(), asf[smask].tolist()):
                    print('    {0:>9.3f} {1:>10.3f}'.format(wi, fi))
                print('')
            # **************************************************************** <

            # Use wavelength difference region to do cross correlation
            nwf = find_cross_correlation(parr[omask], farr[omask],
                                         asw[smask], asf[smask], wf,
                                         dcoef=dcoef, **kw['xcor'])
            # Generate wavelength array from cross correlation fit
            nwp = nwf.value(p)
            # Set difference with line list wavelength
            d = abs(nwp - sw[i])

            # **************************************************************** >
            # Check debug
            if kw['match_line']:
                print('')
                if d.min() >= kw['tol']:
                    print(('    i: {0:>2d}, sw[i]: {1:>9.3f}, '
                           'd.min(): {2:>6.3f}').format(i, sw[i], d.min()))
            # **************************************************************** <

            # Check if minimum difference is less than required tolerance
            if d.min() < kw['tol']:
                # Set index of smallest difference
                j = d.argmin()
                # Check if rank orders of lines match
                matched = line_order(p, f, sw, sf, nwf, sw[i], p[j], kw['wtol'])

                # ************************************************************ >
                # Check debug
                if kw['match_line']:
                    print(('    i: {0:>2d}, sw[i]: {1:>9.3f}, '
                           'd.min(): {2:>6.3f}, j: {3:>2d}, p[j]: {4:>8.3f}, '
                           'abs(wf.value(p[j]) - sw[i]): {5:>6.3f}, match: {6}'
                           '').format(i, sw[i], d.min(), j, p[j],
                                      abs(wf.value(p[j]) - sw[i]),
                                      matched[0]))
                # ************************************************************ <

                # Check if matched and difference is less than match tolerance
                if matched and abs(wf.value(p[j]) - sw[i]) < kw['mtol']:
                    # Add pixel and wavelength values to lists
                    p_list.append(p[j])
                    w_list.append(sw[i])

    # ************************************************************************ >
    # Check debug
    if kw['match_line']:
        print('')
    # ************************************************************************ <

    # Convert matched pixel positions and wavelengths lists to arrays
    pixels = np.array(sorted(p_list))
    wavelengths = np.array(sorted(w_list))

    return pixels, wavelengths

# ---------------------------------------------------------------------------- #
def find_flux_peaks(parr, farr, width, height, snr, sections=0):
# ---------------------------------------------------------------------------- #

    """
    Find all peak positions and fluxes in a spectrum.

    parr: <array> 1D pixel array: observed spectrum
    farr: <array> 1D flux array: observed spectrum
    width: <int> width threshold for detecting line peaks
    height: <int> height threshold for detecting line peaks
    snr: <float> minimum snr for detecting line peaks
    sections: <int> nr of sections for detecting line peaks in sections

    return p, f: <arrays> peak positions and fluxes
    """

    # Check if peaks must be found in sections
    if sections > 0:
        # Set section size
        s = len(parr) / sections

        # Initialise peak array
        p = None
        # Initialise section boundaries
        p1, p2 = 0, 0

        # Loop for nr of sections...
        for i in range(sections):

            # Set section boundaries
            p1 = max(int(i * s), p2)
            p2 = min(int(p1 + s) + 1, len(parr))
            # Set section pixel and flux arrays
            sparr = parr[p1:p2]
            sfarr = farr[p1:p2]
            # Find peak positions in section
            ps = find_line_peaks(sparr, sfarr, width=width, height=height,
                                 snr=snr, centre=True)
            # Add to peak array
            if p is None:
                p = ps.copy()

            else:
                p = np.concatenate((p, ps))

    else:
        # Find peak positions
        p = find_line_peaks(parr, farr, width=width, height=height,
                            snr=snr, centre=True)

    # Set peak fluxes
    f = farr[p.astype(int)]

    return p, f

# ---------------------------------------------------------------------------- #
def set_find_peaks_parameters(asf, log=None, **kw):
# ---------------------------------------------------------------------------- #

    """
    Set optimum parameters (width and height) for finding peaks in a 1D spectrum

    asf: <array> 1D flux array: line list artificial spectrum
    kw: <dictionary> additional keyword arguments:
        - ..
        - width: <int> width threshold for detecting line peaks
        - height: <int> height threshold for detecting line peaks
        - snr: <float> minimum SNR ratio
        - ..

    return width, height: <int> find peaks parameters
    """

    width = kw['match_line_list']['width']
    height = kw['match_line_list']['height']
    snr = kw['match_line_list']['snr']
####>
    # # Find all peaks in artificial spectrum flux: scipy.signal.find_peaks
    # peaks = find_peaks(asf, width=width)
    # # Get minimum of peak widths
    # min_width = int(peaks[1]['widths'].min())
    # # Get minimum of peak base heights
    # min_height = int(peaks[1]['width_heights'].min())
    # # Check if log is required
    # if log:
    #     # Add messages to log
    #     msg = '   - default peak width: {0}'.format(width)
    #     log.message(msg, with_header=False)
    #     msg = '   - minimum peak width: {0}'.format(min_width)
    #     log.message(msg, with_header=False)
    #     msg = '   - default peak height: {0}'.format(height)
    #     log.message(msg, with_header=False)
    #     msg = '   - minimum peak height: {0}'.format(min_height)
    #     log.message(msg, with_header=False)

    # # Set height
    # height = max(min_height, height)
####<
    return width, height, snr

# ---------------------------------------------------------------------------- #
def find_line_peaks(parr, farr, width=5, height=150, snr=2.,
                    centre=False, kern=KERN):
# ---------------------------------------------------------------------------- #

    """
    Find line (emission) peaks in a 1D spectrum

    parr: <array> 1D pixel array: observed spectrum
    farr: <array> 1D flux array: observed spectrum
    width: <int or list> width threshold for detecting line peaks
    height: <int> height threshold for detecting line peaks
    snr: <float> minimum snr for detecting line peaks
    centre: <bool> if True return centroids and not pixels
    kern: <list> kernel to convolve the array with

    return p: <array> peak positions
    """
####>
    # # Find all peaks: scipy.signal.find_peaks
    # peaks = find_peaks(farr, height=height, width=width)
    # # Set peak pixels array
    # p = peaks[0]
####<
####>
    # Set widths array
    widths = np.array([width])
    # Find all peaks: scipy.signal.find_peaks_cwt
    p = find_peaks_cwt(farr, widths=widths, min_snr=snr)
    # Convert peak pixels to array
    p = np.array(p)
####<
    # Check if centroids must be returned
    if centre:
        # Convert peak pixels array to float
        p = p * 1.
        diff = int(0.5 * len(kern) + 1)

        # Loop for nr of peaks...
        for i in range(len(p)):

            # Set guess for peak
            guess = parr[int(p[i])]
            # Get peak centroid
            p[i] = centroid(parr, farr, kern=kern, guess=guess, diff=diff)

    return p

# ---------------------------------------------------------------------------- #
def normalised_correlation(x, y):
# ---------------------------------------------------------------------------- #

    """
    Calculate normalised correlation of two arrays.
    """

    d = np.correlate(x, x) * np.correlate(y, y)
    if d <= 0:
        return 0

    return np.correlate(x, y) / d ** 0.5

# ---------------------------------------------------------------------------- #
def optimise_fit(parr, farr, asw, asf, wf, int_type='interp',
                 method='Nelder-Mead'):
# ---------------------------------------------------------------------------- #

    """
    Optimise the normalised cross correlation coefficient for the full
    wavelength fit.
    """

    try:
        # Copy existing wavelength fit
        nwf = copy.deepcopy(wf)

    except:
        # Set new wavelength fit
        nwf = WavelengthFit(
            wf.parr, wf.warr, function=wf.function, order=wf.order)

    # Minimise coefficients
    bcoef = minimize(min_func, nwf.coef, method=method,
                     args=(parr, farr, asw, asf, nwf, int_type))['x']

    # Set coefficients in wavelength fit
    nwf.set_coef(bcoef)

    return nwf

# ---------------------------------------------------------------------------- #
def min_func(coef, parr, farr, asw, asf, wf, int_type):
# ---------------------------------------------------------------------------- #

    """
    Minimising function.
    """

    # Set wavelength fit coefficients
    wf.set_coef(coef)
    # Set wavelength array
    warr = wf.value(parr)
    # Resample artificial spectrum at same wavelengths as observed spectrum
    rasf = interpolate(warr, asw, asf, type=int_type, left=0., right=0.)

    return abs(1. / normalised_correlation(farr, rasf))

# ---------------------------------------------------------------------------- #
def centroid(parr, farr, guess=None, diff=None, kern=KERN, mode='same'):
# ---------------------------------------------------------------------------- #

    """
    Find the centroid of a line following a similar algorithm as the
    centroid algorithm in IRAF.   parr and farr should be an area
    around the desired feature to be centroided.  The default kernel
    is used if one is not specified.

    The algorithm solves for the solution to the equation

    ..math:: \int (I-I_0) f(x-x_0) dx = 0

    parr: <array> 1D pixel array: observed spectrum
    farr: <array> 1D flux array: observed spectrum
    guess: <int> initial guess (pixel)
    diff: <int> nr of pixels around guess to use for convolution
    kern: <list> kernel to convolve array
    mode: <str> mode of convolution
        full: output shape of (N+M-1,)
        same: output of length max(M, N)
        valid: output of length max(M, N) - min(M, N) + 1

    return c: <float> centroid
    """

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
def interpolate(x, xarr, yarr, type='interp', order=3, left=None, right=None):
# ---------------------------------------------------------------------------- #

    """
    Perform interpolation on value x using arrays xarr and yarr.

    type: <str> type of interpolation
        interp: use numpy interp
        spline: use scipy splrep and splev
    order: <int> order of spline
    left: <float> value to return for x < xarr[0]
    right: <float> value to return for x > xarr[-1]

    return y: <float> interpolated value
    """

    if type == 'interp':
        y = np.interp(x, xarr, yarr, left=left, right=right)

    elif type == 'spline':
        tck = splrep(xarr, yarr, k=order)
        y = splev(x, tck, der=0)

    return y

# ---------------------------------------------------------------------------- #
def modulate_coefficients(coef, dcoef, index, step):
# ---------------------------------------------------------------------------- #

    """
    For a given index, return a list of modulations in that coefficient.
    """

    dlist = []

    # Check if last coefficient has been reached
    if index >= len(coef):
        return dlist

    # If coefficient doesn't need modulation move on to next coefficient
    if dcoef[index] == 0:
        if index < len(coef) - 1:
            dlist.extend((modulate_coefficients(coef, dcoef, index + 1, step)))

        else:
            dlist.append(coef)

        return dlist

    # If coefficient does need modulations proceed in one of two ways:
    # - if it isn't last coefficient, iterate over values and step down to
    #   do all other coefficients
    # - if it is last coefficient, iterate over values and create lowest
    #   level coefficient
    if index < len(coef) - 1:

        for x in np.arange(-dcoef[index], dcoef[index],
                            2 * dcoef[index] / float(step)):

            ncoef = coef.copy()
            ncoef[index] = coef[index] + x
            dlist.extend(modulate_coefficients(ncoef, dcoef, index + 1, step))

    else:

        for x in np.arange(-dcoef[index], dcoef[index],
                            2 * dcoef[index] / float(step)):

            ncoef = coef.copy()
            ncoef[index] = coef[index] + x
            dlist.append(ncoef)

    return dlist

# ---------------------------------------------------------------------------- #
def line_order(p, f, sw, sf, wf, swv, opv, tol):
# ---------------------------------------------------------------------------- #

    """
    Determine the rank order of spectral line from the line list and that of
    the observed line.

    p: <array> 1D peak pixel positions array
    f: <array> 1D peak fluxes array

    parr: <array> 1D pixel array: observed spectrum
    farr: <array> 1D flux array: observed spectrum
    sw: <array> 1D wavelength array: line list
    sf: <array> 1D flux array: line list
    wf: <WavelengthFit> current wavelength fit
    swv: <float> spectral line wavelength value
    opv: <int> observed line pixel position
    tol: <float> wavelength difference tolerance for line order check

    return True if orders match
    """

    # Set mask for the spectral line
    smask = abs(sw - swv) < tol
    # Identify the order of the spectral line
    i = sf[smask].argsort()
    i_ord = i[sw[smask][i] == swv]
####>
    # print('i:', i, 'i_ord:', i_ord)
####<
    if len(i_ord) > 1:
        return False

    # Set mask for the observed line
    omask = abs(wf.value(p) - swv) < tol
    # Identify the order of the observed line
    j = f[omask].argsort()
    j_ord = j[p[omask][j] == opv]
####>
    # print('j:', j, 'j_ord:', j_ord)
####<
    if len(j_ord) > 1:
        return False

    return i_ord == j_ord

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
class SALTError(Exception):
# ---------------------------------------------------------------------------- #

    """Basic exception"""
    pass

# ---------------------------------------------------------------------------- #

# # ---------------------------------------------------------------------------- #
# def calibrate_2D_model(image_arr, axis, spectrograph):
# # ---------------------------------------------------------------------------- #

#     """
#     Wavelength calibrate 2D image array using spectrograph model.

#     image_arr: <array> 2D image array
#     axis: <integer> dispersion axis (0=vertical, 1=horizontal)
#     spectrograph: <class> initialised spectrograph class

#     return image_arr: <array> recitified and wavelength calibrated image array
#     """

#     # Set rows and columns
#     rows, cols = image_arr.shape

#     # Check if dispersion is vertical
#     if axis == 0:
#         # Set nr of pixel (wavelength) and flux points
#         nw = rows
#         nf = cols

#     # Else check if dispersion is horizontal
#     elif axis == 1:
#         # Set nr of pixel (wavelength) and flux points
#         nw = cols
#         nf = rows

#     # Set pixel array
#     parr = np.arange(nw, dtype=float)

#     # Set 'centre' row (column) for lower (left) half...
#     centre = int(nf / 2) - 1
#     # Wavelength calibrate the rows (columns) from the centre down (left)
#     image_arr, w1, dw = calibrate_model(
#         axis, image_arr, parr, centre, -1, -1, spectrograph)

#     # Set 'centre' row (column) for upper (right) half
#     centre = int(nf / 2)
#     # Wavelength calibrate the rows (columns) from the centre up (right)
#     image_arr, w1, dw = calibrate_model(
#         axis, image_arr, parr, centre, nf, 1, spectrograph)

#     # Set wavelength array
#     w = w1 + dw * parr
#     # Round the wavelength array
#     w = np.around(w, decimals=DEC)

#     return image_arr, w, dw

# # ---------------------------------------------------------------------------- #
# def calibrate_1D_model(flux_arr, axis, j, spectrograph):
# # ---------------------------------------------------------------------------- #

#     """
#     Wavelength calibrate 1D flux array using spectrograph model.

#     flux_arr: <array> 1D flux array
#     axis: <integer> dispersion axis (0=vertical, 1=horizontal)
#     j: <integer> column (axis=0) or row (axis=1) nr in 2D image array
#     spectrograph: <class> initialised spectrograph class

#     return flux_arr: <array> recitified and wavelength calibrated flux array
#     """

#     # Set nr of pixel (wavelength) points
#     nw = flux_arr.shape[0]
#     # Set pixel (wavelength) array
#     parr = np.arange(nw, dtype=float)

#     # Wavelength calibrate the row (column)
#     flux_arr, w1, dw = calibrate_model(
#         axis, flux_arr, parr, j, j+1, 1, spectrograph)

#     # Set wavelength array
#     w = w1 + dw * parr
#     # Round the wavelength array
#     w = np.around(w, decimals=DEC)

#     return flux_arr, w, dw

# # ---------------------------------------------------------------------------- #
# def calibrate_model(axis, farr, parr, start, stop, step, spectrograph):
# # ---------------------------------------------------------------------------- #

#     # Loop for rows (columns) (from start to stop with step)...
#     for j in range(start, stop, step):

#         # Get model wavelength for row (column) j
#         w = 1e7 * spectrograph.get_wavelength(parr, j=j, axis=axis)
#         w = np.around(w, decimals=DEC)
#         # Check if it's the starting (i.e., 'centre') row (column)
#         if j == start:
#             # Get 'centre' (evenly spaced) interpolation wavelength array
#             cw, w1, dw = get_evenly_spaced_array(w.min(), w.max(), w.size)

#         # Set flux array, i.e., the flux data of row (column) j
#         if farr.ndim == 1:
#             f = farr

#         else:
#             # Check if dispersion is vertical
#             if axis == 0:
#                 f = farr[:, j]
#             # Else check if dispersion is horizontal
#             elif axis == 1:
#                 f = farr[j, :]

#         # Apply wavelength fit to the flux data of row (column) j
#         f = np.interp(cw, w, f, left=0., right=0.)
#         # Update row (column) j in flux data array
#         if farr.ndim == 1:
#             farr = f

#         else:
#             # Check if dispersion is vertical
#             if axis == 0:
#                 farr[:, j] = f
#             # Else check if dispersion is horizontal
#             elif axis == 1:
#                 farr[j, :] = f

#     return farr, w1, dw

# # ---------------------------------------------------------------------------- #
# def calibrate_2D(image_arr, axis, spectrograph, line_list, **kw):
# # ---------------------------------------------------------------------------- #

#     """
#     Wavelength calibrate 2D image array using wavelength line list.

#     image_arr: <array> 2D image array
#     axis: <integer> dispersion axis (0=vertical, 1=horizontal)
#     spectrograph: <class> initialised spectrograph class
#     line_list: <dictionary> line list arrays:
#         - swarr: <array> 1D wavelength array: line list
#         - sfarr: <array> 1D flux array: line list
#         - aswarr: <array> 1D wavelength array: line list artificial spectrum
#         - asfarr: <array> 1D flux array: line list artificial spectrum
#     kw: <dictionary> additional wavelength calibration keyword arguments

#     return image_arr: <array> recitified and wavelength calibrated image array
#     """

#     # Set rows and columns
#     rows, cols = image_arr.shape

#     # Check if dispersion is vertical
#     if axis == 0:
#         # Set nr of wavelength and flux points
#         nw = rows
#         nf = cols

#     # Else check if dispersion is horizontal
#     elif axis == 1:
#         # Set nr of wavelength and flux points
#         nw = cols
#         nf = rows

#     # Set pixel array
#     parr = np.arange(nw, dtype=float)

#     # Set 'centre' row (column) for lower (left) half...
#     centre = int(nf / 2) - 1
#     # Wavelength calibrate the rows (columns) from the centre down (left)
#     image_arr, w1, dw = calibrate(
#         axis, image_arr, parr, centre, -1, -1, spectrograph, line_list, **kw)

#     # Set 'centre' row (column) for upper (right) half
#     centre = int(nf / 2)
#     # Wavelength calibrate the rows (columns) from the centre up (right)
#     image_arr, w1, dw = calibrate(
#         axis, image_arr, parr, centre, nf, 1, spectrograph, line_list, **kw)

#     # Set wavelength array
#     w = w1 + dw * parr
#     # Round the wavelength array
#     w = np.around(w, decimals=DEC)

#     return image_arr

# # ---------------------------------------------------------------------------- #
# def calibrate_1D(flux_arr, axis, j, spectrograph, line_list, **kw):
# # ---------------------------------------------------------------------------- #

#     """
#     Wavelength calibrate 1D flux array using wavelength line list.

#     flux_arr: <array> 1D flux array
#     axis: <integer> dispersion axis (0=vertical, 1=horizontal)
#     j: <integer> column (axis=0) or row (axis=1) nr in 2D image array
#     spectrograph: <class> initialised spectrograph class
#     line_list: <dictionary> line list arrays:
#         - swarr: <array> 1D wavelength array: line list
#         - sfarr: <array> 1D flux array: line list
#         - aswarr: <array> 1D wavelength array: line list artificial spectrum
#         - asfarr: <array> 1D flux array: line list artificial spectrum
#     kw: <dictionary> additional wavelength calibration keyword arguments

#     return flux_arr: <array> recitified and wavelength calibrated flux array
#     """

#     # Set nr of pixel (wavelength) points
#     nw = flux_arr.shape[0]
#     # Set pixel (wavelength) array
#     parr = np.arange(nw, dtype=float)

#     flux_arr, w1, dw = calibrate(
#         axis, flux_arr, parr, j, j+1, 1, spectrograph, line_list, **kw)

#     # Set wavelength array
#     w = w1 + dw * parr
#     # Round the wavelength array
#     w = np.around(w, decimals=DEC)

#     return flux_arr, w, dw

# # ---------------------------------------------------------------------------- #
# def calibrate(axis, farr, parr, start, stop, step, spectrograph,
#               line_list, **kw):
# # ---------------------------------------------------------------------------- #

#     # Set wavelength fit function
#     function = kw['WavelengthFit']['function']
#     # Set wavelength fit function order
#     order = kw['WavelengthFit']['order']

#     # Loop for rows (columns) (from start to stop with step)...
#     for j in range(start, stop, step):

#         # Set flux array, i.e., the flux data of row (column) j
#         if farr.ndim == 1:
#             f = farr

#         else:
#             # Check if dispersion is vertical
#             if axis == 0:
#                 f = farr[:, j]
#             # Else check if dispersion is horizontal
#             elif axis == 1:
#                 f = farr[j, :]

#         # Get model wavelength for row (column) j
#         w = 1e7 * spectrograph.get_wavelength(parr, j=j, axis=axis)
#         w = np.around(w, decimals=DEC)
#         # Initialise wavelength fit
#         wf = WavelengthFit(parr, w, function=function, order=order)
#         # Calculate wavelength fit for row (column) j
#         wf = calculate_wavelength_fit(parr, f, line_list, wf, **kw)
#         # Set wavelength array for fit
#         w_fit = wf.value(parr)
#         # Round the wavelength array
#         w = np.around(w_fit, decimals=DEC)

#         # Check if it's the starting (i.e., 'centre') row (column)
#         if j == start:
#             # Get 'centre' (evenly spaced) interpolation wavelength array
#             cw, w1, dw = get_evenly_spaced_array(w.min(), w.max(), w.size)

#         # Apply wavelength fit to the flux data of row (column) j
#         f = np.interp(cw, w, f, left=0., right=0.)
#         # Update row (column) j in flux data array
#         if farr.ndim == 1:
#             farr = f

#         else:
#             # Check if dispersion is vertical
#             if axis == 0:
#                 farr[:, j] = f
#             # Else check if dispersion is horizontal
#             elif axis == 1:
#                 farr[j, :] = f

#     return farr, w1, dw

# # ---------------------------------------------------------------------------- #