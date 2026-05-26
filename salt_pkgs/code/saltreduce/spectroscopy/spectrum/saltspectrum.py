# ---------------------------------------------------------------------------- #
"""
SALT spectrum utilities:
- saltspectrum provides general utilities for handling SALT spectra:
  - ...
"""
# ---------------------------------------------------------------------------- #

# numpy import
import numpy as np
# scipy import
import scipy.ndimage as nd

# Application imports
from .Spectrum import Spectrum

# ---------------------------------------------------------------------------- #

MYNAME = 'saltspectrum'

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
def artificial_spectrum(sw, sf, **kw):
# ---------------------------------------------------------------------------- #

    """
    Make an artifical spectrum for the given line list wavelengths and fluxes.
    
    sw: <array> 1D wavelength array: line list
    sf: <array> 1D flux array: line list
    kw: <dictionary> additional keyword arguments
        - res: <float>
        - dres: <float>
        - wrange: <list> wavelength range for artificial spectrum

    return wavelength, flux: <arrays>
    """

    # Check wavelength range
    if kw['wrange']:
        # Set wavelength range
        wrange = kw['wrange']

    else:
        # Set wavelength range
        wrange = [sw.min(), sw.max()]

    # Generate artificial spectrum
    asp = Spectrum(
        sw, sf, wrange=wrange, res=kw['res'], dres=kw['dres'], stype='line')

    return asp.wavelength, asp.flux

# ---------------------------------------------------------------------------- #
def find_object_spectra(data, method='median', axis=1, minsize=3, thresh=3.):
# ---------------------------------------------------------------------------- #

    """
    Detect object spectrum (or spectra) in 2-D image

    data: <numpy array> 2D image array
    method: <str> Method for combining the array (median, average or sum)
    axis: <int> Dispersion axis (0=vertical, 1=horizontal)
    minsize: <int> Min size (width if axis=0, height if axis=1) of object
    thresh: <float> Threshold for object detection

    return: <list> Magnitude ordered list of tuples
    """

    # Compress the data (along the dispersion axis):
    # - call method dynamically as attribute of numpy (np)
    ldata = getattr(np, method)(data, axis=axis)
    # Median the data
    ldata = nd.filters.median_filter(ldata, size=minsize)
    # Determine the continuum values (median and median absolute deviation)
    med, mad = np.median(ldata), np.median(abs(ldata - np.median(ldata)))
    # Detect the peak in the distribution
    # obj_arr, obj_num = nd.label(abs(ldata - med) > thresh * mad)
    obj_arr, obj_num = nd.label(ldata - med > thresh * mad)

    # Initialise object lists
    obj_list = []
    mag_list = []

    # Determine the boundries for all objects
    for i in range(1, obj_num + 1):

        ind = np.where(obj_arr == i)[0]
        my1 = ind.min()
        my2 = ind.max()

        if my2 - my1 > minsize and my1 < my2:

            objs = deblend_objs(ldata, my1, my2, minsize)

            for y1, y2 in objs:
####>
                if y2 - y1 > minsize and y1 < y2:

                    obj_list.append((y1, y2))
                    mag_list.append(ldata[y1:y2].max())

                # if 0 < y1 < len(ldata) and 0 < y2 < len(ldata):
                #     if y2 < y1:
                #         y1, y2 = y2, y1

                #     if y2 == y1:
                #         y2 = y1 + 1

                #     obj_list.append((y1, y2))
                #     mag_list.append(ldata[y1:y2].max())
####<
    # Sort the objects in magnitude order
    mag_arr = np.array(mag_list)
    mag_id = mag_arr.argsort()
    obj_list_sorted = []

    for i in mag_id[::-1]:

        obj_list_sorted.append(obj_list[i])

    return obj_list_sorted

# ---------------------------------------------------------------------------- #
def deblend_objs(ldata, y1, y2, minsize=3):
# ---------------------------------------------------------------------------- #

    """
    Deblend a set of objects. Deblend produces a list of y1,y2 for an array
    created by scipy.ndimages.label based on a set of data.
    """

    # Take the gradient of the data
    gdata = np.gradient(ldata[y1:y2])

    # Determine if there is more than one object
    try:
        pos_ind = np.where(gdata >= 0)[0].max()
        neg_ind = np.where(gdata <= 0)[0].min()

    except:
        return [(y1, y2)]

    # If this is true, then there is only a single object to extract
    if abs(pos_ind - neg_ind) < minsize:
        return [(y1, y2)]

    # Manually go through the points and determine where it starts and stops
    obj_list = []
    dy1 = y1
    neg = False

    for i in range(len(gdata)):

        if gdata[i] <= 0 and neg is False:
            neg = True

        if gdata[i] > 0 and neg is True:
            dy2 = dy1 + i
            obj_list.append((dy1, dy2))
            dy1 = dy2 + 1
            neg = False

    obj_list.append((dy1, y2))

    return obj_list

# ---------------------------------------------------------------------------- #
def extract_spectrum(data, extraction_box):
# ---------------------------------------------------------------------------- #

    # Set extraction 'box' values
    y1, y2 = extraction_box

    # Get spectrum:
    # - check extraction 'box' against data 'box'
    y1 = min(len(data), y1)
    y2 = min(len(data), y2)
    # - check if extraction 'box' is a single line
    if abs(y1 - y2) <= 1:
        # - set extraction 'box' as 1 line
        y1 = y2 - 1
####>
    #     # - set extraction 'box' data as line y1
    #     extr = data[y1, :]

    # else:
    #     # - sum the extraction 'box' data
    #     extr = data[y1:y2, :].sum(axis=0)
####<
    # - sum the extraction 'box' data
    extr = data[y1:y2, :].sum(axis=0)
####>
    extr[-1] = extr[-2]
####<
    return extr, y1, y2

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
class SALTError(Exception):
# ---------------------------------------------------------------------------- #

    """Basic exception"""
    pass

# ---------------------------------------------------------------------------- #