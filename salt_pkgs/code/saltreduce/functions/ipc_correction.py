import numpy as np


OVERFLOW_LIMIT = 1e4   # counts/s; same cutoff experiment_bpm.ipynb uses to exclude corrupted/overflow pixels


def ipc_correction_global(image, alpha, bad=None):
    '''
    Approximate first-order inverse of the 4-neighbor IPC kernel, applied to every pixel.
    '''
    good = np.isfinite(image) & (np.abs(image) < OVERFLOW_LIMIT)
    if bad is not None:
        good &= ~bad
    source = np.where(good, image, 0.0)   # a bad pixel can't leak a value onto its neighbors
    neighbor_sum = np.zeros_like(image, dtype=float)
    neighbor_sum[1:, :] += source[:-1, :]    # from the pixel above
    neighbor_sum[:-1, :] += source[1:, :]    # from the pixel below
    neighbor_sum[:, 1:] += source[:, :-1]    # from the pixel to the left
    neighbor_sum[:, :-1] += source[:, 1:]    # from the pixel to the right
    corrected = image + alpha * (4 * image - neighbor_sum)
    return np.where(good, corrected, image)   # bad pixels keep their original value
