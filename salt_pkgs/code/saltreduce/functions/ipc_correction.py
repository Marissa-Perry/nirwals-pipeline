import numpy as np
from scipy.ndimage import convolve


OVERFLOW_LIMIT = 1e4   # counts/s; same cutoff experiment_bpm.ipynb uses to exclude corrupted/overflow pixels

# 0-indexed right edge of each 64-column amplifier readout region (fixed detector hardware property)
AMP_BOUNDARIES = np.array([64., 128., 192., 256., 320., 384., 448., 512., 576., 640., 704., 768.,
                            832., 896., 960., 1024., 1088., 1152., 1216., 1280., 1344., 1408.,
                            1472., 1536., 1600., 1664., 1728., 1792., 1856., 1920., 1984., 2048.]) - 1


def amp_reads_left_to_right(n_cols):
    '''
    True at every column whose amplifier reads left-to-right, False otherwise.
    '''
    amp_index = np.searchsorted(AMP_BOUNDARIES, np.arange(n_cols), side='left')
    return (amp_index % 2) == 1   # bright amp-boundary convention (readout directions point toward each other) -> odd amp index reads left-to-right


def ipc_correction_global(image, alpha_asym_3, alpha_asym_1, bad=None, trusted_source=None):
    '''
    Approximate first-order inverse of the 4-neighbor IPC kernel, applied to every pixel.
    '''
    alpha_asym_3 = np.broadcast_to(alpha_asym_3, image.shape).astype(float)   # up/down + the row-neighbor read after the source pixel; scalar or per-pixel
    alpha_asym_1 = np.broadcast_to(alpha_asym_1, image.shape).astype(float)   # the row-neighbor read before the source pixel; scalar or per-pixel
    # a source pixel's own amp direction decides which of its row-neighbors was read first
    l_to_r = amp_reads_left_to_right(image.shape[1])
    alpha_toward_left = np.where(l_to_r, alpha_asym_1, alpha_asym_3)
    alpha_toward_right = np.where(l_to_r, alpha_asym_3, alpha_asym_1)

    overflow_ok = np.isfinite(image) & (np.abs(image) < OVERFLOW_LIMIT)
    good = overflow_ok.copy()
    if bad is not None:
        good &= ~bad
    source_ok = good.copy()
    if trusted_source is not None:
        source_ok |= (trusted_source & overflow_ok)   # trusted: value still known well enough to leak, even though flagged bad
    source = np.where(source_ok, image, 0.0)   # an untrusted pixel can't leak a value onto its neighbors
    source_alpha_col = np.where(source_ok, alpha_asym_3, 0.0)   # an untrusted pixel's own coupling strength can't leak either
    source_alpha_left = np.where(source_ok, alpha_toward_left, 0.0)
    source_alpha_right = np.where(source_ok, alpha_toward_right, 0.0)
    neighbor_sum = np.zeros_like(image, dtype=float)
    neighbor_sum[1:, :] += source_alpha_col[:-1, :] * source[:-1, :]    # from the pixel above
    neighbor_sum[:-1, :] += source_alpha_col[1:, :] * source[1:, :]    # from the pixel below
    neighbor_sum[:, 1:] += source_alpha_right[:, :-1] * source[:, :-1]    # from the pixel to the left, leaking rightward into this one
    neighbor_sum[:, :-1] += source_alpha_left[:, 1:] * source[:, 1:]    # from the pixel to the right, leaking leftward into this one
    corrected = image + (alpha_toward_left + alpha_toward_right + 2 * alpha_asym_3) * image - neighbor_sum
    return np.where(good, corrected, image)   # bad pixels keep their original value


def isolated_hot_pixels(data, threshold, radius=5):
    '''Pixels above threshold, and the subset with no other such pixel within radius pixels.'''
    hot_pixel = np.isfinite(data) & (data > threshold)
    box = np.ones((2 * radius + 1, 2 * radius + 1))
    nearby = convolve(hot_pixel.astype(float), box, mode='constant', cval=0.0)
    return hot_pixel, hot_pixel & (nearby == 1)


def build_alpha_maps(dark, med, candidates, fallback_alpha_asym_3, fallback_alpha_asym_1, min_measured=200):
    '''
    Per-pixel alpha_asym_3, alpha_asym_1, and a diagnostic alpha_sym, measured from isolated hot pixels in dark.
    '''
    clean = np.where(np.abs(dark) < OVERFLOW_LIMIT, dark, np.nan)
    l_to_r = amp_reads_left_to_right(dark.shape[1])
    all_asym_3_ests, all_asym_1_ests = [], []   # every candidate's raw estimate, for the frame-median fallback below
    coords, asym_3_ests, asym_1_ests, sym_ests = [], [], [], []   # ones that pass coupling limits, for per-pixel assignment
    for r, c in zip(*np.where(candidates)):
        if r < 1 or r > dark.shape[0] - 2 or c < 1 or c > dark.shape[1] - 2:
            continue   # skip pixels at the detector edge
        if not (l_to_r[c - 1] == l_to_r[c] == l_to_r[c + 1]):
            continue   # within an amplifier boundary "read first" isn't meaningful
        patch = clean[r-1:r+2, c-1:c+2] - med
        central = patch[1, 1]
        left_val, right_val, up_val, down_val = patch[1, 0], patch[1, 2], patch[0, 1], patch[2, 1]
        if not np.isfinite(central) or central <= 0 or not np.all(np.isfinite([left_val, right_val, up_val, down_val])):
            continue
        left_est, right_est = left_val / central, right_val / central
        up_est, down_est = up_val / central, down_val / central
        asym_1_est = left_est if l_to_r[c] else right_est   # left-to-right amp: left neighbor is read first
        after_est = right_est if l_to_r[c] else left_est        # the other row-neighbor, read after the source pixel
        asym_3_est = (up_est + down_est + after_est) / 3        # mean of 3 other neighbors
        sym_est = (up_est + down_est + left_est + right_est) / 4   # all 4 neighbors equally weighted, ignoring the split
        all_asym_3_ests.append(asym_3_est)
        all_asym_1_ests.append(asym_1_est)
        # reasonable coupling fraction limits
        if 0 < asym_3_est < 0.125 and 0 < asym_1_est < 0.125: 
            coords.append((r, c))
            asym_3_ests.append(asym_3_est)
            asym_1_ests.append(asym_1_est)
            sym_ests.append(sym_est)

    if len(all_asym_3_ests) >= min_measured:
        default_alpha_asym_3 = np.median(all_asym_3_ests)
        default_alpha_asym_1 = np.median(all_asym_1_ests)
    else:
        default_alpha_asym_3 = fallback_alpha_asym_3   # too few measured pixels to trust this frame's own median
        default_alpha_asym_1 = fallback_alpha_asym_1

    alpha_asym_3_map = np.full(dark.shape, default_alpha_asym_3, dtype=float)
    alpha_asym_1_map = np.full(dark.shape, default_alpha_asym_1, dtype=float)
    alpha_sym_map = np.full(dark.shape, np.median(sym_ests) if sym_ests else fallback_alpha_asym_3, dtype=float)   # diagnostic only, never fed into the correction
    measured = np.zeros(dark.shape, dtype=bool)   # these pixels can still act as leak sources even where otherwise masked out
    for (r, c), asym_3_est, asym_1_est, sym_est in zip(coords, asym_3_ests, asym_1_ests, sym_ests):
        alpha_asym_3_map[r, c] = asym_3_est
        alpha_asym_1_map[r, c] = asym_1_est
        alpha_sym_map[r, c] = sym_est
        measured[r, c] = True
    return alpha_asym_3_map, alpha_asym_1_map, alpha_sym_map, measured, default_alpha_asym_3, default_alpha_asym_1


def expected_dark_noise(background_rate, exptime, n_frames, gain, read_noise_e):
    '''
    Expected shot+read noise (counts/s) on a master dark's background level.
    '''
    dark_signal_e = background_rate * exptime * gain   # accumulated signal per raw frame, in electrons
    shot_noise_e = np.sqrt(dark_signal_e)
    frame_noise_e = np.sqrt(shot_noise_e ** 2 + read_noise_e ** 2)
    return (frame_noise_e / gain) / exptime / np.sqrt(n_frames)   # back to counts/s, averaged over n_frames
