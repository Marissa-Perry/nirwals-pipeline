import numpy as np
from scipy.ndimage import convolve


OVERFLOW_LIMIT = 1e4   # counts/s; same cutoff experiment_bpm.ipynb uses to exclude corrupted/overflow pixels


def ipc_correction_global(image, alpha_row, alpha_col, bad=None, trusted_source=None):
    '''
    Approximate first-order inverse of the 4-neighbor IPC kernel, applied to every pixel.
    alpha_row (left/right coupling) and alpha_col (up/down coupling) can each be a single
    value or a per-pixel array matching image.shape.
    trusted_source marks bad pixels whose value is still known well enough to leak onto
    their neighbors (their own output value stays unchanged either way).
    '''
    alpha_row = np.broadcast_to(alpha_row, image.shape).astype(float)
    alpha_col = np.broadcast_to(alpha_col, image.shape).astype(float)
    overflow_ok = np.isfinite(image) & (np.abs(image) < OVERFLOW_LIMIT)
    good = overflow_ok.copy()
    if bad is not None:
        good &= ~bad
    source_ok = good.copy()
    if trusted_source is not None:
        source_ok |= (trusted_source & overflow_ok)
    source = np.where(source_ok, image, 0.0)   # an untrusted pixel can't leak a value onto its neighbors
    source_alpha_row = np.where(source_ok, alpha_row, 0.0)   # an untrusted pixel's own coupling strength can't leak either
    source_alpha_col = np.where(source_ok, alpha_col, 0.0)
    neighbor_sum = np.zeros_like(image, dtype=float)
    neighbor_sum[1:, :] += source_alpha_col[:-1, :] * source[:-1, :]    # from the pixel above
    neighbor_sum[:-1, :] += source_alpha_col[1:, :] * source[1:, :]    # from the pixel below
    neighbor_sum[:, 1:] += source_alpha_row[:, :-1] * source[:, :-1]    # from the pixel to the left
    neighbor_sum[:, :-1] += source_alpha_row[:, 1:] * source[:, 1:]    # from the pixel to the right
    corrected = image + 2 * (alpha_row + alpha_col) * image - neighbor_sum
    return np.where(good, corrected, image)   # bad pixels keep their original value


def isolated_hot_pixels(data, threshold, radius=5):
    '''
    Pixels above threshold, and the subset with no other such pixel within radius pixels.
    '''
    hot_pixel = np.isfinite(data) & (data > threshold)
    box = np.ones((2 * radius + 1, 2 * radius + 1))
    nearby = convolve(hot_pixel.astype(float), box, mode='constant', cval=0.0)
    return hot_pixel, hot_pixel & (nearby == 1)


def build_alpha_maps(dark, med, candidates, fallback_alpha_row, fallback_alpha_col, min_measured=200):
    '''
    Per-pixel alpha_row (left/right) and alpha_col (up/down) measured from isolated hot pixels in
    dark. Everywhere else gets this frame's own median measured value, so the default tracks
    whatever this specific dark actually shows -- falling back to fallback_alpha_row/
    fallback_alpha_col only when too few pixels were measured to trust that median. The median is
    taken over every candidate's raw estimate (it's robust to outliers by construction, so
    filtering first would only bias it -- a sanity bound like (0, 0.125) trims one tail harder
    than the other whenever the true value sits close to zero). The (0, 0.125) bound still applies
    separately when deciding whether one specific pixel's own estimate is sane enough to apply
    directly to that pixel, since a single noisy ratio has no such robustness on its own. Also
    returns which pixels got a measured value (so they can still act as leak sources even where
    they're otherwise masked out) and the default row/col values actually used, for logging and
    plotting.
    '''
    clean = np.where(np.abs(dark) < OVERFLOW_LIMIT, dark, np.nan)
    all_row_ests, all_col_ests = [], []
    coords, row_ests, col_ests = [], [], []
    for r, c in zip(*np.where(candidates)):
        if r < 1 or r > dark.shape[0] - 2 or c < 1 or c > dark.shape[1] - 2:
            continue   # skip pixels right at the detector edge
        patch = clean[r-1:r+2, c-1:c+2] - med
        central = patch[1, 1]
        row_neighbors = np.array([patch[1, 0], patch[1, 2]])
        col_neighbors = np.array([patch[0, 1], patch[2, 1]])
        if not np.isfinite(central) or central <= 0 or not np.all(np.isfinite(row_neighbors)) or not np.all(np.isfinite(col_neighbors)):
            continue   # NaN central (overflow-clipped) fails "central <= 0" silently, so check it explicitly
        row_est = row_neighbors.mean() / central
        col_est = col_neighbors.mean() / central
        all_row_ests.append(row_est)
        all_col_ests.append(col_est)
        if 0 < row_est < 0.125 and 0 < col_est < 0.125:
            coords.append((r, c))
            row_ests.append(row_est)
            col_ests.append(col_est)

    if len(all_row_ests) >= min_measured:
        default_alpha_row = np.median(all_row_ests)
        default_alpha_col = np.median(all_col_ests)
    else:
        default_alpha_row = fallback_alpha_row
        default_alpha_col = fallback_alpha_col

    alpha_row_map = np.full(dark.shape, default_alpha_row, dtype=float)
    alpha_col_map = np.full(dark.shape, default_alpha_col, dtype=float)
    measured = np.zeros(dark.shape, dtype=bool)
    for (r, c), row_est, col_est in zip(coords, row_ests, col_ests):
        alpha_row_map[r, c] = row_est
        alpha_col_map[r, c] = col_est
        measured[r, c] = True
    return alpha_row_map, alpha_col_map, measured, default_alpha_row, default_alpha_col


def expected_dark_noise(background_rate, exptime, n_frames, gain, read_noise_e):
    '''
    Expected shot+read noise (counts/s) on a master dark's background level, given its
    exposure time, how many raw frames were mean-combined into it, and the detector's gain
    (e-/ADU) and read noise (e-) at that exposure time.
    '''
    dark_signal_e = background_rate * exptime * gain
    shot_noise_e = np.sqrt(dark_signal_e)
    frame_noise_e = np.sqrt(shot_noise_e ** 2 + read_noise_e ** 2)
    return (frame_noise_e / gain) / exptime / np.sqrt(n_frames)
