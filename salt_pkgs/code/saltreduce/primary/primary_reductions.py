"""
Primary reductions are run for all instruments by the SALT DRP daily (Kotez+2025). 
While the NIRWALS DRP has been under development, NIRWALS data is not run through the SALT DRP. 
This is a temporary primary-reduction script for NIRWALS data, to be executed before the NIRWALS DRP. Once the NIRWALS DRP is re-integrated into the SALT DRP, this is no longer necessary.

primary reduction steps:
  1. build an observation-log FITS table and move it to the product directory
  2. combine darks -> master dark(s) and move them to the product directory
  3. combine flats -> master flat(s) and move them to the product directory
  4. dark-subtract frames and move these to the product directory with a product prefix "dph"
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u


# --------------------------------------------------------------------------- #
# configs
# --------------------------------------------------------------------------- #

# FITS extension handles used throughout (match your existing scripts).
PRIMARY = 'PRIMARY'
SCI = 'SCI'
EXPTYPE = 'EXPTYPE'
OBJECT = 'OBJECT'

# Darks and flats grouped by EXPTIME
DARK_GROUP = ('EXPTIME', )
FLAT_GROUP = ('EXPTIME', )

# --------------------------------------------------------------------------- #
# helper functions
# --------------------------------------------------------------------------- #

def is_dark(exptype):
    return 'dark' in str(exptype).lower()

def is_flat(exptype):
    return 'flat' in str(exptype).lower()

def is_arc(exptype, obj):
    return ('arc' in str(exptype).lower()) or (str(obj).strip().upper() == 'ARC')

def is_sky(exptype):
    return 'sky' in str(exptype).lower()

def product_dir(obs_date):
    """
    Product directory for a given observation date.
    """
    return os.path.join('nirwals_pipeline', obs_date, 'nirwals', 'product')

def raw_dir(obs_date):
    """
    Raw directory for a given observation date.
    """
    return os.path.join('nirwals_pipeline', obs_date, 'nirwals', 'raw')

def read_header_key(hdulist, key, default=None):
    """
    Read key from the PRIMARY (or SCI) header
    """
    for ext in (PRIMARY, SCI):
        try:
            if key in hdulist[ext].header:
                return hdulist[ext].header[key]
        except (KeyError, IndexError):
            pass
    for hdu in hdulist:
        if hdu.header is not None and key in hdu.header:
            return hdu.header[key]
    return default

def exposure_number(filename):
    """
    Pull the 4-digit exposure number from a frame name, with or without a
    prefix: 'N202407250012...' or 'dphN202407250012...' -> '0012'.
    """
    core = os.path.basename(filename).split('.')[0]     # '[dph]N202407250012'
    digits = ''.join(c for c in core if c.isdigit())    # '202407250012' (date + exp)
    return digits[-4:] if len(digits) >= 4 else ''      # last 4 = exposure number

    
def master_dark_name(obs_date, filename):
    return f'N{obs_date}Dark{exposure_number(filename)}.fits'


def master_flat_name(obs_date, filename):
    return f'N{obs_date}Flat{exposure_number(filename)}.fits'


def duplicate(path):
    """
    Reject duplicate copies like '*Dark0013 2.fits' or '*copy.fits'.
    """
    b = os.path.basename(path)
    return (' ' in b) or ('copy' in b.lower())


def group_frames(files, predicate, group_keys):
    groups = {}
    for f in files:
        with fits.open(f) as hdu:
            exptype = read_header_key(hdu, EXPTYPE)
            if not predicate(exptype):
                continue
            key = tuple(read_header_key(hdu, k) for k in group_keys)
            groups.setdefault(key, []).append(f)
    return groups


def combine_group(group_files, out_path):
    """
    Mean-combine the SCI extension across group_files, inheriting the header of the first file. 

    Note: these master files are not dark subtracted
    """
    stack = [np.asarray(fits.getdata(gf, SCI), dtype=float) for gf in group_files]
    avg = np.mean(np.stack(stack, axis=0), axis=0)
    with fits.open(group_files[0]) as hdu:
        hdu[SCI].data = avg
        hdu.writeto(out_path, overwrite=True)
    return avg

def flat_group_key(filename):
    """
    Return the exposure-number block used to group flat exposures.

    For example:
        N20240701001.1.1.reduced.fits -> '001.1'
        N20240701001.1.2.reduced.fits -> '001.1'
        N20240701002.2.1.reduced.fits -> '002.2'
    """
    name = os.path.basename(filename)
    core = name.split('.reduced.fits')[0]

    # Remove the leading N + YYYYMMDD
    remainder = core[1 + 8:]

    # First two components define the flat exposure block
    parts = remainder.split('.')

    if len(parts) < 2:
        raise ValueError(f'Could not determine flat exposure block from {filename}')

    return f'{parts[0]}.{parts[1]}'


# --------------------------------------------------------------------------- #
# observation log
# --------------------------------------------------------------------------- #

def generate_obs_log_fits(obs_date, prefix='dph', no_sky_exp=False):
    """
    Build N{date}OBSLOG.fits automatically from the raw frames
    Each {prefix}N{date}*reduced.fits frame is typed from its headers EXPTYPE and OBJECT.
    """
    rawdir = raw_dir(obs_date)
    files = sorted(glob.glob(os.path.join(rawdir, f'N{obs_date}*reduced.fits')))

    rows = []            # (OBSTYPE, FILENAME) decided immediately from headers
    object_frames = []   # object frames with no explicit sky flag -- may need pointing
    exptypes_seen = set()
    n_expsky = 0         # frames flagged Sky by EXPTYPE

    for path in files:
        if duplicate(path):
            continue
        name = os.path.basename(path)
        store = name[len(prefix):] if name.startswith(prefix) else name  # pipeline re-adds prefix
        with fits.open(path) as hdu:
            exptype = read_header_key(hdu, EXPTYPE)
            obj = read_header_key(hdu, OBJECT, default='')
            exptypes_seen.add(str(exptype))
            if is_dark(exptype):
                rows.append(('Dark', store)); continue
            if is_flat(exptype):
                rows.append(('Flat', store)); continue
            if is_arc(exptype, obj):
                rows.append(('Arc', store)); continue
            if is_sky(exptype):
                rows.append(('Sky', store)); n_expsky += 1; continue
            object_frames.append({'store': store, 
                                  'obj': str(obj),
                                  'exp': exposure_number(store)})

    # if there are no Sky exposures and user did not pass no_sky_exp flag, raise an error
    if (n_expsky == 0) and not no_sky_exp:
        raise RuntimeError(f'No "Sky" EXPTYPE found for files in {obs_date}. If expecting sky frames, rename data using EXPTYPE in PRIMARY header. If not, set --no_sky_exp flag.')
    else:
        for fr in object_frames:
            rows.append(('Science', fr['store']))

    df = pd.DataFrame(rows, columns=['OBSTYPE', 'FILENAME'])
    print(f"   - {len(df)} files in OBSLOG fits  {df['OBSTYPE'].value_counts().to_dict()}")

    table = Table.from_pandas(df)
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(table, name='OBSLOG')])
    prddir = product_dir(obs_date)
    output_filename = f'N{obs_date}OBSLOG.fits'
    hdul.writeto(os.path.join(prddir, output_filename), overwrite=True)
    print(f'   - observation log written to {output_filename}')
    return table

# --------------------------------------------------------------------------- #
# combine darks
# --------------------------------------------------------------------------- #

def combine_darks(obs_date):
    rawdir = raw_dir(obs_date)
    files = sorted(glob.glob(os.path.join(rawdir, f'N{obs_date}*reduced.fits')))
    groups = group_frames(files, is_dark, DARK_GROUP)
    if not groups:
        print('   - no dark frames found')
        return
    for key, gfiles in sorted(groups.items(), key=lambda kv: str(kv[0])):
        exptime = key[0]
        prddir = product_dir(obs_date)
        out = os.path.join(prddir, master_dark_name(obs_date, gfiles[0]))  # use first file in group for name
        combine_group(gfiles, out)
        print(f'   - EXPTIME={exptime}: {len(gfiles)} frame(s) -> {os.path.basename(out)}')


# --------------------------------------------------------------------------- #
# combine flats
# --------------------------------------------------------------------------- #

def combine_flats(obs_date):
    rawdir = raw_dir(obs_date)
    files = sorted(glob.glob(os.path.join(rawdir, f'N{obs_date}*reduced.fits')))
    flat_files = []
    for f in files:
        with fits.open(f) as hdu:
            exptype = read_header_key(hdu, EXPTYPE)
        if is_flat(exptype):
            flat_files.append(f)
    if not flat_files:
        print('   - no flat frames found')
        return
    groups = {}
    for f in flat_files:
        key = flat_group_key(f)
        groups.setdefault(key, []).append(f)
    prddir = product_dir(obs_date)
    for key, gfiles in sorted(groups.items()):
        out = os.path.join(prddir, master_flat_name(obs_date, gfiles[0]))
        combine_group(gfiles, out)
        print(f'   - exposure block={key}: {len(gfiles)} frame(s) -> {os.path.basename(out)}')


# --------------------------------------------------------------------------- #
#  dark subtraction
# --------------------------------------------------------------------------- #

def set_dark_file(exp_time, prd_dir, obs_date, subtract=True, tol=3.0):
    """
    Find master dark with the same EXPTIME as the current frame.
    """
    if not subtract:
        return None
    wildcard = os.path.join(prd_dir, f'N{obs_date}Dark*.fits')
    dark_files = [f for f in sorted(glob.glob(wildcard)) if not duplicate(f)]
    if not dark_files:
        raise FileNotFoundError(f'No master dark files found with wildcard: {wildcard}')

    best, best_dt = None, None
    for f in dark_files:
        with fits.open(f) as dh:
            d_exptime = read_header_key(dh, 'EXPTIME')
        if d_exptime is None:
            continue
        dt = abs(float(d_exptime) - float(exp_time))
        if best_dt is None or dt < best_dt:
            best, best_dt = f, dt

    if best is None or best_dt > tol:
        raise ValueError(f'No master dark within {tol}s of EXPTIME={exp_time} '
                         f'(nearest off by {best_dt}) using wildcard {wildcard}')
    return best


def _dark_subtract(in_path, out_path, obs_date, pdir, inplace):
    """
    Subtract the EXPTIME-matched master dark from raw SCI.
    Returns 1 if a subtraction was written, 0 if skipped.
    """
    mode = 'update' if inplace else 'readonly'
    with fits.open(in_path, mode=mode) as hdu:
        if bool(read_header_key(hdu, 'DARKSUB', default=False)):
            return 0  # already dark-subtracted
        exp_time = read_header_key(hdu, 'EXPTIME')
        try:
            dark_file = set_dark_file(exp_time, pdir, obs_date, subtract=True)
        except (FileNotFoundError, ValueError) as e:
            print(f'   - WARNING: {os.path.basename(in_path)}: {e}')
            return 0

        with fits.open(dark_file) as dh:
            dark = np.asarray(dh[SCI].data, dtype=float)
        dark = np.where(np.isfinite(dark), dark, 0.0)  # zero non-finite dark pixels

        hdu[SCI].data = np.asarray(hdu[SCI].data, dtype=float) - dark
        hdu[PRIMARY].header['DARKSUB'] = (True, 'Master dark subtracted')
        hdu[PRIMARY].header['DARKFILE'] = (os.path.basename(dark_file), 'Master dark used')

        if inplace:
            hdu.flush()
        else:
            hdu.writeto(out_path, overwrite=True)
    return 1


def dark_subtract_all(obs_date, prefix):
    """
    Dark-subtract frames.
    Raw N{obs_date}...reduced.fits are never modified. New files are created with the prefix "dph".
    A DARKSUB primary-header keyword is written.
    """
    rawdir = raw_dir(obs_date)
    n_done = n_skip = 0

    # science / sky / arc: raw -> dark-subtracted, prefixed copy (raw kept)
    for f in sorted(glob.glob(os.path.join(rawdir, f'N{obs_date}*reduced.fits'))):
        if duplicate(f):
            continue
        with fits.open(f) as hdu:
            exptype = read_header_key(hdu, EXPTYPE)
        if is_dark(exptype) or is_flat(exptype):
            continue  # darks aren't subtracted; flats are handled via the master flat
        prddir = product_dir(obs_date)
        out = os.path.join(prddir, prefix + os.path.basename(f))
        ok = _dark_subtract(f, out, obs_date, prddir, inplace=False)
        n_done += ok
        n_skip += (1 - ok)

    print(f'   - dark-subtracted {n_done} frame(s), skipped {n_skip}')


# --------------------------------------------------------------------------- #
# Running primary reductions
# --------------------------------------------------------------------------- #

def run(obs_date, prefix='dph', skip=(), no_sky_exp=False):

    if 'obslog' not in skip:
        print('\nobservation log\n')
        try:
            generate_obs_log_fits(obs_date, prefix=prefix, no_sky_exp=no_sky_exp)
        except Exception as e:
            raise RuntimeError(f'ERROR building obs log: \n{e}')

    if 'darks' not in skip:
        print('\ncombine darks\n')
        combine_darks(obs_date)

    if 'flats' not in skip:
        print('\ncombine flats\n')
        combine_flats(obs_date)

    if 'darksub' not in skip:
        print('\ndark subtraction\n')
        dark_subtract_all(obs_date, prefix)


def main():
    p = argparse.ArgumentParser(description='primary-reductions')
    p.add_argument('obs_date', help='observation date, e.g. 20240701')
    p.add_argument('--no_sky_exp', action='store_true', help='reduction continues if no sky exposures are found')
    p.add_argument('--skip', nargs='*', default=[],
                       choices=['obslog', 'darks', 'flats', 'darksub'],
                       help='steps to skip')
    args = p.parse_args()

    run(obs_date=args.obs_date,
        skip=args.skip,
        no_sky_exp=args.no_sky_exp)


if __name__ == '__main__':
    main()
