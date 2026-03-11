# ---------------------------------------------------------------------------- #
"""
SALT general utilities:
- saltdirs provides directory utilities for use in the daily primary pipeline,
  data reduction modules and other tools.
"""
# ---------------------------------------------------------------------------- #

# Standard library imports
import os
import shutil

# ---------------------------------------------------------------------------- #

MYNAME = 'saltdirs'

# ---------------------------------------------------------------------------- #
def make_directory(directory):
# ---------------------------------------------------------------------------- #

    # Check if directory exists
    exists = os.path.isdir(directory)
    if not exists:
        # NOTE: This procedure may be called from multi-processing, so the
        # initial check may return 'not exists' for, say, process 2, but at
        # this point process 1 may have already made the directory, so...
        try:
            # Make directory
            os.mkdir(directory)

        except FileExistsError:
            pass

    return

# ---------------------------------------------------------------------------- #
def remove_directory(directory):
# ---------------------------------------------------------------------------- #

    # Check if directory exists
    exists = os.path.isdir(directory)
    if exists:
        # Remove directory (and sub directories)
        shutil.rmtree(directory)

    return

# ---------------------------------------------------------------------------- #
def copy_directory(source_dir, target_dir, raise_error=False):
# ---------------------------------------------------------------------------- #

    # Check if source directory exists
    exists = os.path.isdir(source_dir)
    if exists:
        # Check if target directory exists
        exists = os.path.isdir(target_dir)
        if exists:
            # Check raise error
            if raise_error:
                err = 'Target directory {0} already exists.'.format(target_dir)
                raise SALTError(err)
        else:
            # Copy source directory
            shutil.copytree(source_dir, target_dir)

    else:
        # Check raise error
        if raise_error:
            err = 'Source directory {0} does not exist.'.format(source_dir)
            raise SALTError(err)

    return

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
class SALTError(Exception):
# ---------------------------------------------------------------------------- #

    """Basic exception"""
    pass

# ---------------------------------------------------------------------------- #