# ---------------------------------------------------------------------------- #
"""
SALT utilities:
- saltobslog reads critical header keywords from SALT fits files and collates
  them into a fits table. This is used as the basis for an observation log and
  as a meta-table for pipeline processing.
  Functionality also exists for summarising information on the observed
  proposals and blocks, as well as getting block extracts from the night log.
"""
# ---------------------------------------------------------------------------- #

# Standard library imports
import os
import time

# astropy import
from astropy.io import fits

# Local application imports
# - saltutil.files
from ..files import load_json_file
# - saltutil.logging
from ..logging import logging

# ---------------------------------------------------------------------------- #

MYNAME = 'saltobslog'

# ---------------------------------------------------------------------------- #
def saltobslog(fits_files, obslog_file, obslog_config, log_file='salt.log',
               with_stdout=False):
# ---------------------------------------------------------------------------- #

    # Start log
    with logging(log_file, with_stdout=with_stdout) as log:

        # Create observation log with given config for given fits files
        create_obslog(fits_files, obslog_file, obslog_config, log)

    return

# ---------------------------------------------------------------------------- #
def create_obslog(fits_files, obslog_file, obslog_config, log=None):
# ---------------------------------------------------------------------------- #

    # Write message that log was created
    if log:
        obslog_name = os.path.basename(obslog_file)
        msg = ('{0} -- Create observation log: {1}'
               '\n').format(MYNAME.upper(), obslog_name)
        log.message(msg, with_header=False)

    # Load obslog config file (JSON input file)
    obslog_list = load_json_file(obslog_config)
    # Set obslog columns config
    obslog_cols = obslog_list['log_columns']
    # Set lists needed to create header dictionary and fits table:
    # - keywords
    key_list = [k for k, f, w in obslog_cols]
    # - formats
    frm_list = [f for k, f, w in obslog_cols]
    # - warn flags
    wrn_list = [w for k, f, w in obslog_cols]
    # Set obslog exclusions
    excl_list = obslog_list['log_exclusions']

    # Create header dictionary
    header_dict = create_header_dictionary(fits_files, key_list, frm_list,
                                           wrn_list, excl_list, log)

    # Create fits table
    fits_table = create_fits_table(key_list, frm_list, header_dict, 'OBSLOG')

    # Write fits table to file
    fits_table.writeto(obslog_file, overwrite=True, output_verify='ignore')

    return

# ---------------------------------------------------------------------------- #
def create_header_dictionary(fits_files, key_list, frm_list, wrn_list,
                             excl_list, log, ext='Primary'):
# ---------------------------------------------------------------------------- #

    """
    For a list of fits files and a list of header keys, create a dictionary for
    the header keys and values from the primary (or specified) header extension
    of the files.
    Lists for the format of the keys as well as indicators if a warning must be
    added to the log if the corresponding key is not found in the header (0=no,
    1=yes), are also given as input.
    The list of header keys must start with the key 'FILENAME'.
    Before a file and its header keys are added to the dictionary an exclusion
    check is done based on the given exclusion list containing header keys and
    values to check.
    """

    # Initialise header dictionary
    header_dict = {}
    for key in key_list:

        header_dict[key] = []

    # Sort list of fits files
    fits_files.sort()

    # Write message to log
    if log:
        msg = ' Read fits headers:\n'
        log.message(msg, with_header=False)

    # Check if fits files exist
    if fits_files:

        # Loop for fits files...
        for fits_file in fits_files:

            # Open fits file
            with fits.open(fits_file) as hdulist:

                # Set default excluded flag
                excluded = False
                # Check exclusions
                for excl in excl_list:
                    if hdulist[ext].header[excl['key']] in excl['values']:
                        excluded = True

                # Check if file is not excluded
                if not excluded:
                    # Add file name
                    file_name = os.path.basename(fits_file)
                    header_dict['FILENAME'].append(file_name)

                    # Write message to log
                    if log:
                        msg = ' {0}'.format(file_name)
                        log.message(msg, with_header=False)

                    # Loop for keywords, formats and warnings
                    for k, f, w in zip(key_list[1:], frm_list[1:], wrn_list[1:]):

                        # Ingest keywords from files in the image list
                        d = get_default(f)
                        value = get_value(hdulist[ext], k, d, w, log)
                        header_dict[k].append(value)

    else:
        # Write message to log
        if log:
            log.message(' None', with_header=False)

    # 'Beautify' log
    if log:
        log.message('', with_header=False)

    return header_dict

# ---------------------------------------------------------------------------- #
def get_default(frmt):
# ---------------------------------------------------------------------------- #

    """
    Set the default value for a given format.
    """

    # - string format...
    if frmt.count('A'): 
        default = 'UNKNOWN'

    # - integer
    elif frmt.count('I') or frmt.count('J') or frmt.count('K'):
        default = -999

    # - other
    else: 
        default = -999.99

    return default

# ---------------------------------------------------------------------------- #
def get_value(hdu, key, default, warn, log):
# ---------------------------------------------------------------------------- #

    """
    Get the value for the key. Return default value if retrieved value does not
    pass validation and add warning to log if required.
    """

    try:
        value = hdu.header[key]

        # If value is a string...
        if isinstance(value, str):
            # Remove any spaces
            value = value.strip()

        # If value is 'empty'...
        if str(value).strip() == '':
            # Override value with default
            value = default

        # # If data types of value and default differs...
        # elif type(value) != type(default):
        #     if warn and log:
        #         val_frm = str(type(value))
        #         def_frm = str(type(default))
        #         if log:
        #             msg = ('WARNING: Type mismatch for keyword {0} - '
        #                    'value {1} has type {2} instead of '
        #                    '{3}').format(key, str(value), val_frm, def_frm)
        #             log.message(msg, with_header=False)
        #     # Override value with default
        #     value = default

    except:
        value = default
        if warn and log:
            # Write message to log
            msg = 'WARNING: Cannot find keyword {0}'.format(key)
            log.message(msg, with_header=False)

    return value

# ---------------------------------------------------------------------------- #
def create_fits_table(key_list, frm_list, header_dict, ext_name,
                      process='daily pipeline'):
# ---------------------------------------------------------------------------- #

    """
    Create fits table as log with key and format lists and header dictionary.
    """

    # Initialise column list
    cols = []

    # Create columns based on key and format lists and header dictionary
    for k, f in zip(key_list, frm_list):
        cols.append(fits.Column(name=k, format=f, array=header_dict[k]))

    # Define fits table columns
    columns = fits.ColDefs(cols)
    # Create fits table from columns

    fits_table = fits.BinTableHDU.from_columns(columns)

    # Add additional header keywords
    fits_table.header['EXTNAME'] = (ext_name,
                                    'Name of this binary table extension')
    fits_table.header['OBSERVAT'] = ('SALT',
                                     'Southern African Large Telescope')
    fits_table.header['SAL-TLM'] = (time.asctime(time.localtime()),
                                    'File last updated by {0}'.format(process))

    return fits_table

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
class SALTError(Exception):
# ---------------------------------------------------------------------------- #

    """Basic exception"""
    pass

# ---------------------------------------------------------------------------- #