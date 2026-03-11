# ---------------------------------------------------------------------------- #
"""
SALT general utilities:
- saltfiles provides file utilities for use in the daily primary pipeline,
  data reduction modules and other tools.
"""
# ---------------------------------------------------------------------------- #

# Standard library imports
import os
import sys

# json imports (for loading/dumping files)
from json import load
from json import dump
# numpy import
import numpy as np
# astropy import
from astropy.io import fits
# importlib import (for dynamic importing of task modules)
from importlib import import_module

# ---------------------------------------------------------------------------- #

MYNAME = 'saltfiles'

# ---------------------------------------------------------------------------- #

# Primary fits extension nr
PRIMARY = 0

# ---------------------------------------------------------------------------- #
def load_json_file(json_file, default=None, raise_error=False):
# ---------------------------------------------------------------------------- #

    # Initialise json data as default
    json_data = default

    # Check if json file exists
    exists = os.path.isfile(json_file)
    if exists:
        # Load file (JSON input file)
        with open(json_file, 'r') as in_file:
            json_data = load(in_file)

    else:
        # Check raise error
        if raise_error:
            err = 'JSON file {0} does not exist.'.format(json_file)
            raise SALTError(err)

    return json_data

# ---------------------------------------------------------------------------- #
def dump_json_file(json_data, json_file, indent=4):
# ---------------------------------------------------------------------------- #

    # Dump file (JSON output file)
    with open(json_file, 'w') as outfile:
        dump(json_data, outfile, indent=indent)

    return

# ---------------------------------------------------------------------------- #
def load_log_file(log_file, log_name):
# ---------------------------------------------------------------------------- #

    try:
        # Open log file
        with fits.open(log_file, mode='readonly') as hdulist:
            # Get all log data
            log_data = hdulist[log_name].data

    except:
        # Set log data as an empty array
        log_data = np.array([])

    return log_data

# ---------------------------------------------------------------------------- #
def load_prereduction_dicts(obs_date, prereductions, config_dir):
# ---------------------------------------------------------------------------- #

    # Initialise prereduction dictionaries
    prered_keys = {}
    prered_configs = {}

    # Loop for prereductions...
    for prered, prered_dict in prereductions.items():

        # Set prereduction key and prereduction config dictionary
        prered_keys[prered], prered_configs[prered] = load_prereduction_dict(
            obs_date, prered_dict, config_dir)

    return prered_keys, prered_configs

# ---------------------------------------------------------------------------- #
def load_prereduction_dict(obs_date, prered_dict, config_dir):
# ---------------------------------------------------------------------------- #

    # Set prereduction key
    prered_key = prered_dict['key']

    # Check prereduction config file
    if prered_dict['config']['file']:
        # Add config directory path to config file
        config = os.path.join(config_dir, prered_dict['config']['file'])

        # Check if config load procedure is given, i.e., config is a dated
        # list of config files...
        if prered_dict['config']['procedure']:
            # Load config list (JSON input file)
            config_list = load_json_file(config)
            # Set config list entries
            entries = config_list['entries']
            # Format date for retrieval of config entry
            date = ('{0}-{1}-{2}'
                    '').format(obs_date[:4], obs_date[4:6], obs_date[6:8])
            # Get 1st entry with date smaller or equal to observation date
            entry = next(x for x in entries if x['date'] <= date)
            # Add config directory path to config file
            config_file = os.path.join(config_dir, entry['file'])
            # Load module procedure
            module_procedure = load_module_procedure(
                prered_dict['config']['procedure'])
            # Call module procedure
            config_dict = module_procedure(config_file, entry)

            # NOTE: SALT AstroOps meeting 2020-04-23 resolution
            # Update raw fits header with the same gain and read noise
            # values written to product fits header to 'enable' Ken N's
            # POLSALT pipeline... I consider this to be 'non-standard'
            # processing, but what the hell... Let's do it!
            if 'update_raw' in config_list.keys():
                update_raw = bool(config_list['update_raw'])

            else:
                update_raw = False

            # Set update raw indicator in config dictionary
            config_dict['update_raw'] = update_raw

            # NOTE: This 'hack' is part of a solution for prereducing
            # polarimetric data to the 'correct' product format
            if 'spillover' in config_list.keys():
                # Set spillover in config dictionary
                config_dict['spillover'] = config_list['spillover']
            if 'apply_legacy' in config_list.keys():
                # Set apply_legacy in config dictionary
                config_dict['apply_legacy'] = bool(config_list['apply_legacy'])

        else:
            # Load config dictionary (JSON input file) directly
            config_dict = load_json_file(config)

        # Set config directory in config dictionary
        config_dict['config_dir'] = config_dir

    else:
        config_dict = None

    # Set prereduction config dictionary
    prered_config = config_dict

    return prered_key, prered_config

# ---------------------------------------------------------------------------- #
def load_gain_dictionary(gain_file, entry):
# ---------------------------------------------------------------------------- #

    # load rospeed, gainset, gain, rdnoise, bias, amp
    rospeed = np.loadtxt(gain_file, dtype=str, unpack=True, usecols=(0))
    gainset = np.loadtxt(gain_file, dtype=str, unpack=True, usecols=(1))
    gain = np.loadtxt(gain_file, dtype=float, unpack=True, usecols=(2))
    rdnoise = np.loadtxt(gain_file, dtype=float, unpack=True, usecols=(3))
    bias = np.loadtxt(gain_file, dtype=float, unpack=True, usecols=(4))
    amp = np.loadtxt(gain_file, dtype=str, unpack=True, usecols=(5))

    # make dictionary
    gain_dict = {
        "date": entry['date'],
        "ROSPEED": [rs[0] for rs in rospeed],
        "GAINSET": [gs[0] for gs in gainset],
        "GAIN": list(gain),
        "RDNOISE": list(rdnoise),
        "BIAS": list(bias),
        "AMP": [int(a.split('amp')[1]) for a in amp]
    }

    return gain_dict

# ---------------------------------------------------------------------------- #
def load_xtalk_dictionary(xtalk_file, entry):
# ---------------------------------------------------------------------------- #

    # load xcoeff
    xcoeff = np.loadtxt(xtalk_file, dtype=float, unpack=True,
                        usecols=entry['cols'])

    # make dictionary
    xtalk_dict = {
        "date": entry['date'],
        "pair": entry['pair'],
        "xcoeff": list(xcoeff)
    }

    return xtalk_dict

# ---------------------------------------------------------------------------- #
def load_mosaic_dictionary(mosaic_file, entry):
# ---------------------------------------------------------------------------- #

    # Intialise x-shift, y-shift and rotation lists
    xshift = []; yshift = []; rotation = []

    # load gap from mosaic file
    gap = np.loadtxt(mosaic_file, dtype=float, unpack=True, usecols=(1))

    # Loop for ccds...
    for ccd in entry['ccds']:

        # Check if ccd has columns in mosaic file
        if ccd['cols']:
            # load xshift, yshift and rotation for ccd
            xs, ys, rot = np.loadtxt(mosaic_file, dtype=float, unpack=True,
                                     usecols=ccd['cols'])

        else:
            # set xshift, yshift and rotation - all zero
            xs, ys, rot = 0., 0., 0.

        # Add xshift, yshift and rotation to lists
        xshift.append(xs)
        yshift.append(ys)
        rotation.append(rot)

    # make dictionary
    mosaic_dict = {
        "date": entry['date'],
        "ccd": [ccd['nr'] for ccd in entry['ccds']],
        "xshift": xshift,
        "yshift": yshift,
        "rotation": rotation,
        "gap": gap
    }

    return mosaic_dict

# ---------------------------------------------------------------------------- #
def load_dated_config_file(obs_date, config_dir, db_file):
# ---------------------------------------------------------------------------- #

    # Add config directory path to db file name
    db_file = os.path.join(config_dir, db_file)
    # Load config db (JSON input file)
    db = load_json_file(db_file)
    # Filter db entries for <= observation date
    entries = list(filter(lambda e: e['date'] <= obs_date, db['entries']))
    if entries:
        # Get latest file from last entry in list
        config_file = entries[-1]['file']
        # Add config directory path to file name
        config_file = os.path.join(config_dir, config_file)
        # Load config (JSON input file)
        config = load_json_file(config_file)

    return config

# ---------------------------------------------------------------------------- #
def get_dated_config_file(obs_date, config_dir, configs, config):
# ---------------------------------------------------------------------------- #

    # Set dated config file name for current observation date
    config_file = configs[config].format(obs_date)
    # Add config directory path to file name
    config_file = os.path.join(config_dir, config_file)

    # Check if file for current observation date exists
    if os.path.isfile(config_file):
        # Yippeeee! We're outa here!
        pass

    else:
        # Get latest file name from db file:
        # - set config db file name
        db_file = configs['{0}_db'.format(config)]
        # - add config directory path to file name
        db_file = os.path.join(config_dir, db_file)
        # - load config db (JSON input file)
        db = load_json_file(db_file)
        # - filter db entries for <= observation date
        entries = list(filter(lambda e: e['date'] <= obs_date, db['entries']))
        if entries:
            # - get latest file from last entry in list
            config_file = entries[-1]['file']
            # Add config directory path to file name
            config_file = os.path.join(config_dir, config_file)

    return config_file

# ---------------------------------------------------------------------------- #
def load_module_procedure(module_name, procedure_name):
# ---------------------------------------------------------------------------- #

    # Import task module procedure
    module = import_module(module_name)
    module_procedure = getattr(module, procedure_name)

    return module_procedure

# ---------------------------------------------------------------------------- #
def load_module_procedure(procedure_name):
# ---------------------------------------------------------------------------- #

    # Split procedure name to get package module and procedure
    procedure_split = procedure_name.split('.')

    if len(procedure_split) == 4:
        # Get module and procedure names
        module_name = '{0}.{1}.{2}'.format(
            procedure_split[0], procedure_split[1], procedure_split[2])
        procedure = procedure_split[3]

    elif len(procedure_split) == 3:
        # Get module and procedure names
        module_name = '{0}.{1}'.format(
            procedure_split[0], procedure_split[1])
        procedure = procedure_split[2]

    elif len(procedure_split) == 2:
        # Get module and procedure names
        module_name = procedure_split[0]
        procedure = procedure_split[1]

    else:
        # Unknown procedure name
        err = 'Unknown procedure name: {0}.'.format(procedure_name)
        raise SALTError(err)

    # Import task module procedure
    module = import_module(module_name)
    module_procedure = getattr(module, procedure)

    return module_procedure

# ---------------------------------------------------------------------------- #
def link_file(target_file, link_file):
# ---------------------------------------------------------------------------- #

    # Check if symbolic link exists
    exists = os.path.isfile(link_file)
    if exists:
        # Remove symbolic link
        os.remove(link_file)

    # Make symbolic link
    os.symlink(target_file, link_file)

    return

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
class SALTError(Exception):
# ---------------------------------------------------------------------------- #

    """Basic exception"""
    pass

# ---------------------------------------------------------------------------- #