# ---------------------------------------------------------------------------- #
"""
SALT general utilities:
- saltkeys provides general utilities for use in the daily primary pipeline,
  data reduction modules and other tools.
"""
# ---------------------------------------------------------------------------- #

MYNAME = 'saltkeys'

# ---------------------------------------------------------------------------- #

# Primary fits extension nr
PRIMARY = 0

# ---------------------------------------------------------------------------- #
def key_values(hdulist, key_list, key_exts=None, key_frmts=None, strip=True,
               include_blank=True):
# ---------------------------------------------------------------------------- #

    # Initialise fits header key value dictionary
    key_value_dict = {}

    # Get fits header key values for keys in the key list
    for key in key_list:

        try:
            # Set fits extension for header key
            if key_exts and key in key_exts:
                ext = key_exts[key]
            else:
                ext = PRIMARY
            # Get header key value
            value = hdulist[ext].header[key]
            # Format header key value (in needed)
            if key_frmts and key in key_frmts:
                value = key_frmts[key].format(value)
            # Strip header key value (in needed)
            if strip:
                value = str(value).strip()

        except:
            value = ''

        if not value and not include_blank:
            continue

        # Add fits header key and value to key value dictionary
        key_value_dict[key] = value

    return key_value_dict

# ---------------------------------------------------------------------------- #
def key_values_id(key_frmts, key_dict):
# ---------------------------------------------------------------------------- #

    # Initialise identifier
    identifier = ''

    # Add formatted key values to identifier
    for key in key_dict:

        # Get key value
        value = key_dict[key]
        # Get key format
        key_frmt = next((item for item in key_frmts if item['key']==key))

        # Format value based on key format type
        if key_frmt['type'] == 'abbreviate':
            value = key_frmt['frmt'][value]

        elif key_frmt['type'] == 'replace':
            value = value.replace(key_frmt['frmt'][0], key_frmt['frmt'][1])

        elif key_frmt['type'] == 'format':
            value = key_frmt['frmt'].format(value)

        elif key_frmt['type'] == 'format_milli':
            value = key_frmt['frmt'].format(int(value) / 1000)

        # Add formatted key value to identifier
        identifier = '{0}{1}'.format(identifier, value)

    return identifier

# ---------------------------------------------------------------------------- #
def key_values_file(prefix, obs_date, tag, key_frmts, key_dict):
# ---------------------------------------------------------------------------- #

    """
    Construct the file name based on key format and value dictionaries.
    """

    # Get key values identifier
    identifier = key_values_id(key_frmts, key_dict)
    # Add file tag to identifier
    identifier = '{0}{1}'.format(tag.lower().capitalize(), identifier)
    # Set file name
    file_name = '{0}{1}{2}.fits'.format(prefix, obs_date, identifier)

    return file_name

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
class SALTError(Exception):
# ---------------------------------------------------------------------------- #

    """Basic exception"""
    pass

# ---------------------------------------------------------------------------- #