# ---------------------------------------------------------------------------- #
"""
SALT general utilities:
- saltutil provides subprocess utilities for use in the daily primary pipeline,
  data reduction modules and other tools.
"""
# ---------------------------------------------------------------------------- #

# Standard library imports
import subprocess

# ---------------------------------------------------------------------------- #

MYNAME = 'saltutil'

# ---------------------------------------------------------------------------- #
def run_subprocess(subproc, ignore_return_code=[], raise_error=True):
# ---------------------------------------------------------------------------- #

    # Initialise result
    result = {'error': '', 'return_code': 0}

    try:

        # Run the command in the line entry
        subprocess.run(subproc, shell=True, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, universal_newlines=True)

    except subprocess.CalledProcessError as error:

        # Check if sub process exited with a return code to ignore
        if error.returncode in ignore_return_code:
            # Set return code
            result['return_code'] = error.returncode

        else:
            # Check stderr
            if error.stderr:
                # Check SALTError
                if 'SALTError' in error.stderr:
                    # Extract SALTError message
                    err_msg = error.stderr.split('SALTError:')[1].strip()
                    # Override error with SALTError
                    error = SALTError(err_msg)

            # Set error
            result['error'] = str(error)
            # Raise error (if needed)
            if raise_error:
                raise error

    except Exception as error:

        raise error

    return result

# ---------------------------------------------------------------------------- #
def process_plugins(obs_date, plugins_file, log, ignore_return_code=[]):
# ---------------------------------------------------------------------------- #

    # Initialise plugins list
    plugins = []

    # Open plugins file
    with open(plugins_file, 'r') as p:

        # Write message to log
        msg = '{0} -- Process plugin(s):\n'.format(MYNAME.upper())
        log.message(msg, with_header=False)

        # Get all the line entries
        plugins_entries = p.readlines()

    # Check if entries exist
    if plugins_entries:

        # Loop through the entries
        for plugins_entry in plugins_entries:

            # 'Clean' the entry
            plugins_entry = plugins_entry.strip()

            # Check that the entry is not a comment
            if plugins_entry and not plugins_entry.startswith('#'):
                # Initialise plugin dictionary
                plugin_dict = {}
                # Replace observation date placeholder
                plugin = plugins_entry.replace('CCYYMMDD', obs_date)
                # Write message to log
                msg = ' {0}\n'.format(plugin)
                log.message(msg, with_header=False)
                # Run subprocess
                result = run_subprocess(plugin, ignore_return_code, False)
                # Set plugin dictionary
                plugin_dict['plugin'] = plugin
                plugin_dict['result'] = result['error']
                # Add to plugins list
                plugins.append(plugin_dict)

    if plugins:
        pass

    else:
        # Add to plugins list
        plugins.append({'plugin': 'No plugins', 'result': None})
        # Write message to log
        msg = ' No plugins\n'
        log.message(msg, with_header=False)

    return plugins

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
class SALTError(Exception):
# ---------------------------------------------------------------------------- #

    """Basic exception"""
    pass

# ---------------------------------------------------------------------------- #