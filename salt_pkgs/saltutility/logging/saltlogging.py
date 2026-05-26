# ---------------------------------------------------------------------------- #
"""
SALT utilities:
- saltlogging provides uniform process logging functionalities for use in the
  daily data pipeline, data reduction modules and other tools.
"""
# ---------------------------------------------------------------------------- #

# Standard library imports
import os
import inspect
from time import strftime
from traceback import print_exc
from contextlib import contextmanager

# ---------------------------------------------------------------------------- #

MYNAME = 'saltlogging'

# ---------------------------------------------------------------------------- #

# Defaults
EXCLUDE = ['workflows', 'stats']

ERROR = 'ERROR ------------------------------------------------------'
WARNING = 'WARNING ----------------------------------------------------'
MESSAGE = 'MESSAGE ----------------------------------------------------'

# ---------------------------------------------------------------------------- #
def call_info(level=3, wrap=True, wrapchar=80, exclude=EXCLUDE):
# ---------------------------------------------------------------------------- #
    """
    Return the call information. This includes the name of the current program
    as well as all the parameters (i.e., arguments) passed to it.
    The return is the name of the program along with a string listing all its
    parameters.
    Function parameters:
        level -- the level of the frame of interest
        wrap  -- wraps characters if true
        wrapchar -- number of characters to wrap at
        exclude -- options (parameters) to exclude 
    Returns pname (str), pinfo (str) 
    """

    # Get program frame
    frame = inspect.getouterframes(inspect.currentframe())[level][0]
    # Get program name
    pname = str(inspect.getframeinfo(frame)[2])
    # Get program arguments
    args, _, _, values = inspect.getargvalues(frame)
    # Add program name to info
    pinfo = '{0}'.format(pname.upper())
    # initialise character count
    count = 0

    # Loop for program arguments...
    for arg in args:

        # If argument is not excluded
        if arg not in exclude: 
            # Compose argument string
            argstr = ' {0}={1}'.format(arg, values[arg])
            # Add length of argument string to character count
            count += len(argstr)

            # If wrapping is required and character count exceeds limit...
            if wrap and count > wrapchar:
                # Add line feed to info
                pinfo += '\n'
                # Set character count to length of current argument string
                count = len(argstr)

            # If argument contains substring 'pass' (i.e., it's a password)...
            if arg.count('pass'):
                # Add current argument and '****' as value to info
                pinfo += ' {0}={1}'.format(arg, '****')

            else:
                # Add current argument string to info
                pinfo += ' {0}={1}'.format(arg, values[arg])

    return pname, pinfo

# ---------------------------------------------------------------------------- #
@contextmanager
# ---------------------------------------------------------------------------- #
def logging(logfile, only_stdout=False, with_stdout=True, with_call=True):
# ---------------------------------------------------------------------------- #
    """Context manager to ensure proper error handling and logging.

    Example usage:
        
    with logging('logfile.txt') as log:
        # User code
        log.message('Hello world!') # Writes message to log
        # Some more user code
        log.warning('This is a warning message') # Writes warning to log
        # Again some user code
        raise SALTError('Oops!') # Writes error message and traceback to log
    """

    # Create LogFile object
    log = SALTLog(logfile, only_stdout=only_stdout, with_stdout=with_stdout)

    # Log call
    if with_call:
        # Get program name and info
        pname, pinfo = call_info()       
        # Log program info (includes program name)
        log.message('{0}\n'.format(pinfo), with_header=False)
        log.message('{0} starting\n'.format(pname))

    # Wrap code block in try, except to ensure proper error handling
    try:
        # Transfer control to wrapped code with access to the log
        yield log
        # Log completion
        if with_call:
            msg = '{0} completed\n'.format(pname)
            log.message(msg)

    except Exception as err:
        # Catch and log any errors that may have occured
        log.error(err)
        # Log abort
        if with_call:
            msg = '{0} aborted\n'.format(pname)
            log.message(msg)

        # Raise the error to quit out of the program and allow any wrapper
        # program to catch the error
        raise err

    finally:
        # Any additional cleanup code and logging goes here
        pass

# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
class SALTLog:
# ---------------------------------------------------------------------------- #
    """Class providing uniform logging."""

    def __init__(self, logfile, only_stdout=False, with_stdout=True,
                 with_traceback=True):
        """Constructor, binds logfile to file object *f*."""

        # Bind to logfile
        self.logfile = logfile
        self.logdir = os.getcwd()
        self.only_stdout = only_stdout
        self.with_stdout = with_stdout
        self.with_traceback = with_traceback

    def error(self, e):
        """Prints error message *e* and traceback (optional) to logfile."""

        curdir = os.getcwd()
        os.chdir(self.logdir)

        # Get current time
        time = strftime('%Y-%m-%d %H:%M:%S')
        # Define header
        header = '{0} {1}\n'.format(time, ERROR)
        # Compose final error message
        log_message = '{0}{1}\n'.format(header, e)

        if not self.only_stdout:

            with open(self.logfile, 'a') as f:

                # Write header + error message to logfile
                f.write('{0}\n'.format(log_message))
                # Optional error traceback
                if self.with_traceback:
                    # Write error traceback to logfile
                    print_exc(file=f)
                    f.write('\n')

        # Optional print to standard output
        if self.with_stdout:
            # Print header + error message
            print('{0}'.format(log_message))
            # Optional error traceback
            if self.with_traceback:
                # Print error traceback
                print_exc()
                print('\n')

        os.chdir(curdir)

    def warning(self, w):
        """Prints warning message *w* to logfile."""

        curdir = os.getcwd()
        os.chdir(self.logdir)

        # Get current time
        time = strftime('%Y-%m-%d %H:%M:%S')
        # Define header
        header = '{0} {1}\n'.format(time, WARNING)
        # Compose final warning message
        log_message = '{0}{1}\n'.format(header, w)

        if not self.only_stdout:

            with open(self.logfile, 'a') as f:

                # Write header + warning message to logfile
                f.write('{0}\n'.format(log_message))

        # Optional print to standard output
        if self.with_stdout:
            print('{0}'.format(log_message))

        os.chdir(curdir)

    def message(self, m, with_header=True):
        """Prints message *m* to logfile."""

        curdir = os.getcwd()
        os.chdir(self.logdir)

        # Get current time
        time = strftime('%Y-%m-%d %H:%M:%S')
        # Define header
        header = '{0} {1}\n'.format(time, MESSAGE)
        # Compose final message
        if with_header:
            log_message = '{0}{1}'.format(header, m)

        else:
            log_message = '{0}'.format(m)

        if not self.only_stdout:

            with open(self.logfile, 'a') as f:

                # Write header + message to logfile
                f.write('{0}\n'.format(log_message))

        # Optional print to standard output
        if self.with_stdout:
            print('{0}'.format(log_message))

        os.chdir(curdir)

# ---------------------------------------------------------------------------- #