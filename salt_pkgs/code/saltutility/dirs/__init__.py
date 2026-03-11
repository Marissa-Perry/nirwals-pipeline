# __init__.py
name = "dirs"
# Specific imports to expose only needed methods publically:
from .saltdirs import make_directory
from .saltdirs import copy_directory
from .saltdirs import remove_directory