# __init__.py
name = "logging"
# Specific imports to expose only needed methods publically:
from .saltlogging import logging
from .saltlogging import call_info