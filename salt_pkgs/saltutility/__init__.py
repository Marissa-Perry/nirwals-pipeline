# __init__.py
name = "saltutility"
# Specific imports to expose only needed methods publically:
from .saltutility import run_subprocess
from .saltutility import process_plugins