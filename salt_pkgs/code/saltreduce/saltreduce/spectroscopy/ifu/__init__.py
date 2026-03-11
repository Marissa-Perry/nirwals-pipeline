# __init__.py
name = "ifu"
# Specific imports to expose only needed methods publically:
from .saltifu import find_fibres
from .saltifu import set_fibre_traces
from .saltifu import trace_fibres
from .saltifu import extract_fibres