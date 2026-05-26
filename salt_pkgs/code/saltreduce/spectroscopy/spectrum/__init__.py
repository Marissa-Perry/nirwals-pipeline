# __init__.py
name = "spectrum"
# Specific imports to expose only needed methods publically:
from .saltspectrum import artificial_spectrum
from .saltspectrum import find_object_spectra
from .saltspectrum import extract_spectrum