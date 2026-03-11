# __init__.py
name = "wavelength"
# Specific imports to expose only needed methods publically:
from .saltwavelength import calculate_wavelength_fit
from .saltwavelength import find_cross_correlation
from .saltwavelength import find_line_peaks
from .saltwavelength import fit_zero_point