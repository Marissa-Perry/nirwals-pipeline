# __init__.py
name = "functions"
# Specific imports to expose only needed methods publically:
# - functions
from .saltfunctions import air_to_vac
from .saltfunctions import normal_distribution
from .saltfunctions import get_evenly_spaced_array
# - fitting
from .saltfitting import Fit1D
from .saltfitting import CurFit
from .saltfitting import InterFit
# - detector corrections
from .ipc_correction import ipc_correction_global
from .ipc_correction import isolated_hot_pixels
from .ipc_correction import build_alpha_maps
from .ipc_correction import expected_dark_noise