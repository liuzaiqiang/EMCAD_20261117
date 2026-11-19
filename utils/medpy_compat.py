"""Bridge the numpy.bool alias expected by the installed MedPy release."""

import numpy as np


if "bool" not in vars(np):
    np.bool = np.bool_
