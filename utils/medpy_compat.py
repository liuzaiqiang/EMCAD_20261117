"""Bridge the numpy.bool alias expected by this project's installed MedPy."""

import numpy as np


if "bool" not in vars(np):
    np.bool = np.bool_
