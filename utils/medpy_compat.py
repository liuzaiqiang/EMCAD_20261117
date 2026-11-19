"""Compatibility for MedPy releases that still reference removed NumPy aliases."""

import numpy as np

if "bool" not in vars(np):
    np.bool = np.bool_
