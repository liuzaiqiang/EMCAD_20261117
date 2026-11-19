"""Keep installed MedPy's NumPy 1.x boolean alias working with NumPy 1.24+."""

import numpy as np


# MedPy's binary metrics call ``numpy.bool`` internally. This environment has
# a newer NumPy that removed that spelling; the scalar type is unchanged.
if "bool" not in vars(np):
    np.bool = np.bool_
