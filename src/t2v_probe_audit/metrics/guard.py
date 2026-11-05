"""The degenerate-fit guard for the flow channels.

Three rules:

  1. if the median observed displacement is below GUARD_MIN_MAG, the pair
     carries no motion and no global model is fitted;
  2. if the model's induced displacement exceeds DEGENERATE_INFLATION times the
     observed displacement plus 0.5 px, the fit is rejected;
  3. rejected pairs are reported as a zero field; the direction chain bridges
     to the last informative pair.
"""

import numpy as np

from ..config import DEGENERATE_INFLATION, GUARD_MIN_MAG


def has_motion(residual, min_mag: float = GUARD_MIN_MAG):
    return float(np.median(np.sqrt(residual[:, :, 0] ** 2
                                   + residual[:, :, 1] ** 2))) >= min_mag


def guard_residual(flow, residual):
    """Return the residual if the fit is admissible, otherwise zeros."""
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    if float(np.median(mag)) < GUARD_MIN_MAG:
        return np.zeros_like(flow)
    induced = flow - residual
    induced_mean = float(np.mean(np.sqrt(induced[:, :, 0] ** 2
                                         + induced[:, :, 1] ** 2)))
    if induced_mean > DEGENERATE_INFLATION * float(np.mean(mag)) + 0.5:
        return np.zeros_like(flow)
    return residual
