"""Small statistical helpers."""

import numpy as np


def remove_outliers(data):
    """Drop NaNs and IQR-based outliers (1.5 * IQR rule) from ``data``."""
    data = [x for x in data if not np.isnan(x)]
    if not data:
        return data
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return [x for x in data if lower <= x <= upper]
