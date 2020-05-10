"""Internal utils."""
import numpy as np


def _mad(X):
    """Median absolute deviation."""
    return np.median(np.abs(X - np.median(X, axis=0)), axis=0)


def _sliding_window(array, window, steps=1, axis=1):
    """Efficient sliding window."""
    # Sub-array shapes
    shape = list(array.shape)
    shape[axis] = np.ceil(array.shape[axis] / steps - window / steps + 1).astype(int)
    shape.append(window)

    # Strides (in bytes)
    strides = list(array.strides)
    strides[axis] *= steps
    strides.append(array.strides[axis])

    # Window samples
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)