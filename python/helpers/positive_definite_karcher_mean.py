from pymanopt.manifolds.psd import PositiveDefinite
from pymanopt.solvers.nelder_mead import compute_centroid

import numpy as np


def positive_definite_karcher_mean(A: np.ndarray):
    """Compute Karcher means using pymanopt.

    Inspired in https://www.manopt.org/reference/examples/positive_definite_karcher_mean.html#_subfunctions

    Parameters
    ----------
    A: np.ndarray
        of shape (n, n, m)

    Returns
    -------
    np.ndarray
        m Karcher means

    """
    n = A.shape[0]
    assert n == A.shape[1], "The slices of A must be square"

    M = PositiveDefinite(n)
    return compute_centroid(M, A.T)


