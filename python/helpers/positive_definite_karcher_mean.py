from pymanopt.manifolds.psd import PositiveDefinite
from pymanopt.solvers.nelder_mead import compute_centroid


def positive_definite_karcher_mean(A):
    """Compute Karcher means using pymanopt.
       Inspired in https://www.manopt.org/reference/examples/positive_definite_karcher_mean.html#_subfunctions
       A =
    """
    n = A.shape[0]
    assert n == A.shape[1], "The slices of A must be square"

    M = PositiveDefinite(n)
    return compute_centroid(M, A.T)


