import numpy as np
from numpy.matlib import repmat


def geometric_median(X: np.ndarray, y: float, tol: float = 1e-5, max_iter: int = 500) -> np.ndarray:
    """Calculate the geometric median for a set of observations.

    This is using Weiszfeld's algorithm (mean under a Laplacian noise
    distribution)

    Parameters
    ----------
    X : np.ndarray
        The data, as in mean
    y : float
        Initial value. The median of X.
    tol : float
        Tolerance.
    max_iter: int (default: 500)
        Max number of iterations.

    Returns
    -------
    np.ndarray
        geometric median over X

    """
    for i in range(max_iter):
        invnorms = 1 / np.sqrt(np.sum((X - repmat(y, X.shape[0], 1))**2, axis=1))
        oldy = y
        y = np.sum(X * np.transpose(repmat(invnorms, X.shape[1], 1)), axis=0) / np.sum(invnorms)

        if (np.linalg.norm(y - oldy) / np.linalg.norm(y)) < tol:
            break
    return y