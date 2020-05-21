import numpy as np
from .geometric_median import geometric_median


def block_geometric_median(X: np.ndarray, blocksize: int, tol: float = 1e-5, max_iter: int = 500) -> np.ndarray:
    """Calculate a blockwise geometric median.

    This is faster and less memory-intensive than the regular geom_median
    function. This statistic is not robust to artifacts that persist over a
    duration that is significantly shorter than the blocksize.

    Parameters
    ----------
    X : array,
        The data of shape=(observations, variables)
    blocksize : int
        The number of successive samples over which a regular mean should be taken.
    tol : float (default: 1e-5)
        Tolerance.
    max_iter : int (default: 500)
        Max number of iterations.

    Returns
    -------
    g : pd.ndarray,
        Geometric median over X.

    Notes
    -----
    This function is noticeably faster if the length of the data is divisible
    by the block size. Uses the GPU if available.

    """
    if blocksize > 1:
        o, v = X.shape
        r = np.mod(o, blocksize)
        b = int((o - r) / blocksize)
        X_replace = np.zeros((b + 1, v))
        if r > 0:
            X_replace[0:b, :] = np.reshape(np.sum(np.reshape(X[0:(o - r), :], (blocksize, b * v)), axis=0), (b, v))
            X_replace[b, :] = np.sum(X[(o - r + 1):o, :], axis=0) * (blocksize / r)
        else:
            X_replace = np.reshape(np.sum(np.reshape(X, (blocksize, b * v)), axis=0), (b, v))
        X = X_replace

    y = np.median(X, axis=0)
    y = geometric_median(X=X, y=y, tol=tol, max_iter=max_iter) / blocksize

    return y