"""Internal utils."""
import numpy as np
from scipy.signal import lfilter
from scipy.linalg import toeplitz
from scipy.linalg import lstsq, solve
from mne import pick_channels
from mne.io.eeglab.eeglab import RawEEGLAB


def _mad(X):
    """Median absolute deviation."""
    axis = -1 if X.ndim > 1 else 0
    return np.median(np.abs(X - np.median(X, axis=0)), axis=axis)


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


def _histc(x, nbins):
    """Histogram count (bin-centered). As implemented in histc in Matalb."""
    # bin_edges = np.r_[-np.Inf, 0.5 * (bin_centers[:-1] + bin_centers[1:]),
    #     np.Inf]
    bin_edges = np.r_[np.arange(nbins - 1), np.Inf]
    counts, edges =  np.histogram(x, bin_edges)
    return counts


def _kl_divergence(p, q):
    """KL divergence"""
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


# def _block_covariance(data, window=128, overlap=0.5, padding=True, estimator='cov'):
#     """Compute blockwise covariance."""
#     from pyriemann.utils.covariance import _check_est
#
#     assert 0 <= overlap < 1, "overlap must be < 1"
#     est = _check_est(estimator)
#     X = []
#     n_chans, n_samples = data.shape
#     if padding:  # pad data with zeros
#         pad = np.zeros((n_chans, int(window / 2)))
#         data = np.concatenate((pad, data, pad), axis=1)
#
#     jump = int(window * overlap)
#     ix = 0
#     while (ix + window < n_samples):
#         X.append(est(data[:, ix:ix + window]))
#         ix = ix + jump
#
#     return np.array(X)


def _polystab(a):
    """Polynomial stabilization.

    POLYSTAB(A), where A is a vector of polynomial coefficients,
    stabilizes the polynomial with respect to the unit circle;
    roots whose magnitudes are greater than one are reflected
    inside the unit circle.

    Examples
    --------
    Convert a linear-phase filter into a minimum-phase filter with the same
    magnitude response.
    # >>> h = fir1(25,0.4);               # Window-based FIR filter design
    # >>> flag_linphase = islinphase(h)   # Determines if filter is linear phase
    # >>> hmin = polystab(h) * norm(h)/norm(polystab(h));
    # >>> flag_minphase = isminphase(hmin)# Determines if filter is minimum phase

    """
    v = np.roots(a)
    i = np.where(v != 0)
    vs = 0.5 * (np.sign(np.abs(v[i]) - 1) + 1)
    v[i] = (1 - vs) * v[i] + vs / np.conj(v[i])
    ind = np.where(a != 0)
    b = a[ind[0][0]] * np.poly(v)

    # Return only real coefficients if input was real:
    if not(np.sum(np.imag(a))):
        b = np.real(b)

    return b


def _numf(h, a, nb):
    """Find numerator B given impulse-response h of B/A and denominator A.
    NB is the numerator order.  This function is used by YULEWALK.
    """
    nh = np.max(h.size)
    xn = np.concatenate((1, np.zeros((1, nh - 1))), axis=None)
    impr = lfilter(np.array([1.0]), a, xn)

    b = np.linalg.lstsq(
        toeplitz(impr, np.concatenate((1, np.zeros((1, nb))), axis=None)),
        h.T, rcond=None)[0].T

    return b


def _denf(R, na):
    """Compute denominator from covariances.
    A = DENF(R,NA) computes order NA denominator A from covariances
    R(0)...R(nr) using the Modified Yule-Walker method. This function is used
    by YULEWALK.
    """
    nr = np.max(np.size(R))
    Rm = toeplitz(R[na:nr - 1], R[na:0:-1])
    Rhs = - R[na + 1:nr]
    A = np.concatenate(
        (1, np.linalg.lstsq(Rm, Rhs.T, rcond=None)[0].T), axis=None)
    return A


def _mldivide(A, B):
    """Matrix left-division (A\B).

    Solves the AX = B for X. In other words, X minimizes norm(A*X - B), the
    length of the vector AX - B:
        - linalg.solve(A, B) if A is square
        - linalg.lstsq(A, B) otherwise

    References
    ----------
    .. [1] https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html
    """

    if A.shape[0] == A.shape[1]:
        return solve(A, B)
    else:
        return lstsq(A, B)


def _pick_good_channels(signal: RawEEGLAB) -> list:
    """Pick bad channels from `info` structure and return channels indices."""
    return pick_channels(ch_names=signal.ch_names, include=signal.ch_names, exclude=signal.info["bads"])