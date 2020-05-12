import numpy as np
from scipy.signal import freqz
from .utils import _polystab, _denf, _numf


def yulewalk(order: int, F: np.ndarray, M: np.ndarray):
    """Recursive filter design using a least-squares method.

    [B,A] = YULEWALK(N,F,M) finds the N-th order recursive filter
    coefficients B and A such that the filter:

    B(z)   b(1) + b(2)z^-1 + .... + b(n)z^-(n-1)
    ---- = -------------------------------------
    A(z)    1   + a(1)z^-1 + .... + a(n)z^-(n-1)

    matches the magnitude frequency response given by vectors F and M.
    The YULEWALK function performs a least squares fit in the time domain. The
    denominator coefficients {a(1),...,a(NA)} are computed by the so called
    "modified Yule Walker" equations, using NR correlation coefficients
    computed by inverse Fourier transformation of the specified frequency
    response H.
    The numerator is computed by a four step procedure. First, a numerator
    polynomial corresponding to an additive decomposition of the power
    frequency response is computed. Next, the complete frequency response
    corresponding to the numerator and denominator polynomials is evaluated.
    Then a spectral factorization technique is used to obtain the impulse
    response of the filter. Finally, the numerator polynomial is obtained by a
    least squares fit to this impulse response. For a more detailed explanation
    of the algorithm see [1]_.

    Parameters
    ----------
    order : int
        Filter order.
    F : np.ndarray
        Normalised frequency breakpoints for the filter. The frequencies in F
        must be between 0.0 and 1.0, with 1.0 corresponding to half the sample
        rate. They must be in increasing order and start with 0.0 and end with
        1.0.
    M : np.ndarray
        Magnitude breakpoints for the filter such that PLOT(F,M) would show a
        plot of the desired frequency response.

    References
    ----------
    .. [1] B. Friedlander and B. Porat, "The Modified Yule-Walker Method of
           ARMA Spectral Estimation," IEEE Transactions on Aerospace Electronic
           Systems, Vol. AES-20, No. 2, pp. 158-173, March 1984.

    Examples
    --------
    Design an 8th-order lowpass filter and overplot the desired
    frequency response with the actual frequency response:

    # >>> f = [0, .6, .6, 1]         # Frequency breakpoints
    # >>> m = [1, 1, 0, 0]           # Magnitude breakpoints
    # >>> [b, a] = yulewalk(8, f, m) # Filter design using a least-squares method

    """
    F = np.asarray(F)
    M = np.asarray(M)
    npt = 512
    lap = np.fix(npt / 25).astype(int)
    mf = F.size
    npt = npt + 1  # For [dc 1 2 ... nyquist].
    Ht = np.array(np.zeros((1, npt)))
    nint = mf - 1
    df = np.diff(F)

    nb = 0
    Ht[0][0] = M[0]
    for i in range(nint):
        if df[i] == 0:
            nb = nb - int(lap / 2)
            ne = nb + lap
        else:
            ne = int(np.fix(F[i + 1] * npt)) - 1

        j = np.arange(nb, ne + 1)
        if ne == nb:
            inc = 0
        else:
            inc = (j - nb) / (ne - nb)

        Ht[0][nb:ne + 1] = np.array(inc * M[i + 1] + (1 - inc) * M[i])
        nb = ne + 1

    Ht = np.concatenate((Ht, Ht[0][-2:0:-1]), axis=None)
    n = Ht.size
    n2 = np.fix((n + 1) / 2)
    nb = order
    nr = 4 * order
    nt = np.arange(0, nr)

    # compute correlation function of magnitude squared response
    R = np.real(np.fft.ifft(Ht * Ht))
    R = R[0:nr] * (0.54 + 0.46 * np.cos(np.pi * nt / (nr - 1)))   # pick NR correlations  # noqa

    # Form window to be used in extracting the right "wing" of two-sided
    # covariance sequence
    Rwindow = np.concatenate((1 / 2, np.ones((1, int(n2 - 1))), np.zeros((1, int(n - n2)))), axis=None)
    A = _polystab(_denf(R, order))  # compute denominator

    # compute additive decomposition
    Qh = _numf(np.concatenate((R[0] / 2, R[1:nr]), axis=None), A, order)

    # compute impulse response
    _, Ss = 2 * np.real(freqz(Qh, A, worN=n, whole=True))

    hh = np.fft.ifft(np.exp(np.fft.fft(Rwindow * np.fft.ifft(np.log(Ss, dtype=np.complex)))))
    B = np.real(_numf(hh[0:nr], A, nb))

    return B, A