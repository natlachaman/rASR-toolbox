from typing import Optional
import numpy as np
import scipy
from .window_func import window_func


def design_fir(N: int, F: np.ndarray, A: np.ndarray, nfft: Optional[int] = None, W: Optional[np.ndarray] = None
               ) -> np.ndarray:
    """Design an FIR filter using the frequency-sampling method.

    The frequency response is interpolated cubically between the specified frequency points.

    Parameters
    ----------
    N: int
        order of the filter
    F: np.ndarray
        vector of frequencies at which amplitudes shall be defined (starts with 0 and goes up to 1; try to avoid too
        sharp transitions)
    A: np.ndarray
        vector of amplitudes, one value per specified frequency
    nfft: Optional[int] (Default: None)
        optionally number of FFT bins to use
    W: Optional[np.ndarray] (Default: None)
        optionally the window function to use (default: Hamming)

    Returns
    -------
    np.ndarray
        designed filter kernel

    Notes
    -----
    [1] No `query points` in scipy.interpolate.interp1d()

    """
    assert A.shape == F.shape, "A and F must be of the same length."

    if nfft is None:
        nfft = max(512., 2 ** np.ceil(np.log(N) / np.log(2)))

    if W is None:
        W = window_func('hamming', m=N + 1)

    # calculate interpolated frequency response
    F = scipy.interpolate.interp1d(np.round(F * nfft), A, kind="cubic")

    # set phase & transform into time domain
    F = F * np.exp(-(0.5 * N) * np.sqrt(-1) * np.pi * np.arange(nfft) / nfft)
    B = np.fft.ifft(F[::-1][1:-1].conj()).real

    # apply window to kernel
    return B[:N+1] * W.ravel()