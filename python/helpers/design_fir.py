from typing import Optional
import numpy as np
import scipy
import logging
from .window_func import window_func


def design_fir(N: int, F: np.ndarray, A:np.ndarray, nfft: Optional[int] = None, W: Optional[str] = "hamming"
               ) -> np.ndarray:
    """

    Parameters
    ----------
    N
    F
    A
    nfft
    W

    Returns
    -------


    Notes
    -----
    [1] No `query points` in scipy.interpolate.interp1d()

    """
    assert A.shape == F.shape, "A and F must be of the same length."

    if nfft is None:
        nfft = max(512., 2 ** np.ceil(np.log(N) / np.log(2)))

    try:
        W = window_func(W, m=N+1)
    except Exception as e:
        logging.exception(exc_info=e)
        logging.info("Falling back to `hamming` filter...")
        W = window_func('hamming', m=N + 1)

    # calculate interpolated frequency response
    F = scipy.interpolate.interp1d(np.round(F * nfft), A, kind="cubic")

    # set phase & transform into time domain
    F = F * np.exp(-(0.5 * N) * np.sqrt(-1) * np.pi * np.arange(nfft) / nfft)
    B = np.fft.ifft(F[::-1][1:-1].conj()).real

    # apply window to kernel
    return B[:N+1] * W.ravel()