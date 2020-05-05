import numpy as np
from .window_func import window_func


def design_kaiser(lo: float, hi: float, attenuation: int, odd: bool) -> np.ndarray:
    """Design a Kaiser window for a low-pass FIR filter.

    Parameters
    ----------
    lo: float
        normalized lower frequency of transition band
    hi: float
        normalized upper frequency of transition band
    attenuation: int
        stop-band attenuation in dB (-20log10(ratio))
    odd: bool
         whether the length shall be odd

    Returns
    -------
    np.ndarray
         designed window

    """
    # determine beta of the kaiser window
    if attenuation < 21:
        beta = 0
    elif attenuation <= 50:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21)
    else:
        beta = 0.1102 * (attenuation - 8.7)

    # determine the number of points
    N = np.round((attenuation - 7.95) / (2 * np.pi * 2.285 * (hi - lo))) + 1
    if odd and (N % 2) == 0:
        N += 1

    # design the window
    return window_func('kaiser', m=N, beta=beta)