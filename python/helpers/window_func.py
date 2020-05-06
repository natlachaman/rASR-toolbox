from typing import Union
import numpy as np
import scipy


def window_func(name: str, m: int, **kwargs: Union[float, int]) -> np.ndarray:
    """Design a window for a given window function.

    Parameters
    ----------
    name: str
        name of the window, can be any of the following:
              'bartlett' : Bartlett window
              'barthann' : Bartlett-Hann window
              'blackman' : Blackman window
              'blackmanharris' : Blackman-Harris window
              'flattop'  : Flat-top window
              'gauss'    : Gaussian window with parameter alpha (default: 2.5)
              'hamming'  : Hamming window
              'hann'     : Hann window
              'kaiser'   : Kaiser window with parameter beta (default: 0.5)
              'lanczos'  : Lanczos window
              'nuttall'  : Blackman-Nuttall window
              'rect'     : Rectangular window
              'triang'   : Triangular window
    m: int
        number of points in the window
    kwargs: Union[float, int]
        window parameter(s) (if any)

    Returns
    -------
    np.ndarray
        designed window (column vector)

    """

    p = np.arange(m - 1) / (m - 1)

    if name == 'bartlett':
        w = 1 - np.abs((np.arange(m - 1) - (m - 1) / 2) / ((m - 1) / 2))

    elif name in ['barthann', 'barthannwin']:
        w = 0.62 - 0.48 * np.abs(p - 0.5) - 0.38 * np.cos(2 * np.pi * p)

    elif name == 'blackman':
        w = 0.42 - 0.5 * np.cos(2 * np.pi * p) + 0.08 * np.cos(4 * np.pi * p)

    elif name == 'blackmanharris':
        w = 0.35875 - 0.48829 * np.cos(2 * np.pi * p) + 0.14128 * np.cos(4 * np.pi * p) \
            - 0.01168 * np.cos(6 * np.pi * p)

    elif name in ['bohman', 'bohmanwin']:
        w = (1 - np.abs(p * 2 - 1)) * np.cos(np.pi * np.abs(p * 2 - 1)) + (1 / np.pi) \
            * np.sin(np.pi * np.abs(p * 2 - 1))

    elif name in ['flattop', 'flattopwin']:
        w = 0.2157 - 0.4163 * np.cos(2 * np.pi * p) + 0.2783 * np.cos(4 * np.pi * p) \
            - 0.0837 * np.cos(6 * np.pi * p) + 0.0060 * np.cos(8 * np.pi * p)

    elif name in ['gauss', 'gausswin']:
        if "param" not in kwargs.keys():
            kwargs["param"] = 2.5
        w = np.exp(-0.5 * (kwargs["param"] * 2 * (p - 0.5)) ** 2)

    elif name == 'hamming':
        w = 0.54 - 0.46 * np.cos(2 * np.pi * p)

    elif name == 'hann':
        w = 0.5 - 0.5 * np.cos(2 * np.pi * p)

    elif name == 'kaiser':
        if "param" not in kwargs.keys():
            kwargs["param"] = 0.5
        w = scipy.special.jv(0, kwargs["param"] * np.sqrt(1 - (2 * p - 1) ** 2)) / scipy.special.jv(0, kwargs["param"])

    elif name == 'lanczos':
        w = np.sin(np.pi * (2 * p -1)) / (np.pi * (2 * p - 1))
        w[np.isnan(w)] = 1

    elif name in ['nuttall','nuttallwin']:
        w = 0.3635819 - 0.4891775 * np.cos(2 * np.pi * p) + 0.1365995 * np.cos(4 * np.pi * p) \
            - 0.0106411 * np.cos(6 * np.pi * p)

    elif name in ['rect', 'rectwin']:
        w = np.ones((1, m))

    elif name == 'triang':
        w = 1 - np.abs((np.arange(m - 1) - (m - 1) / 2) / ((m + 1) / 2))

    else:
        # fall back to the Signal Processing toolbox for unknown windows () scipy
        w = scipy.signal.windows.get_window(name, m, *kwargs)

    return w.ravel()