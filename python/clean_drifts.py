from typing import Tuple
import numpy as np
from scipy.signal import filtfilt
from mne.io.eeglab.eeglab import RawEEGLAB

from python.helpers.design_fir import design_fir
from python.helpers.design_kaiser import design_kaiser
from python.helpers.decorators import catch_exception
from python.helpers.utils import _pick_good_channels

@catch_exception
def clean_drifts(signal: RawEEGLAB, transition: Tuple[float, float] = (0.5, 1.), attenuation: int = 80) -> RawEEGLAB:
    """Removes drifts from the data using a forward-backward high-pass filter.

    his removes drifts from the data using a forward-backward (non-causal) filter.
    NOTE: If you are doing directed information flow analysis, do no use this filter but some other one.

    Parameters
    ----------
    signal: RawEEGLAB
        the continuous data to filter
    transition: Tuple[float, float] (default: (0.5, 1.))
        the transition band in Hz, i.e. lower and upper edge of the transition
    attenuation: int (default: 80)
        stop-band attenuation, in db

    Returns
    -------
    RawEEGLAB
        the filtered signal

    """
    # design highpass FIR filter
    transition = 2 * (np.array((transition)) / signal.info["sfreq"])

    wnd = design_kaiser(transition[0], transition[1], attenuation, odd=True)
    F, A = np.array([0, transition[0], transition[1], 1]), np.array([0, 0, 1, 1])
    B = design_fir(len(wnd), F, A, W=wnd)

    # apply it, channel by channel to save memory
    fun_filtfilt = lambda x, b, a: filtfilt(b, a, x) # use
    signal.apply_function(fun=fun_filtfilt, channel_wise=True, b=B, a=1)

    return signal