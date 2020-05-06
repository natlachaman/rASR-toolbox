from typing import Tuple
import numpy as np
import scipy
from mne.io.eeglab.eeglab import RawEEGLAB

from .helpers.design_fir import design_fir
from .helpers.design_kaiser import design_kaiser


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
    transition = 2 * (transition / signal.info["sfreq"])

    wnd = design_kaiser(transition[0], transition[1], attenuation, odd=True)
    F, A = np.array([0, transition[0], transition[1], 1]), np.array([0, 0, 1, 1])
    B = design_fir(len(wnd) - 1, F, A, W=wnd)

    # apply it, channel by channel to save memory
    for c in range(signal.info["nchan"]):
        signal._data[c, :] = scipy.signal.filtfilt(B, 1, signal.data[c, :])
    signal.info["clean_drift_kernel"] = B

    return signal