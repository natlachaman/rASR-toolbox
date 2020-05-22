import numpy as np
import logging
from mne.io.eeglab.eeglab import RawEEGLAB
from python.helpers.decorators import catch_exception
from python.helpers.utils import _remove_nan


@catch_exception
def clean_flatlines(signal: RawEEGLAB, max_flatline_duration: int = 5 ,max_allowed_jitter: int = 20) -> RawEEGLAB:
    """Remove (near-) flat-lined channels.

    This is an automated artifact rejection function which ensures that the data contains no flat-lined channels.

    Parameters
    ----------
    signal: RawEEGLAB
        continuous data set, assumed to be appropriately high-passed (e.g. >0.5Hz or with a 0.5Hz - 2.0Hz
        transition band).
    max_flatline_duration: int (default: 5)
        maximum tolerated flatline duration. In seconds. If a channel has a longer flatline than this,
        it will be considered abnormal.
    max_allowed_jitter: int (default: 20)
        maximum tolerated jitter during flatlines. As a multiple of epsilon.

    Returns
    -------
    RawEEGLAB
        data set with flat channels removed.

    Notes
    -----
    [1] `pop_select()` from EEGLab not implemented. Only the manual implementation is available.
    [2] `signal.icawinv`, `signal.icasphere`, `signal.icaweights`, `signal.icaact`, `signal.stats`, signal.specdata`,
    `signal.specicaact` are not included when the data is read with `mne` package.
    """
    # flag channels
    include_channels = np.ones((signal.info["nchan"],), dtype="bool")
    eps = np.finfo(float).eps
    X = _remove_nan(signal.get_data())

    for c in range(signal.info["nchan"]):
        allowed_or_not = np.r_[False, np.abs(np.diff(X[c, :])) < (max_allowed_jitter * eps)]
        zero_intervals = np.diff(np.r_[allowed_or_not, False]).cumsum() * allowed_or_not
        lengths_intervals = np.unique(zero_intervals, return_counts=True)[1][1:]

        if lengths_intervals.size > 0 and  max(lengths_intervals) > (max_flatline_duration * signal.info["sfreq"]):
            include_channels[c] = 0

    # remove them
    if np.sum(include_channels) == 0:
        logging.warning("All channels have a flat-line portion; not removing anything.")

    else:
        logging.info("Now removing flat-line channels...")
        good_channels = set(signal.ch_names).difference(set(signal.info["bads"]))
        signal.info["bads"] = [i for (i, v) in zip(good_channels, include_channels) if not v]
        signal.drop_channels(signal.info["bads"])

    return signal