import numpy as np
import logging
from mne.io.eeglab.eeglab import RawEEGLAB


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
    include_channels = np.ones((1, signal.info["nchan"]))
    eps = np.finfo(float).eps
    for c in range(signal.info["nchan"]):
        allowed_or_not = np.abs(np.diff(signal._data[c, :])) < (max_allowed_jitter * eps)
        zero_intervals = np.diff([False, allowed_or_not, False]).nonzero()[0].reshape(2, 1)

        if max(zero_intervals[:, 2]) - max(zero_intervals[:, 1]) > (max_flatline_duration * signal.info["sfreq"]):
            include_channels[c] = 0

    # remove them
    if sum(include_channels) == signal.info["nchan"]:
        logging.warning("All channels have a flat-line portion; not removing anything.")

    else:
        logging.info("Now removing flat-line channels...")

        if len(signal.info['chs']) == len(signal._data):
            # update info
            signal.info["chs"] = [i for (i, v) in zip(signal.info["chs"], include_channels) if v]
            # signal.info["bads"] = [i for (i, v) in zip(signal.ch_names, include_channels) if not v]
            signal.nbchan = sum(include_channels)

            # apply cleaning
            signal.data = signal.data[include_channels, :]
            pos = signal._get_channel_positions(signal.ch_names)[include_channels, :]
            signal._set_channel_positions(pos, signal.ch_names)

    return signal