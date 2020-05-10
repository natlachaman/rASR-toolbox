import numpy as np
import scipy
import logging

from mne.io.eeglab.eeglab import RawEEGLAB
from .helpers.design_fir import design_fir
from .helpers.design_kaiser import design_kaiser
from .helpers.utils import _sliding_window

def clean_channels_nolocs(signal: RawEEGLAB, min_corr: float = .45, ignored_quantile: float = .1, window_len: int = 2,
                          max_broken_time: float = 0.5, linenoise_aware: bool = True) -> RawEEGLAB:
    """Remove channels with abnormal data from a continuous data set.

    This is an automated artifact rejection function which ensures that the data contains no channels that
    record only noise for extended periods of time. If channels with control signals are contained in the data
    these are usually also removed. The criterion is based on correlation: if a channel is decorrelated from all
    others (pairwise correlation < a given threshold), excluding a given fraction of most correlated channels --
    and if this holds on for a sufficiently long fraction of the data set -- then the channel is removed.

    Parameters
    ----------
    signal: RawEEGLAB
        Continuous data set, assumed to be appropriately high-passed (e.g. >0.5Hz or with a 0.5Hz - 2.0Hz
        transition band).
    min_corr: float (default: 0.5)
        Minimum correlation between a channel and any other channel (in a short period of time) below which the
        channel is considered abnormal for that time period. Reasonable range: 0.4 (very lax) to 0.6 (quite aggressive).
    ignored_quantile: float (default: 0.1)
        Fraction of channels that need to have at least the given MinCorrelation value w.r.t. the channel under
        consideration. This allows to deal with channels or small groups of channels that measure the same noise source,
         e.g. if they are shorted. If many channels can be disconnected during an experiment and you have strong noise
         in the room, you might increase this fraction, but consider that this a) requires you to decrease the
         MinCorrelation appropriately and b) this can make the correlation measure more brittle.
         Reasonable range: 0.05 (rather lax) to 0.2 (very tolerant re disconnected/shorted channels).
    window_len: int (default: 2)
        Length of the windows (in seconds) for which correlation is computed; ideally short enough to reasonably
        capture periods of global artifacts (which are ignored), but not shorter (for statistical reasons).
    max_broken_time: float (default: 0.5)
        Maximum time (either in seconds or as fraction of the recording) during which a retained channel may be broken.
        Reasonable range: 0.1 (very aggressive) to 0.6 (very lax).
    linenoise_aware: bool (default: True)
        Whether the operation should be performed in a line-noise aware manner. If enabled, the correlation measure
        will not be affected by the presence or absence of line noise (using a temporary notch filter).

    Returns
    -------
    RawEEGLAB
        data set with bad channels removed

    """
    # flag channels
    if (max_broken_time >= 0) and (max_broken_time <= 1):
        max_broken_time = signal._data.shape[1] * max_broken_time
    else:
        max_broken_time = signal.info["sfreq"] * max_broken_time

    # optionally ignore both 50 and 60 Hz spectral components...
    C, S = signal._data.shape
    if linenoise_aware:
        B = design_kaiser(lo=2 * 45 / signal.info["sfreq"],
                          hi=2 * 50 / signal.info["sfreq"],
                          attenuation=60,
                          odd=True)

        if signal.info["sfreq"] <= 130:
            F = np.r_[np.array([0, 45, 50, 55]) * 2 / signal.info["sfreq"], 1]
            A = np.array([1, 1, 0, 1, 1])
        else:
            F = np.r_[np.array([0, 45, 50, 55, 60, 65]) * 2 / signal.info["sfreq"], 1]
            A = np.array([1, 1, 0, 1, 0, 1, 1])
        B = design_fir(N=len(B) - 1, F=F, A=A, wnd=B)
        X = np.vstack([scipy.signal.filtfilt(B, 1, signal._data[c, :]) for c in reversed(range(C))])

    else:
        X = signal._data

    # for each window, flag channels with too low correlation to any other channel (outside the ignored quantile)
    flagged = []
    retained = np.arange(C - np.ceil(C * ignored_quantile))
    window_len *= signal.info["sfreq"]
    for x in _sliding_window(X, window=window_len):
        sort_cc = np.sort(np.abs(np.corrcoef(x)), axis=0)
        flagged.append(np.all(sort_cc[retained, :] < min_corr, axis=1))

    include_channels = np.sum(np.vstack(flagged).T, axis=1) * window_len <= max_broken_time

    # apply removal
    if np.all(~include_channels):
        logging.warning("All channels are flagged bad according to the used criterion: not removing anything.")

    else:
        logging.info("Now removing bad channels...")

        if len(signal.info["chs"]) == len(signal._data):
            # update info
            signal.info["chs"] = [i for (i, v) in zip(signal.info["chs"], include_channels) if v]
            # signal.info["bads"] = [i for (i, v) in zip(signal.ch_names, include_channels) if not v]
            signal.nbchan = sum(include_channels)

            # apply cleaning
            signal.data = signal.data[include_channels, :]
            pos = signal._get_channel_positions(signal.ch_names)[include_channels, :]
            signal._set_channel_positions(pos, signal.ch_names)

            signal.info["clean_channel_mask"] = include_channels

    return signal