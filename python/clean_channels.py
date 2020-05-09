import numpy as np
import scipy
import logging

from mne.io.eeglab.eeglab import RawEEGLAB
from mne.channels.interpolation import _make_interpolation_matrix
from .helpers.design_fir import design_fir


def clean_channels(signal: RawEEGLAB, corr_threshold: float = 0.85, noise_threshold: int = 4, window_len: int = 5,
                   max_broken_time: float = 0.4, num_samples: int = 50, subset_size: float = 0.25) -> RawEEGLAB:
    """Remove channels with abnormal data from a continuous data set.

    This is an automated artifact rejection function which ensures that the data contains no channels
    that record only noise for extended periods of time. If channels with control signals are
    contained in the data these are usually also removed. The criterion is based on correlation: if a
    channel has lower correlation to its robust estimate (based on other channels) than a given threshold
    for a minimum period of time (or percentage of the recording), it will be removed.

    Parameters
    ----------
    signal: RawEEGLAB
        Continuous data set, assumed to be appropriately high-passed (e.g. >0.5Hz or with a 0.5Hz - 2.0Hz
        transition band).
    corr_threshold: float (default: 0.85)
        Correlation threshold. If a channel is correlated at less than this value to its robust estimate
        (based on other channels), it is considered abnormal in the given time window.
    noise_threshold: int (default: 4)
        If a channel has more line noise relative to its signal than this value, in standard deviations from the
        channel population mean, it is considered abnormal.
    window_len: int (default: 5)
        Length of the windows (in seconds) for which correlation is computed; ideally short enough to reasonably
        capture periods of global artifacts or intermittent sensor dropouts, but not shorter (for statistical reasons).
    max_broken_time: float (default: 0.4)
        Maximum time (either in seconds or as fraction of the recording) during which a retained channel may be broken.
        Reasonable range: 0.1 (very aggressive) to 0.6 (very lax).
    num_samples: int (default: 50)
        Number of RANSAC samples. This is the number of samples to generate in the random sampling consensus process.
        The larger this value, the more robust but also slower the processing will be.
    subset_size: float (default: 0.25)
        Subset size. This is the size of the channel subsets to use for robust reconstruction, as a fraction of the
        total number of channels.

    Returns
    -------
    RawEEGLAB
        data set with bad channels removed

    """
    def _mad(X):
        """Median absolute deviation."""
        return np.median(np.abs(X - np.median(X, axis=0)), axis=0)

    def _sliding_window(array, window, steps=1, axis=1):
        """Efficient sliding window."""
        # Sub-array shapes
        shape = list(array.shape)
        shape[axis] = np.ceil(array.shape[axis] / steps - window / steps + 1).astype(int)
        shape.append(window)

        # Strides (in bytes)
        strides = list(array.strides)
        strides[axis] *= steps
        strides.append(array.strides[axis])

        # Window samples
        return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    # flag channels
    if (max_broken_time >= 0) and (max_broken_time <= 1):
        max_broken_time = signal._data.shape[1] * max_broken_time
    else:
        max_broken_time = signal.info["sfreq"] * max_broken_time

    C, S = signal._data.shape
    logging.info("Scannning for bad channels...")
    if signal.info["sfreq"] > 100:

        # remove signal content above 50 Hz
        F = np.r_[np.array([0, 45, 50]) * 2 / signal.info["sfreq"], 1]
        A = np.array([1, 1, 0, 0])
        B = design_fir(100, F, A)

        X = np.vstack([scipy.signal.filtfilt(B, 1, signal._data[c, :]) for c in reversed(range(C))])

        # determine z-scored level of EM noise-to-signal ratio for each channel
        noisiness = _mad(signal._data - X) / _mad(X)
        znoise = (noisiness - np.median(noisiness, axis=0)) / (_mad(noisiness) * 1.4826)

        # trim channels based on that
        noise_mask = znoise < noise_threshold
    else:

        X = signal._data
        # transpose added. Otherwise gives an error below at removed_channels = removed_channels | noise_mask
        noise_mask = np.ones(C)

    # test spherical fit
    # todo: origin?
    pos = signal._get_channel_positions(signal.ch_names)
    origin = (0., 0., 0.)
    distance = np.linalg.norm(pos - origin, axis=-1)
    distance = np.mean(distance / np.mean(distance))
    if np.abs(1. - distance) > 0.1:
        logging.warning('Your spherical fit is poor, interpolation results are likely to be inaccurate.')

    # interpolation: RANSAC method
    np.random.seed(435656)
    subset_size = int(np.round(subset_size * len(signal._data)))
    logging.info('Computing interpolation matrix from {} sensor positions using RANSAC method'.format(len(pos)))
    interpolation = []
    for _ in reversed(range(num_samples)):
        subset = np.random.choice(len(pos), size=subset_size, replace=False)
        interpolation.append( _make_interpolation_matrix(pos[subset, :], pos))
    interpolation = np.vstack(interpolation)

    # calculate each channel's correlation to its RANSAC reconstruction for each window
    corrs = [] # (channels, window)
    for x in _sliding_window(X, window=window_len * signal.info["sfreq"]):
        y = np.sort(np.reshape(interpolation * x, (window_len, C, num_samples)), axis=2)
        y = y[:, :, int(np.round(num_samples / 2))]
        corrs.append(np.sum(y * x, axis=1) / (np.sqrt(np.sum(x ** 2, axis=1)) * np.sqrt(np.sum(y ** 2, axis=1))))

    # flag channels to include
    flagged = np.vstack(corrs) < corr_threshold
    include_channels = np.sum(flagged, axis=1) * (window_len * signal.info["sfreq"]) < max_broken_time
    include_channels = np.logical_or(include_channels, noise_mask)

    # remove them
    if np.mean(~include_channels) > 0.75:
        logging.warning("More than 75% of your channels were removed -- "
                        "this is probably caused by incorrect channel location measurements (e.g., wrong cap design).")

    else:
        logging.info(f"Removing channels {signal.ch_names[~include_channels]} and dropping signal meta-data...")
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

