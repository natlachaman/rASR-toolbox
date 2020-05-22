import numpy as np
from scipy.signal import filtfilt
import logging
from tqdm import tqdm

from mne.io.eeglab.eeglab import RawEEGLAB
from mne.channels.interpolation import _make_interpolation_matrix
from python.helpers.design_fir import design_fir
from python.helpers.utils import _mad, _sliding_window
from python.helpers.decorators import catch_exception


@catch_exception
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
    X = signal.get_data()
    X = X[~np.isnan(X)]
    C, S = X.shape

    # flag channels
    if (max_broken_time >= 0) and (max_broken_time <= 1):
        max_broken_time = S * max_broken_time
    else:
        max_broken_time = signal.info["sfreq"] * max_broken_time

    # optionally ignore < 50 Hz spectral components...
    logging.info("Scannning for bad channels...")
    if signal.info["sfreq"] > 100:
        # remove signal content above 50 Hz
        F = np.r_[np.array([0, 45, 50]) * 2 / signal.info["sfreq"], 1]
        A = np.array([1, 1, 0, 0])
        B = design_fir(100, F, A)

        X_ = np.vstack([filtfilt(B, 1, X[c, :]) for c in reversed(range(C))])

        # determine z-scored level of EM noise-to-signal ratio for each channel
        noisiness = _mad(X - X_) / _mad(X_)
        znoise = (noisiness - np.median(noisiness, axis=0)) / (_mad(noisiness) * 1.4826)

        # trim channels based on that
        noise_mask = znoise < noise_threshold
    else:
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
    subset_size = int(np.round(subset_size * C))
    logging.info('Computing interpolation matrix from {} sensor positions using RANSAC method'.format(len(pos)))
    interpolation = []
    for _ in range(num_samples):
        interp_sample = np.zeros((C, C))
        subset = np.random.choice(len(pos), size=subset_size, replace=False)
        interp_sample[subset, :] = _make_interpolation_matrix(pos[subset, :], pos).T
        interpolation.append(interp_sample)
    interpolation = np.hstack(interpolation)

    # calculate each channel's correlation to its RANSAC reconstruction for each window
    logging.info("Computing each channel's correaltion to its RANSAC reconstruction for each window..")
    window_len *= int(signal.info["sfreq"])
    corrs = [] # (channels, window)
    for x in tqdm(_sliding_window(X, window=window_len, steps=window_len, axis=1).swapaxes(0, 1)):
        y = np.sort(np.reshape(np.dot(x.T, interpolation), (window_len, C, num_samples)), axis=2)
        y = y[:, :, int(np.round(num_samples / 2))]
        corrs.append(np.sum(np.dot(x, y), axis=1) / (np.sqrt(np.sum(x ** 2, axis=1)) * np.sqrt(np.sum(y ** 2, axis=0))))

    # flag channels to include
    flagged = np.vstack(corrs) < corr_threshold
    include_channels = np.sum(flagged, axis=0) * window_len <= max_broken_time
    include_channels = np.logical_or(include_channels, noise_mask)

    # remove them
    if np.mean(~include_channels) > 0.75:
        logging.exception("More than 75% of your channels were removed -- "
                          "this is probably caused by incorrect channel " \
                          "location measurements (e.g., wrong cap design).")
        raise

    else:
        logging.info(f"Removing bad channels...")
        # update info
        good_channels = set(signal.ch_names).difference(set(signal.info["bads"]))
        signal.info["bads"] = [i for (i, v) in zip(good_channels, include_channels) if not v]
        signal.drop_channels(signal.info["bads"])
        # signal.info["clean_channel_mask"] = include_channels

    return signal

