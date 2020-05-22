from typing import Tuple, List
import numpy as np
import logging
import mne
from mne.io.eeglab.eeglab import RawEEGLAB
from tqdm import tqdm

from python.helpers.utils import _sliding_window, _remove_nan
from python.helpers.decorators import catch_exception
from python.helpers.fit_eeg_distribution import fit_eeg_distribution


@catch_exception
def clean_windows(signal: RawEEGLAB, max_bad_channels: float = .2, z_thresholds: Tuple[float, float] = (-3.5, 5),
                  window_len: float = .66, window_overlap: float = .66, max_dropout_fraction: float = .1,
                  min_clean_fraction: float = .25, truncate_quant: Tuple[float, float] = (0.022, 0.6),
                  step_sizes: Tuple[float, float] = (0.01, 0.01),
                  shape_range: Tuple[float, float, float] = (1.7, 3.5, .15)) -> RawEEGLAB:
    """Remove periods with abnormally high-power content from continuous data.

    This function cuts segments from the data which contain high-power artifacts.
    Specifically, only windows are retained which have less than a certain fraction of "bad" channels, where a channel
    is bad in a window if its power is above or below a given upper/lower threshold (in standard deviations from a
    robust estimate of the EEG power distribution in the channel).

    Parameters
    ----------
    signal: RawEEGLAB
        Continuous data set, assumed to be appropriately high-passed (e.g. >1Hz or 0.5Hz - 2.0Hz transition band)
    max_bad_channels: float (default: 0.2)
        The maximum number or fraction of bad channels that a retained window may still contain (more than this and it
        is removed). Reasonable range is 0.05 (very clean output) to 0.3 (very lax cleaning of only coarse artifacts).
    z_thresholds: Tuple[float, float] (default: (-3.5, 5))
        The minimum and maximum standard deviations within which the power of a channel must lie (relative to a robust
        estimate of the clean EEG power distribution in the channel) for it to be considered "not bad".
    window_len: int (default: 0.66)
        Window length that is used to check the data for artifact content. This is ideally as long as the expected
        time scale of the artifacts but not shorter than half a cycle of the high-pass filter that was used.
    window_overlap: float (default: 0.66)
        Window overlap fraction. The fraction of two successive windows that overlaps.Higher overlap ensures
        that fewer artifact portions are going to be missed (but is slower).
    max_dropout_fraction: float (default: 0.1)
        Maximum fraction that can have dropouts. This is the maximum fraction of time windows that may have arbitrarily
        low amplitude (e.g., due to the sensors being unplugged).
    min_clean_fraction: float (default: 0.25)
        Minimum fraction that needs to be clean. This is the minimum fraction of time windows that need to contain
        essentially uncontaminated EEG.
    truncate_quant: Tuple[float, float] (default: (0.022, 0.6))
        Truncated Gaussian quantile. Quantile range [upper,lower] of the truncated Gaussian distribution that shall
        be fit to the EEG contents.
    step_sizes: Tuple[float, float] (default: (0.01, 0.01))
        Grid search stepping. Step size of the grid search, in quantiles; separately for [lower,upper] edge of the
        truncated Gaussian. The lower edge has finer stepping because the clean data density is assumed to be lower
        there, so small changes in quantile amount to large changes in data space.
    shape_range: Tuple[float, float, float] (default: (1.7, 3.5 0.15))
        Shape parameter range (start, stop, step). Search range for the shape parameter of the generalized Gaussian
        distribution used to fit clean EEG.

    Returns
    -------
    RawEEGLAB
        data set with bad time periods removed.
    List[bool]
        mask of retained samples.

    """
    X = _remove_nan(signal.get_data())
    C, S = X.shape

    window_len = int(window_len * signal.info["sfreq"])
    window_stride = int(np.round((window_len * (1 - window_overlap))))

    logging.info("Determining time window rejection thresholds...")
    X = _sliding_window(X, window=window_len, steps=window_stride)[:, :-1, :]
    z_rms = np.zeros(X.shape[:-1])
    for c in tqdm(reversed(range(C)), total=C):
        # compute RMS amplitude for each window
        _X = np.sqrt(np.sum(X[c, :, :] ** 2, axis=1) / window_len)

        # robustly fit a distribution to the clean EEG part
        mu, sigma, _, _ = fit_eeg_distribution(X=_X,
                                               min_clean_fraction=min_clean_fraction,
                                               max_dropout_fraction=max_dropout_fraction,
                                               quants=truncate_quant,
                                               step_sizes=step_sizes,
                                               beta=shape_range)

        # calculate z scores relative to that
        z_rms[c, :] = (_X - mu) / sigma

    # sort z scores into quantiles
    sorted_z_rms = np.sort(z_rms, axis=0)

    # determine which windows to remove
    max_bad_channels = int(np.round(C * max_bad_channels))
    remove_windows = np.zeros((sorted_z_rms.shape[1],)).astype(bool)
    if np.max(z_thresholds) > 0:
        remove_windows[sorted_z_rms[-max_bad_channels, :] > np.max(z_thresholds)] = True

    if np.min(z_thresholds) < 0:
        remove_windows[sorted_z_rms[max_bad_channels, :] < np.min(z_thresholds)] = True

    # apply removal
    logging.info('Removing windows...')
    # annotate bad windows that will be dropped
    onsets = np.round(np.arange(S - window_len, step=window_stride)).astype(int)
    onsets_in_seconds = signal.times[onsets]
    signal.annotations.append(onset=onsets_in_seconds[remove_windows],
                              duration=window_len / signal.info["sfreq"],
                              description=["bad_win"] * len(onsets_in_seconds[remove_windows]))

    # drop them
    sample_mask = np.zeros((S,)).astype(bool)
    for o in onsets[remove_windows]:
        sample_mask[o:o+window_len] = True

    signal._data[:, sample_mask] = np.nan
    # signal.n_times = signal._data.shape[1]
    # signal.times.max = signal.times.min + (signal.n_time - 1) / signal.info["srate"]

    return signal