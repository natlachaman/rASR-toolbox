from typing import Tuple, List
import numpy as np
import logging
from mne.io.eeglab.eeglab import RawEEGLAB

from .helpers.utils import _sliding_window
from .helpers.fit_eeg_distribution import fit_eeg_distribution


def clean_windows(signal: RawEEGLAB, max_bad_channels: float = .2, z_thresholds: Tuple[float, float] = (-3.5, 5),
                  window_len: float = .66, window_overlap: float = .66, max_dropout_fraction: float = .1,
                  min_clean_fraction: float = .25, truncate_quant: Tuple[float, float] = (0.022, 0.6),
                  step_sizes: Tuple[float, float] = (0.01, 0.01), shape_range: Tuple[float, float, float] = (1.7, 3.5, .15)) \
        -> (RawEEGLAB, List[bool]):
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
    window_len: int (default: 1)
        Window length that is used to check the data for artifact content. This is ideally as long as the expected
        time scale of the artifacts but not shorter than half a cycle of the high-pass filter that was used.
    window_overlap: float (default: 0.66)
        Window overlap fraction. The fraction of two successive windows that overlaps.Higher overlap ensures
        that fewer artifact portions are going to be missed (but is slower).
    max_dropout_fraction: float (default: 0.1)
        Maximum fraction that can have dropouts. This is the maximum fraction of time windows that may have arbitrarily
        low amplitude (e.g., due to the sensors being unplugged).
    min_clean_fraction float (default: 0.25)
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
    C, S = signal._data.shape
    window_len *= signal.srate
    window_stride = window_len * (1 - window_overlap)

    logging.info("Determining time window rejection thresholds...")
    wz = []
    X = _sliding_window(signal._data, window=window_len, steps=window_stride)
    for c in reversed(range(C)):
        # compute RMS amplitude for each window
        X = np.sqrt(np.sum(signal._data[c, :, :] ** 2, axis=1) / window_len)

        # robustly fit a distribution to the clean EEG part
        mu, sigma, _, _ = fit_eeg_distribution(X=X,
                                               min_clean_fraction=min_clean_fraction,
                                               max_dropout_fraction=max_dropout_fraction,
                                               quants=truncate_quant,
                                               step_sizes=step_sizes,
                                               beta=shape_range)

        # calculate z scores relative to that
        wz.append((X - mu) / sigma)

    # sort z scores into quantiles
    swz = np.sort(wz);
    # determine which windows to remove
    remove_mask = false(1, size(swz, 2));
    if max(zthresholds) > 0
        remove_mask(swz(end - max_bad_channels,:) > max(zthresholds)) = true;
        end
    if min(zthresholds) < 0
        remove_mask(swz(1 + max_bad_channels,:) < min(zthresholds)) = true;
        end
    removed_windows = find(remove_mask);