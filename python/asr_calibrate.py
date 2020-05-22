from typing import Dict, Any, Optional
import numpy as np
import logging
from scipy import linalg
from scipy.signal import lfilter, lfilter_zi
from mne.io.eeglab.eeglab import RawEEGLAB

from python.helpers.fit_eeg_distribution import fit_eeg_distribution
from python.helpers.block_geometric_median import block_geometric_median
from python.helpers.utils import _sliding_window, _remove_nan
from python.helpers.yukewalk import yulewalk


def asr_calibrate(signal: RawEEGLAB, sfreq: float , cutoff: float = 10., blocksize: int = 5, window_len: float = 0.5,
                  window_overlap: float = 0.66, max_dropout_fraction: float = 0.1, min_clean_fraction: float = 0.25
                  ) -> Dict[str, Optional[Any]]:
    """Calibration function for the Artifact Subspace Reconstruction method.

    The input to this data is a multi-channel time series of calibration data.
    In typical uses the calibration data is clean resting EEG data of ca. 1
    minute duration (can also be longer). One can also use on-task data if the
    fraction of artifact content is below the breakdown point of the robust
    statistics used for estimation (50% theoretical, ~30% practical). If the
    data has a proportion of more than 30-50% artifacts then bad time windows
    should be removed beforehand. This data is used to estimate the thresholds
    that are used by the ASR processing function to identify and remove
    artifact components.

    The calibration data must have been recorded for the same cap design from
    which data for cleanup will be recorded, and ideally should be from the
    same session and same subject, but it is possible to reuse the calibration
    data from a previous session and montage to the extent that the cap is
    placed in the same location (where loss in accuracy is more or less
    proportional to the mismatch in cap placement).
    The calibration data should have been high-pass filtered (for example at
    0.5Hz or 1Hz using a Butterworth IIR filter).

    Parameters
    ----------
    X : RawEEGLAB
        Calibration data [#channels x #samples]; *zero-mean* (e.g., high-pass filtered) and
        reasonably clean EEG of not much less than 30 seconds length (this method is typically
        used with 1 minute or more).
    sfreq : float
        Sampling rate of the data, in Hz.
    cutoff: float (default: 10)
        Standard deviation cutoff for rejection. X portions whose variance
        is larger than this threshold relative to the calibration data are
        considered missing data and will be removed. The most aggressive value
        that can be used without losing too much EEG is 2.5. A quite
        conservative value would be 5.
    blocksize : int (default: 5)
        Block size for calculating the robust data covariance and thresholds,
        in samples; allows to reduce the memory and time requirements of the
        robust estimators by this factor.
    window_len : float (default: 0.5)
        Window length that is used to check the data for artifact content. This
        is ideally as long as the expected time scale of the artifacts but
        short enough to allow for several 1000 windows to compute statistics
        over.
    window_overlap : float (default: 0.66)
        Window overlap fraction. The fraction of two successive windows that
        overlaps. Higher overlap ensures that fewer artifact portions are going
        to be missed, but is slower.
    max_dropout_fraction : float (default: 0.1)
        Maximum fraction of windows that can be subject to signal dropouts
        (e.g., sensor unplugged), used for threshold estimation.
    min_clean_fraction : float (default: 0.25)
        Minimum fraction of windows that need to be clean, used for threshold
        estimation.

    Returns
    -------
    state: Dict[str, Any]

    """
    logging.info('ASR Calibrating...')

    # window length for calculating thresholds
    X = _remove_nan(signal.get_data())
    C, S = X.shape
    window_len = int(window_len * sfreq)
    window_stride = int(np.round(window_len * (1 - window_overlap)))

    # use yulewalk to design the filter
    # Initialise yulewalk-filter coefficients with sensible defaults
    F = np.array([0, 2, 3, 13, 16, 40, np.minimum(80.0, (sfreq / 2.0) - 1.0), sfreq / 2.0]) * 2.0 / sfreq
    M = np.array([3, 0.75, 0.33, 0.33, 1, 1, 3, 3])
    B, A = yulewalk(8, F, M)

    # apply the signal shaping filter and initialize the IIR filter state
    X[~np.isfinite(X)] = 0
    # zi = lfilter_zi(B, A)
    # X, zf = lfilter(B, A, X, zi=zi)
    X = lfilter(B, A, X)

    if np.any(~np.isfinite(X)):
        logging.error("The IIR filter diverged on your data. Please try using either a more conservative filter "
                      "or removing some bad sections/channels from the calibration data.")

    U = np.zeros((blocksize, C, C))
    for k in range(blocksize):
        rangevect = np.minimum(S - 1, np.arange(k, S + k, blocksize))
        x = X[:, rangevect]
        U[k, ...] = x @ x.T
    Uavg = block_geometric_median(U.reshape((-1, C * C)) / blocksize, 2)
    Uavg = Uavg.reshape((C, C))

    # get the mixing matrix M
    M = linalg.sqrtm(np.real(Uavg))
    D, Vtmp = np.linalg.eig(M)
    # D, Vtmp = nonlinear_eigenspace(M, C)
    V = Vtmp[:, np.argsort(D)]

    # get the threshold matrix T
    x = np.abs(np.dot(V, X))
    x = _sliding_window(x, window=window_len, steps=window_stride)

    mu = np.empty((C))
    sigma = np.empty((C))
    for c in reversed(range(C)):
        # compute RMS amplitude for each window
        rms = np.sqrt(np.sum(x[c, :, :] ** 2, axis=1) / window_len)

        # robustly fit a distribution to the clean EEG part
        _mu, _sigma, _, _ = fit_eeg_distribution(X=rms,
                                                 min_clean_fraction=min_clean_fraction,
                                                 max_dropout_fraction=max_dropout_fraction)
        mu[c], sigma[c] = _mu, _sigma
    T = np.dot(np.diag(mu + cutoff * sigma), V.T)
    logging.info('ASR calibration complete.')

    return {"M": M, "T": T, "B": B, "A": A}
