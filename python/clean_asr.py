from typing import Tuple, Union
import logging
import numpy as np

from mne.io.eeglab.eeglab import RawEEGLAB
from python.clean_windows import clean_windows
from python.asr_calibrate import asr_calibrate
from python.asr_process import asr_process
from python.helpers.decorators import catch_exception


@catch_exception
def clean_asr(signal: RawEEGLAB, stepsize: Union[int, None] = None, cutoff: int = 5, windowlen: float = 0.5,
              maxdims: float = 0.66, ref_maxbadchannels: Union[float, None] = 0.075,
              ref_tolerances: Union[Tuple[float, float], None] = (-3.5, 5.5), ref_wndlen: Union[int, None] = 1
              ) -> RawEEGLAB:
    """Run the ASR method on some high-pass filtered recording.
    
    This is an automated artifact rejection function that ensures that the data contains no events
    that have abnormally strong power; the subspaces on which those events occur are reconstructed 
    (interpolated) based on the rest of the EEG signal during these time periods.

    The basic principle is to first find a section of data that represents clean "reference" EEG and
    to compute statistics on there. Then, the function goes over the whole data in a sliding window
    and finds the subspaces in which there is activity that is more than a few standard deviations
    away from the reference EEG (this threshold is a tunable parameter). Once the function has found
    the bad subspaces it will treat them as missing data and reconstruct their content using a mixing
    matrix that was calculated on the clean data.
    
    Parameters
    ----------
    signal: RawEEGLAB
        continuous data set, assumed to be *zero mean*, e.g., appropriately high-passed (e.g. >0.5Hz or with a 
        0.5Hz - 1.0Hz transition band)
    stepsize: Union[int, None]
        Step size for processing. The reprojection matrix will be updated every this many
        samples and a blended matrix is used for the in-between samples. If None this will
        be set the WindowLength/2 in samples. 
    cutoff: int (default: 5)
        Standard deviation cutoff for removal of bursts (via ASR). Data portions whose variance
        is larger than this threshold relative to the calibration data are considered missing
        data and will be removed. The most aggressive value that can be used without losing
        much EEG is 3. For new users it is recommended to at first visually inspect the difference 
        between the original and cleaned data to get a sense of the removed content at various 
        levels. A quite conservative value is 5.
    windowlen: float (default: 0.5)
        Length of the statistics window, in seconds. This should not be much longer 
        than the time scale over which artifacts persist, but the number of samples in
        the window should not be smaller than 1.5x the number of channels. 
    maxdims: float (default: 0.66)
        Maximum dimensionality to reconstruct. Up to this many dimensions (or up to this 
        fraction of dimensions) can be reconstructed for a given data segment. This is
        since the lower eigenvalues are usually not estimated very well.
    ref_maxbadchannels: Union[float, None] (default: 0.075)
        If a number is passed in here, the ASR method will be calibrated based
        on sufficiently clean data that is extracted first from the recording
        that is then processed with ASR. This number is the maximum tolerated
        fraction of "bad" channels within a given time window of the recording
        that is considered acceptable for use as calibration data. Any data
        windows within the tolerance range are then used for calibrating the
        threshold statistics. Instead of a number one may also directly pass
        in a data set that contains calibration data (for example a minute of
        resting EEG) or the name of a data set in the workspace.

        If this is set to None, all data is used for calibration. This will
        work as long as the fraction of contaminated data is lower than the
        the breakdown point of the robust statistics in the ASR calibration
        (50%, where 30% of clearly recognizable artifacts is a better estimate
        of the practical breakdown point).

        A lower value makes this criterion more aggressive. Reasonable range:
        0.05 (very aggressive) to 0.3 (quite lax). If you have lots of little
        glitches in a few channels that don't get entirely cleaned you might
        want to reduce this number so that they don't go into the calibration
        data.
    ref_tolerances: Union[Tuple[float, float], None] (default: (-3.5, 5.5))
        These are the power tolerances outside of which a channel in a
        given time window is considered "bad", in standard deviations relative to
        a robust EEG power distribution (lower and upper bound). Together with the
        previous parameter this determines how ASR calibration data is be
        extracted from a recording. Can also be specified as 'off' to achieve the
        same effect as in the previous parameter.
    ref_wndlen: Union[int, None] (default: 1)
        Granularity at which EEG time windows are extracted
        for calibration purposes, in seconds.

    Returns
    -------
    RawEEGLAB
        data set with local peaks removed
    
    Notes
    -----
    This function by default attempts to use the Statistics toolbox in order to automatically
    extract calibration data for use by ASR from the given recording. This step is automatically
    skipped if no Statistics toolbox is present (then the entire recording will be used for
    calibration, which is fine for mildly contaminated data -- see ReferenceMaxBadChannels below).

    """
    windowlen = np.max(windowlen, 1.5 * signal["nchan"] / signal["sfreq"])
    stepsize = np.floor(signal["sfreq"] * windowlen / 2) if stepsize is None else stepsize

    # first determine the reference (calibration) data
    logging.info("Finding a clean section of the data...")
    if ref_maxbadchannels is not None and ref_tolerances is not None and ref_wndlen is not None:
        try:
            ref_section = clean_windows(signal=signal,
                                        max_bad_channels=ref_maxbadchannels,
                                        z_thresholds=ref_tolerances,
                                        window_len=ref_wndlen)
        except Exception as e:
            logging.error(e)
            logging.warning("Falling back to using the entire data for calibration.")
            ref_section = signal
    else:
        ref_section = signal

    # calibrate on the reference data
    logging.info("Estimating calibration statistics; this may take a while...")
    state = asr_calibrate(X=ref_section._data,
                          sfreq=ref_section.info["sfreq"],
                          cutoff=cutoff)
    del ref_section
    
    # extrapolate last few samples of the signal
    sig = np.r_[signal._data, (2 * signal._data[:, -1]) - (signal._data[:, -int(windowlen / 2 * signal["sfreq"]): -2])]
    
    # process signal using ASR
    signal._data, state = asr_process(data=sig,
                                      srate=signal["sfreq"],
                                      state=state,
                                      windowlen=windowlen,
                                      lookahead=windowlen / 2,
                                      stepsize=stepsize,
                                      maxdims=maxdims)
    
    # shift signal content back (to compensate for processing delay)
    signal._data = signal._data[:, state["carry"].shape[1]:]

    return signal