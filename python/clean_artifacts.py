from typing import Tuple, Union
import logging
from mne.io.eeglab.eeglab import RawEEGLAB

from python.clean_windows import clean_windows
from python.clean_channels import clean_channels
from python.clean_channels_nolocs import clean_channels_nolocs
from python.clean_drifts import clean_drifts
from python.clean_flatlines import clean_flatlines
from python.clean_asr import clean_asr
from python.helpers.decorators import catch_exception


@catch_exception
def clean_artifacts(signal:RawEEGLAB, channel_criterion: float = .85, line_noise_criterion: int = 4,
                    burst_criterion: int = 5, window_criterion: float =.25, highpass: Tuple[float, float] = (.25,.75),
                    channel_criterion_max_bad_time: float = .5,
                    burst_criterion_ref_max_bad_chns: Union[float, None] = 0.075,
                    burst_criterion_ref_tolerances: Union[Tuple[float, float], None] = (-3.5, 5.5),
                    window_criterion_tolerances: Union[Tuple[float, float], None] = (-3.5, 7),
                    flatline_criterion: int = 5, nolocs_channel_criterion: float = .45,
                    noloc_channel_criterion_excluded: float = .1
                    ) -> RawEEGLAB:
    """All-in-one function for artifact removal, including ASR.

    This function removes flatline channels, low-frequency drifts, noisy channels, short-time bursts
    and incompletely repaird segments from the data. Tip: Any of the core parameters can also be
    passed in as [] to use the respective default of the underlying functions, or as 'off' to disable
    it entirely.

    Hopefully parameter tuning should be the exception when using this function -- however, there are
    3 parameters governing how aggressively bad channels, bursts, and irrecoverable time windows are
    being removed, plus several detail parameters that only need tuning under special circumstances.

    Parameters
    ----------
    signal: RawEEGLAB
        Raw continuous EEG recording to clean up (as EEGLAB dataset structure).

    NOTE: The following parameters are the core parameters of the cleaning procedure.
    If the method removes too many (or too few) channels, time windows, or general high-amplitude ("burst") artifacts,
    you will want to tune these values. Hopefully you only need to do this in rare cases.

    channel_criterion: float (default: 0.85)
        Minimum channel correlation. If a channel is correlated at less than this
        value to an estimate based on other channels, it is considered abnormal in
        the given time window. This method requires that channel locations are
        available and roughly correct; otherwise a fallback criterion will be used.
    line_noise_criterion: int (default: 4)
        If a channel has more line noise relative to its signal than this value, in
        standard deviations based on the total channel population, it is considered
        abnormal.
    burst_criterion: int (default: 5)
        Standard deviation cutoff for removal of bursts (via ASR). Data portions whose
        variance is larger than this threshold relative to the calibration data are
        considered missing data and will be removed. The most aggressive value that can
        be used without losing much EEG is 3. For new users it is recommended to at
        first visually inspect the difference between the original and cleaned data to
        get a sense of the removed content at various levels. A quite conservative
        value is 5.
    window_criterion: float (default: 0.25)
        Criterion for removing time windows that were not repaired completely. This may
        happen if the artifact in a window was composed of too many simultaneous
        uncorrelated sources (for example, extreme movements such as jumps). This is
        the maximum fraction of contaminated channels that are tolerated in the final
        output data for each considered window. Generally a lower value makes the
        criterion more aggressive. Reasonable range: 0.05 (very aggressive) to 0.3 (very lax).
    highpass: Tuple[float, float] (default: (0.25, 0.75))
        Transition band for the initial high-pass filter in Hz. This is formatted as [transition-start, transition-end].

    NOTE: The following are detail parameters that may be tuned if one of the criteria does
    not seem to be doing the right thing. These basically amount to side assumptions about the
    data that usually do not change much across recordings, but sometimes do.

    channel_criterion_max_bad_time: float (default: 0.5)
        This is the maximum tolerated fraction of the recording duration
        during which a channel may be flagged as "bad" without being
        removed altogether. Generally a lower (shorter) value makes the
        criterion more aggresive. Reasonable range: 0.15 (very aggressive)
        to 0.6 (very lax).
    burst_criterion_ref_max_bad_chns: Union[float, None] (default: 0.075)
        If a number is passed in here, the ASR method will be calibrated based
        on sufficiently clean data that is extracted first from the
        recording that is then processed with ASR. This number is the
        maximum tolerated fraction of "bad" channels within a given time
        window of the recording that is considered acceptable for use as
        calibration data. Any data windows within the tolerance range are
        then used for calibrating the threshold statistics. Instead of a
        number one may also directly pass in a data set that contains
        calibration data (for example a minute of resting EEG).

        If this is set to 'None', all data is used for calibration. This will
        work as long as the fraction of contaminated data is lower than the
        the breakdown point of the robust statistics in the ASR
        calibration (50, where 30 of clearly recognizable artifacts is a
        better estimate of the practical breakdown point).

        A lower value makes this criterion more aggressive. Reasonable
        range: 0.05 (very aggressive) to 0.3 (quite lax). If you have lots
        of little glitches in a few channels that don't get entirely
        cleaned you might want to reduce this number so that they don't go
        into the calibration data.
    burst_criterion_ref_tolerances: Union[Tuple[float, float], None] (default: (-3.5, 5.5))
        These are the power tolerances outside of which a channel in a
        given time window is considered "bad", in standard deviations
        relative to a robust EEG power distribution (lower and upper
        bound). Together with the previous parameter this determines how
        ASR calibration data is be extracted from a recording. Can also be
        specified as 'None' to achieve the same effect as in the previous
        parameter.
    window_criterion_tolerances: Union[Tuple[float, float], None] (default: (-3.5, 7.))
        These are the power tolerances outside of which a channel in the final
        output data is considered "bad", in standard deviations relative
        to a robust EEG power distribution (lower and upper bound). Any time
        window in the final (repaired) output which has more than the
        tolerated fraction (set by the WindowCriterion parameter) of channel
        with a power outside of this range will be considered incompletely
        repaired and will be removed from the output. This last stage can be
        skipped either by setting the WindowCriterion to 'off' or by taking
        the third output of this processing function (which does not include
        the last stage).
    flatline_criterion: int (default: 5)
        Maximum tolerated flatline duration. In seconds. If a channel has a longer
        flatline than this, it will be considered abnormal.
    nolocs_channel_criterion:  float (defaut: 0.45)
        Criterion for removing bad channels when no channel locations are
        present. This is a minimum correlation value that a given channel must
        have w.r.t. a fraction of other channels. A higher value makes the
        criterion more aggressive. Reasonable range: 0.4 (very lax) - 0.6
        (quite aggressive).
    noloc_channel_criterion_excluded: float (default: 0.1)
        The fraction of channels that must be sufficiently correlated with
        a given channel for it to be considered "good" in a given time
        window. Applies only to the NoLocsChannelCriterion. This adds
        robustness against pairs of channels that are shorted or other
        that are disconnected but record the same noise process.
        Reasonable range: 0.1 (fairly lax) to 0.3 (very aggressive);
        note that increasing this value requires the ChannelCriterion
        to be relaxed in order to maintain the same overall amount of
        removed channels.

    Returns
    -------
    RawEEGLAB
        Final cleaned EEG recording.

    Notes
    -----
    * This function uses the Signal Processing toolbox for pre- and post-processing of the data
    (removing drifts, channels and time windows); the core ASR method (clean_asr) does not require
    this toolbox but you will need high-pass filtered data if you use it directly.
    * By default this function will identify subsets of clean data from the given recording to
    enhance the robustness of the ASR calibration phase to strongly contaminated data; this uses
    the Statistics toolbox, but can be skipped/bypassed if needed (see documentation).

    """

    EEG = clean_flatlines(signal, flatline_criterion)
    EEG = clean_drifts(EEG, highpass)

    try:
        EEG = clean_channels(signal=EEG,
                             corr_threshold=channel_criterion,
                             noise_threshold=line_noise_criterion,
                             max_broken_time=channel_criterion_max_bad_time)
    except Exception:
        logging.info("Your dataset appears to lack correct channel locations; "
                     "using a location-free channel cleaning method.")

        EEG = clean_channels_nolocs(signal=EEG,
                                    min_corr=nolocs_channel_criterion,
                                    ignored_quantile=noloc_channel_criterion_excluded,
                                    max_broken_time=channel_criterion_max_bad_time)
    EEG = clean_asr(signal=EEG,
                    cutoff=burst_criterion,
                    ref_maxbadchannels=burst_criterion_ref_max_bad_chns,
                    ref_tolerances=burst_criterion_ref_tolerances)

    logging.info("Now applying final post-cleanup of the output.")
    EEG = clean_windows(signal=EEG,
                        max_bad_channels=window_criterion,
                        z_thresholds=window_criterion_tolerances)

    logging.info("Use EGG.visualize() to compare the cleaned data to the original.")
    return EEG