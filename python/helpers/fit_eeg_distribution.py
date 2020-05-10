from typing import Tuple
import numpy as np


def fit_eeg_distribution(X: np.ndarray, min_clean_fraction: float = .25, max_dropout_fraction: float = .1,
                         quants: Tuple[float, float] = (0.022, 0.6), step_sizes: Tuple[float, float] = (0.01, 0.01),
                         beta: Tuple[float, float, float] = (1.7, 3.5, 0.15)) -> Tuple[float, float, float, float]:
    """Estimate the mean and standard deviation of clean EEG from contaminated data.
    
    This function estimates the mean and standard deviation of clean EEG from a sample of amplitude
    values (that have preferably been computed over short windows) that may include a large fraction
    of contaminated samples. The clean EEG is assumed to represent a generalized Gaussian component in
    a mixture with near-arbitrary artifact components. By default, at least 25% (MinCleanFraction) of
    the data must be clean EEG, and the rest can be contaminated. No more than 10%
    (MaxDropoutFraction) of the data is allowed to come from contaminations that cause lower-than-EEG
    amplitudes (e.g., sensor unplugged). There are no restrictions on artifacts causing
    larger-than-EEG amplitudes, i.e., virtually anything is handled (with the exception of a very
    unlikely type of distribution that combines with the clean EEG samples into a larger symmetric
    generalized Gaussian peak and thereby "fools" the estimator). The default parameters should be
    fine for a wide range of settings but may be adapted to accomodate special circumstances.

    The method works by fitting a truncated generalized Gaussian whose parameters are constrained by
    MinCleanFraction, MaxDropoutFraction, FitQuantiles, and ShapeRange. The alpha and beta parameters
    of the gen. Gaussian are also returned. The fit is performed by a grid search that always finds a
    close-to-optimal solution if the above assumptions are fulfilled.
    
    Parameters
    ----------
    X: np.ndarray
        Vector of amplitude values of EEG, possible containing artifacts (coming from single samples or windowed
        averages).
    min_clean_fraction: float (default: 0.25)
        Minimum fraction of values in X that needs to be clean.
    max_dropout_fraction: float (default: 0.1)
        Maximum fraction of values in X that can be subject to signal dropouts (e.g., sensor unplugged).
    quants: Tuple[float, float] (default: (0.022, 0.6))
        Quantile range [lower,upper] of the truncated generalized Gaussian distribution that shall be fit to the
        EEG contents.
    step_sizes: Tuple[float, float] (default: (0.01, 0.01))
        Step size of the grid search; the first value is the stepping of the lower bound (which essentially steps over
        any dropout samples), and the second value is the stepping over possible scales (i.e., clean-data quantiles).
    beta: Tuple[float, float, float] (default: (1.7, 3.5, 0.15))
        Range (start, stop, step) that the clean EEG distribution's shape parameter beta may take.

    Returns
    -------
    Mu : float
        estimated mean of the clean EEG distribution

    Sigma : float
        estimated standard deviation of the clean EEG distribution

    Alpha : float
        estimated scale parameter of the generalized Gaussian clean EEG distribution (optional)

    Beta : float
        estimated shape parameter of the generalized Gaussian clean EEG distribution (optional)

    """
    pass