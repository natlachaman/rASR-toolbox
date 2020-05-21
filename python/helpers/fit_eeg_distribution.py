from typing import Tuple
import numpy as np
from scipy.special import gammaincinv, gamma

from .utils import _histc, _kl_divergence
np.seterr(divide='ignore', invalid='ignore')

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
    # sort data so we can access quantiles directly
    X = np.sort(X)
    n = len(X)
    beta = np.arange(beta[0], beta[1], step=beta[2])
    quants = np.array(quants)

    # calc z bounds for the truncated standard generalized Gaussian pdf and pdf rescaler
    zbounds = np.zeros((len(beta), len(quants)))
    rescale = np.zeros_like(beta)
    for b in range(len(beta)):
        zbounds[b] = np.sign(quants - 1 / 2) * gammaincinv(np.sign(quants - 1 / 2) * (2 * quants - 1),
                                                           1 / beta[b]) ** (1 / beta[b])
        rescale[b] = beta[b] / (2 * gamma(1 / beta[b]))

    # determine the quantile-dependent limits for the grid search
    lower_min = np.min(quants)                     # we can generally skip the tail below the lower quantile
    max_width = np.diff(quants)                    # maximum width is the fit interval if all data is clean
    min_width = min_clean_fraction * max_width     # minimum width of the fit interval, as fraction of data

    # get matrix of shifted data ranges
    a = np.arange(int(np.round(n * max_width)))
    b = np.round(np.arange(lower_min, lower_min + max_dropout_fraction, step=step_sizes[0]) * n)
    idx = (a[np.newaxis, :] + b[:, np.newaxis]).astype(int)
    X = X[idx]
    X1 = X[:, 0]
    X -= X1[:, np.newaxis]

    # for each interval width...
    opt_val = np.Inf
    opt_beta = 0
    opt_bounds = np.zeros((1, len(quants)))
    opt_lu = np.zeros((1, 2))

    for m in np.round(np.arange(min_width, max_width, step=step_sizes[1]) * n).astype(int)[::-1]:

        # scale and bin the data in the intervals
        nbins = int(np.round(3 * np.log2(1 + m / 2)))
        H = X[:, :m] * (np.divide(nbins, X[:, m]))[:, np.newaxis]
        q = _histc(H, nbins) + 0.01

        # for each shape value...
        for b in range(len(beta)):
            bounds = zbounds[b, :]

            # evaluate truncated generalized Gaussian pdf at bin centers
            x = bounds[0] + np.arange(0.5, nbins-0.5) / nbins * np.diff(bounds)
            p = np.exp(-np.abs(x) ** beta[b]) * rescale[b]
            p /= np.sum(p)

            # calc KL divergences
            kl = _kl_divergence(p, q)

            # update optimal parameters
            min_val, idx = np.min(kl), np.argmin(kl)
            if min_val < opt_val:
                opt_val = min_val
                opt_beta = beta[b]
                opt_bounds = bounds
                opt_lu = np.r_[X1[idx], X1[idx] + X[idx, m]]

    # recover distribution parameters at optimum
    alpha = (opt_lu[1] - opt_lu[0]) / np.diff(opt_bounds)
    mu = opt_lu[0] - opt_bounds[0] * alpha
    beta = opt_beta

    # calculate the distribution's standard deviation from alpha and beta
    sigma = np.sqrt((alpha ** 2) * gamma(3 / beta) / gamma(1 / beta))

    return mu, sigma, alpha, beta