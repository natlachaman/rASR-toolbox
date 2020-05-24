from typing import Tuple, Dict, Union, Any, Optional
import logging
import numpy as np
from scipy.signal import lfilter
from tqdm import tqdm

from rasr_nonlinear_eigenspace import nonlinear_eigenspace
from helpers.positive_definite_karcher_mean import positive_definite_karcher_mean


def asr_process(data: np.ndarray, srate: int, state: Dict[str, Optional[Any]], lookahead: Union[float, None],
                windowlen: float = 0.1,  stepsize: int = 4, maxdims: float = 1., maxmem: int = 256
                ) -> Tuple[np.ndarray, dict]:
    """ Processing function for the Artifact Subspace Reconstruction (ASR) method.

    This function is used to clean multi-channel signal using the ASR method. The required inputs are
    the data matrix, the sampling rate of the data, and the filter state (as initialized by
    asr_calibrate). If the data is used on successive chunks of data, the output state of the previous
    call to asr_process should be passed in.

    Parameters
    ----------
    data: np.ndarray
        Chunk of data to process [#channels x #samples]. This is a chunk of data, assumed to be
        a continuation of the data that was passed in during the last call to asr_process (if any).
        The data should be *zero-mean* (e.g., high-pass filtered the same way as for asr_calibrate).
    srate: int
        sampling rate of the data in Hz (e.g., 250)
    state: dict
         initial filter state (determined by asr_calibrate or from previous call to asr_process)
    lookahead: float
        Amount of look-ahead that the algorithm should use. Since the processing is causal, the output signal will
        be delayed by this amount. This value is in seconds and should be between 0 (no lookahead) and WindowLength/2
        (optimal lookahead). The recommended value is WindowLength/2.

        initial filter state (determined by asr_calibrate or from previous call to asr_process)
    windowlen: float (default: 0.1)
        Length of the statistcs window, in seconds (e.g., 0.5). This should not be much longer than the
        time scale over which artifacts persist, but the number of samples in the window should not be
        smaller than 1.5x the number of channels.
    stepsize: int (default: 4)
        The statistics will be updated every this many samples. The larger this is, the faster the algorithm will be.
        The value must not be larger than WindowLength*SamplingRate. The minimum value is 1 (update for every sample)
        while a good value is 1/3 of a second. Note that an update is always performed also on the first and last
        sample of the data chunk.
    maxdims: float (defautl: 1)
        Maximum dimensionality of artifacts to remove. Up to this many dimensions (or up to this fraction of dimensions)
        can be removed for a given data segment. If the algorithm needs to tolerate extreme artifacts a higher value
        than the default may be used (the maximum fraction is 1.0).
    maxmem: int (default: 256)
        The maximum amount of memory used by the algorithm when processing a long chunk with many channels, in MB.
        The recommended value is at least 256. Using smaller amounts of memory leads to longer running times.

    Returns
    -------
    data : np.ndarray
        cleaned data chunk (same length as input but delayed by LookAhead samples)
    state : dict
        final filter state (can be passed in for subsequent calls)

    """
    logging.info("Note: This is a Riemann adapted processing!")
    window_len = max(windowlen, 1.5 * len(data) / srate)
    lookahead = window_len / 2 if lookahead is None else lookahead

    if maxdims < 1:
        maxdims = int(np.round(len(data) * maxdims))

    if data.size == 0:
        return data, state

    C, S = data.shape
    # N = int(np.round(windowlen * srate))
    P = int(np.round(lookahead * srate))

    # initialize prior filter state by extrapolating available data into the past (if necessary)
    if "carry" not in state.keys():
        state["carry"] = np.tile(2 * data[:, 1], (P, 1)).T - data[:, 1 + (np.arange(P)[::-1] % S)]

    data = np.c_[state["carry"], data]
    data[~np.isfinite(data)] = 0

    # split up the total sample range into k chunks that will fit in memory
    splits = int(np.ceil((C * C * S * 8 * 8 + C * C * 8 * S /
                      stepsize + C * S * 8 * 2 + S * 8 * 5) /
                     (maxmem * 1024 * 1024 - C * C * P * 8 * 3)))

    if splits > 1:
        logging.info(f"Now cleaning data in {splits} blocks; this may take a while...")

    for i in tqdm(range(1, splits+1), total=splits):
        start = int(1 + np.floor((i-1) * S / splits))
        stop = np.minimum(S, int(np.floor(i * S / splits)))
        brange = np.arange(start, stop)

        if brange.size != 0:
            # get spectrally shaped data X for statistics computation (range shifted by lookahead) and also get a
            # subrange of the data (according to splits)
            # X, state["iir"] = lfilter(b=state["B"],
            #                           a=state["A"],
            #                           x=data[:, brange + P],
            #                           zi=state["iir"],
            #                           axis=1)
            X = lfilter(b=state["B"],
                        a=state["A"],
                        x=data[:, brange + P])

            # the Riemann version uses the sample covariance matrix here:
            SCM = 1 / X.shape[-1] * X @ X.T # channels x channels
            # if we have a previous covariance matrix, use it to compute the average to make the current covariance
            # matrix more stable
            if "cov" in state.keys():
                A = np.zeros((C, C, 2))
                A[:, :, 0] = SCM
                A[:, :, 1] = state["cov"]
                Xcov = positive_definite_karcher_mean(A)
            else:
                # we do not have a previous matrix to average, we use SCM as is
                Xcov = SCM

            # if there is no previous R (from the end of the last chunk), we estimate it right at the first sample
            update_at = np.minimum(np.arange(stepsize, (X.shape[-1] + stepsize - 1), step=stepsize), X.shape[-1])
            if "last_R" not in state.keys():
                update_at = np.r_[0, update_at]
                state["last_R"] = np.eye(C)

            V, D = nonlinear_eigenspace(Xcov, C) # np.diag()?
            V = np.real(V[:, np.argsort(D)])
            D = np.real(D[np.argsort(D)])

            # determine which components to keep (variance below directional threshold
            # or not admissible for rejection)
            keep = (D < np.sum(np.dot(state["T"], V) ** 2, axis=0)) | (np.arange(C) < (C - maxdims))

            # update the reconstruction matrix R (reconstruct artifact components using
            # the mixing matrix)
            if keep.all():
                R = np.eye(C)  # trivial case
            else:
                VT = np.dot(V.T, state["M"])
                demux = VT * keep[:, None]
                R = np.dot(np.dot(state["M"], np.linalg.pinv(demux)), V.T)

            # do the reconstruction in intervals of length stepsize (or shorter at the end of a chunk)
            last_keep = state["last_trivial"] if "last_trivial" in state else keep
            for j in range(len(update_at) - 1):
                last_n, n = update_at[j], update_at[j+1]
                # apply the reconstruction to intermediate samples (using raised-cosine blending)
                if ~keep.all() or ~last_keep.all():
                    subrange = np.arange(last_n, n).astype(int)
                    blend = (1 - np.cos(np.pi * np.arange(n - last_n)) / (n - last_n)) / 2
                    data[:, subrange] = blend * R.dot(data[:, subrange]) + \
                                       (1 - blend) * state['last_R'].dot(data[:, subrange])

            state["last_trivial"] = keep.all()
            state["last_R"] = R

    # carry the look-ahead portion of the data over to the state (for successive calls)
    state["carry"] = np.r_[state["carry"], data[:, -P:]]
    state["cov"] = Xcov

    return data[:, :P], state