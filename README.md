## rASR-toolbox
This is a Python implementation of rASR-toolbox written in Matlab. 

### How to run the code
To run the code you'll first need to install **Git, Python3.7 and Anaconda**. 
Installation steps for these tools are different for every OS. 

Once you have those installed, open the terminal and clone this repository by typing the following:
```bash
git clone https://github.com/natlachaman/rASR-toolbox.git
```

Next, create a conda environment to run the code on. Go to the root directory of rASR-toolbox, create a 
conda environment with the required dependencies, and activate it:
```bash
cd /path/to/rASR-toolbox/
conda env create --name rasr-toolbox-env --file requirements.txt python=3.7
conda activate rasr-toolbox-env
```
Finally, you can run the main routine of the toolbox by typing:
```bash
python python/main.py
```

### Main routine
The main class of this toolbox is the `EEG` class defined in `main.py`, that primarily **loads, cleans and 
visualizes** EEG data files (`.set`). The class can be used as follows:
```python
eeg = EEG(filepath="path/to/file.set")
eeg.clean()
eeg.visualize()
```
`EEG(filepath="path/to/file.set")` instantiates the class and loads the file to `EEG.data`. It also sets the
default values as found in input argument variables from `clean_artifact.m`. Parameters, if necessary, can be changed 
as follows:
```python
eeg = EEG(filepath="path/to/file.set")
eeg.flatline_criterion = 7
eeg.highpass = (.2, .8)
```
`EEG.clean_data()` executes the main cleaning routine `clean_artifact()` as done by the `clean_artifact.m` EEG plugin in Matlab.
`EEG.visualize()` offers a minimum working visualization functionality, that can easily be extended.


### Testing
Testing was done manually: running the same block of code in Python and Matlab, and comparing their output arrays visually. 
Whether the end results are exactly equal is uncertain, but if not it should be pretty close.
Ideally, we would have a set of tests for every function, even for the `scipy.sginal` ones to compare that 
implementations are comparable. However, that is very time consuming and time limitations did not allow it.

An great option would be to use [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/call-user-script-and-function-from-python.html).
When in a conda environment, [install MATLAB Engine API for Python in nondefault locations](https://www.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html)
Like so:
```bash
conda activate rASR-tollbox-env
cd {MATLAB ROOT FOLDER}/extern/engines/python # e.g., cd /usr/local/MATLAB/R2020a/extern/engines/python 
python setup.py install --prefix="{HOME FOLDER}/miniconda3/envs/rASR-toolbox-env" 
# e.g., python setup.py install --prefix="/home/nat/miniconda3/envs/rASR-toolbox-env"
```
**Note: admin permissions are necessary for this operation (in Windows, run terminal as "Run as priviledged administrator")**

Go back to the folder where your Matlab scripts live, open python and run, for instance:
```python
import matlab.engine
eng = matlab.engine.start_matlab()
output = eng.clean_artifacts(nargout=0) # run clean_artifacts.m
# use output to compare to python `clean_artifacts.py` output
```

### List of Python-to-Matlab mapping of scripts
```
|-- matlab
|   |-- asr_calibrate.m                 -- python/asr_calibrate.py
|   |-- asr_process.m                   -- python/asr_process.py
|   |-- clean_artifacts.m               -- python/clean_artifacts.py
|   |-- clean_asr.m                     -- python/clean_asr.py
|   |-- clean_channels.m                -- python/clean_channels.py
|   |-- clean_channels_nolocs.m         -- python/clean_channels_nolocs.py
|   |-- clean_drifts.m                  -- python/clean_drifts.py
|   |-- clean_flatlines.m               -- python/clean_flatlines.py
|   |-- rasr_nonlinear_eigenspace.m     -- python/rasr_nonlinear_eigenspace.py
|   |-- vis_artifacts.m                 -- 
|   |-- clean_windows.m                 -- python/clean_windows.py
|   |-- eegplugin_clean_rawdata.m       -- does not apply in python (EEGLab plugin)
|   |-- pop_clean_rawdata.m             -- does not apply in python (EEGLab plugin)
|   |-- clean_rawdata.m                 -- does not apply in python (EEGLab plugin)
|   |-- (load, clean, viz)              -- python/main.py
|   |-- private
|   |   |-- block_geometric_median.m        -- python/helpers/block_geometric_median.py
|   |   |-- design_fir.m                    -- python/helpers/design_fir.py
|   |   |-- design_kaiser.m                 -- python/helpers/design_kaiser.py
|   |   |-- filter_fast.m                   -- scipy.signal.lfilter() 
|   |   |-- filtfilt_fast.m                 -- scipy.signal.filtfilt() 
|   |   |-- fit_eeg_distribution.m          -- python/helpers/fit_egg_distribution.py
|   |   |-- geometric_median.m              -- python/helpers/geometric_median.py
|   |   |-- sphericalSplineInterpolate.m    -- mne.channels.interpolation._make_interpolation_matrix()
|   |   |-- window_func.m                   -- python/helpers/window_func.py
|   |   |-- hlp_handleerror.m               -- python/helpers/decorators.py 
|   |   |-- hlp_microcache.m                -- @functools.lr_cache
|   |   |-- hlp_split.m                     -- str.split()
|   |   |-- hlp_varargin2struct.m           -- does not apply in python
|   |   |-- findjobj.m                      -- does not apply in python
|   |   |-- hlp_memfree.m                   -- does not apply in python
|   |   |-- (yukewalk() from SigProTool)    -- python/helpers/yukewalk.py
|   |   |-- (positive_definite_karcher_...  -- python/helpers/positive_definite_karcher_means.py
            means() from manpot)    
```

### List of known issues (or possible issues)

[1] In `asr_calibrate.py` line 94, `lfilter(B, A, X)` does not return the final state `zj` unless 
an initial state `zi` is passed. When doing so (by running `zi = lfilter_zi(B, A)`), `lfilter()` throws an error.
Parameter `zj` is later on used in `asr_process.py` in line 99 (stored as `irr`) as initial state. 
Since we cannot obtain it, we again don't user an initial state.
This is most probably altering the final output.

[2] In `rasr_nonlinear_eigenspace.py` line 84, `maxinner` is set to 4. This will affect the results to where it converges
and the time it takes to converge. Not sure what is the right tunning of this. Please check with Matlab's `manopt` 
implementation.

[3] Not sure if this is an issue, but part of the code (code related to `clean_asr.py`) was taken from 
[this repository](https://github.com/nbara/python-meegkit/blob/master/meegkit/asr.py). They have a BSD 3-Clause License
that allows for commercial use, modification, distribtuion and private use. So I don't think I'd be a problem to use it, 
but probably you'll need to mention it somewhere to avoid plausible legal issues.