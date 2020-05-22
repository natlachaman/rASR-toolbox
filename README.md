## rASR-toolbox
This is a python implementation of rASR-toolbox written in Matlab. 

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


### List of python-to-matlab mapping of scripts

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

