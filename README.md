# rASR-toolbox

```
|-- matlab
|   |-- asr_calibrate.m                 -- 
|   |-- asr_process.m                   -- 
|   |-- clean_artifacts.m               -- 
|   |-- clean_asr.m                     -- 
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
