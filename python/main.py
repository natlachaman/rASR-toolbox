import os
from mne.io.eeglab.eeglab import read_raw_eeglab

from .clean_artifacts import clean_artifacts


class EEG:
    def __int__(self, filepath=""):
        self._filepath = filepath
        if ~os.path.exists(filepath):
            self._filepath = os.path.join("data", "output", "sme_1_1.xdf_filt.set")

    def load(self, preload=True):
        return read_raw_eeglab(self._filepath, preload=preload)

    def clean(self):
        return clean_artifacts()

    def visualize(self):
        pass
