import os
import logging
import matplotlib.pyplot as plt
from mne.io.eeglab.eeglab import read_raw_eeglab

from python.clean_artifacts import clean_artifacts


class EEG:
    def __init__(self, filepath=""):
        self._filepath = filepath
        if ~os.path.exists(filepath):
            logging.info("Couldn't find file. Loading .set file in data/output instead...")
            data_path = "/".join(os.getcwd().split("/")[:-1])
            self._filepath = os.path.join(data_path, "data", "output", "sme_1_1.xdf_filt.set")
        self.data = read_raw_eeglab(self._filepath, preload=True)
        self.channel_criterion = .85
        self.line_noise_criterion = 4
        self.burst_criterion = 5
        self.window_criterion = .25
        self.highpass = (.25,.75)
        self.channel_criterion_max_bad_time = .5
        self.burst_criterion_ref_max_bad_chns = 0.075
        self.burst_criterion_ref_tolerances = (-3.5, 5.5)
        self.window_criterion_tolerances = (-3.5, 7)
        self.flatline_criterion = 5
        self.nolocs_channel_criterion = .45
        self.noloc_channel_criterion_excluded = .1
        self.clean_data = None

    def clean(self):
        self.clean_data = clean_artifacts(signal=self.data,
                                           channel_criterion=self.channel_criterion,
                                           line_noise_criterion=self.line_noise_criterion,
                                           burst_criterion=self.burst_criterion,
                                           window_criterion=self.window_criterion,
                                           highpass=self.highpass,
                                           channel_criterion_max_bad_time=self.channel_criterion_max_bad_time,
                                           burst_criterion_ref_max_bad_chns=self.burst_criterion_ref_max_bad_chns,
                                           burst_criterion_ref_tolerances=self.burst_criterion_ref_tolerances,
                                           window_criterion_tolerances=self.window_criterion_tolerances,
                                           flatline_criterion=self.flatline_criterion,
                                           nolocs_channel_criterion=self.nolocs_channel_criterion,
                                           noloc_channel_criterion_excluded=self.noloc_channel_criterion_excluded)

    def visualize(self):
        # # simple visualization
        # C, S = self.data.shape
        # fig, ax = plt.subplots(C, 1)
        # ax = ax.ravel()
        #
        # # plot each channels separately
        # for c, ax_i in enumerate(ax):
        #     ax_i.plot(self.data[c, :], color="b")
        #     if self.clean_data is not None:
        #         ax_i.plot(self.clean_data[c, :], color="r")
        fig = self.clean_data.plot(bgcolor='w')


if __name__ == "__main__":
    logging.basicConfig(level=3)
    eeg = EEG()
    eeg.clean()
    eeg.visualize()