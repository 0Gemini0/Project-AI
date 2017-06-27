import numpy as np
import globals
import os
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, lombscargle
import pickle
import random as rand
import matplotlib.pyplot as plt


class Data:
    """Class for handling various types of EEG data"""

    def __init__(self, type,
                 load_pkl=False, preprocess=True,
                 notch_freq=50.0, high_freq=1.0, low_freq=30.0, sampl_rate=128.0, quality=15.0,
                 sup_path = globals.SUPER_PATH, uns_path=globals.UNSUP_PATH, fak_path=globals.FAKE_PATH):

        self.type = type

        # data folder paths
        self.sup_path = sup_path
        self.uns_path = uns_path
        self.fak_path = fak_path
        self.pkl_path = globals.PKL_PATH
        self.signals = globals.FAKE_SIGNALS

        # filter settings
        self.notch_freq = notch_freq
        self.high_freq = high_freq
        self.low_freq = low_freq
        self.sampl_rate = sampl_rate
        self.quality = quality

        if load_pkl:
            # this class can be used to load data from a pickle frame
            self._load_pkl()
            if preprocess:
                self._preprocess()
        else:
            # standard usage, load the data from csv files in a folder and apply simple pre-processing
            self._load()
            if preprocess:
                self._preprocess()

    def get_dataset(self, name):
        """Return a dataset by filename, ending with the csv tag"""
        index = self.names.index(name)
        return self.dataset[index]

    def get_channels(self, name):
        """Return a channel array by name or all channel sets as a big array"""
        if name == "all":
            return np.concatenate(self.channels, axis=0)
        else:
            index = self.names.index(name)
            return self.channels[index]

    def _load(self):
        """Load"""
        if self.type == "unsupervised":
            self._load_unsupervised()
        elif self.type == "supervised":
            self._load_supervised()
        else:
            self._load_fake()

    def _load_pkl(self):
        """Load pickled dataframe"""
        if self.type is not "fake":
            self.dataset = pickle.load(self.pkl_path + self.type + ".pkl")
            self.channels = []
            for data in self.dataset:
                self.channels.append(data.loc[:, 'AF3':'AF4'].values)
        else:
            print("this operation is not currently supported for the chosen data type")

    def _preprocess(self):
        """Basic, automatic data preprocessing"""

        # apply a notch filter and high pass filter to the data, plot the frequency of a random plot as test
        index = rand.randint(0, len(self.dataset) - 1)
        self._lomb_scargle(index)
        self._high_pass()
        self._line_notch()
        self._lomb_scargle(index)

        # remove some motor noise
        self._motor_noise_removal()

    def _load_unsupervised(self):
        dataset = []
        names = []

        # first load files taken with the epoc
        for file in os.listdir(self.uns_path + "old\\"):
            if file.endswith(".CSV"):
                names.append(file)
                data = pd.read_csv(self.uns_path + "old\\" + file)
                print(data.shape)
                dataset.append(data)

        # then load files taken with the epoc+ (different format)
        for i, file in enumerate(os.listdir(self.uns_path)):
            if file.endswith(".md.CSV"):
                db2 = pd.read_csv(self.uns_path + file, skiprows=[0])
                header = {}
                f = open(self.uns_path + file, 'r')
                for h in f.readline().split(','):
                    hh = h.split(':')
                    header[hh[0].replace(' ', '')] = hh[1].split()
                f.close()
                db2.columns = header['labels']
                db2.TIME_STAMP_ms += db2.TIME_STAMP_s * 1000
            elif file.endswith(".CSV"):
                names.append(file)
                db1 = pd.read_csv(self.uns_path + file, skiprows=[0])
                header = {}
                f = open(self.uns_path + file, 'r')
                for h in f.readline().split(','):
                    hh = h.split(':')
                    header[hh[0].replace(' ', '')] = hh[1].split()
                f.close()
                db1.columns = header['labels']
                db1.TIME_STAMP_ms += db1.TIME_STAMP_s * 1000

            if i % 2 == 1:
                data = pd.merge(db1, db2, on='TIME_STAMP_ms', how='left', suffixes=['_eeg', '_md'])
                header['labels'] = list(data.columns)
                dataset.append(data)

        # store the data, channels array and filenames
        self.names = names
        self.dataset = dataset
        self.channels = []
        for data in self.dataset:
            self.channels.append(data.loc[:, 'AF3':'AF4'].values)
        self.new_header = header

    def _load_supervised(self):
        # TODO
        self.dataset = []
        self.channels = []

    def _load_fake(self):
        with open(self.fak_path + "pure_signal_extraction_data_indices.mem", "rb") as input_file:
            indices = pickle.load(input_file)

        # we will get a sinusoid estimation of an eeg signal and a noisy variant
        shuffled_indices = []
        pure_data = []
        noisy_data = []

        for j in range(self.signals):
            # load X hours of data
            dat, shuffled_indices = self._random_signal_loader(indices, shuffled_indices,
                                                               self.fak_path + "pure_signal_extraction_data.mem")
            pure_data.extend(dat[1])
            noisy_data.extend(dat[0])

        # reshape into correctly size numpy arrays
        self.dataset = np.array(pure_data).T
        self.noisy_dataset = np.array(noisy_data).T

    def _line_notch(self):
        # notch filter at specified frequency hz (standard=50)
        fs = self.sampl_rate
        f0 = self.notch_freq / globals.FREQ_CORRECTION  # correct the frequency against scipy failure
        Q = self.quality
        w0 = f0 / (fs / 2)
        b, a = iirnotch(w0, Q)
        for j, data in enumerate(self.channels):
            for i in range(data.shape[1]):
                data[:, i] = filtfilt(b, a, data[:, i])
            self.channels[j] = data

    def _low_pass(self):
        # low pass filter at 30 hz
        fs = self.sampl_rate
        f0 = self.low_freq / globals.FREQ_CORRECTION
        w0 = f0 / (fs / 2)
        b, a = butter(4, w0)  # 8th order butter filter for steep roll off (48db/octave)
        for j, data in enumerate(self.channels):
            for i in range(data.shape[1]):
                data[:, i] = filtfilt(b, a, data[:, i])
            self.channels[j] = data

    def _high_pass(self):
        # high pass filter at 0.1 hz
        fs = self.sampl_rate
        f0 = self.high_freq / globals.FREQ_CORRECTION
        w0 = f0 / (fs / 2)
        b, a = butter(4, w0, "highpass")  # 8th order butter filter for steep roll off (48db/octave)
        for j, data in enumerate(self.channels):
            for i in range(data.shape[1]):
                data[:, i] = filtfilt(b, a, data[:, i])
            self.channels[j] = data

    def _lomb_scargle(self, index):
        """Produce a simple lomb scargle frequency density plot"""
        wave = self.channels[index]
        wave_len = wave.shape[0]
        steps = np.linspace(0.0, wave_len / self.sampl_rate, wave_len)
        freqs = np.linspace(0.01, 70.0, 140)
        pgram = lombscargle(steps, wave[:, 0], freqs)
        plt.figure()
        plt.plot(freqs, np.sqrt(4 * (pgram / wave_len)), "b")

    def _motor_noise_removal(self):
        for j, channel in enumerate(self.channels):
            # center mean around zero
            channel = channel - np.mean(channel, axis=0)

            # cast outliers to zero
            stdev = np.std(channel, dtype="float64", axis=0)
            mask = np.abs(channel) - 3*stdev
            mask[mask > 0] = 0.0
            mask[mask < 0] = 1.0
            channel = mask * channel

            # smooth drop-off of signals
            # this induces artificial peaks/drops in the data, need improved method
            for i in range(channel.shape[1]):
                for k in range(len(channel[:, i]) - 1):
                    if channel[k, i] - channel[k+1, i] > stdev / 10.0:
                     channel[k+1, i] = channel[k, i] - stdev / 10.0
                    elif channel[k, i] - channel[k + 1, i] < - stdev / 10.0:
                     channel[k + 1, i] = channel[k, i] + stdev / 10.0

            self.channels[j] = channel

    @staticmethod
    def _random_signal_loader(static_indices, shuffled_non_static_indices, file_path):
        """Random signal loader, for loading fake EEG data"""
        training_instance = False

        if (len(shuffled_non_static_indices) == 0):
            shuffled_non_static_indices = list(static_indices)
            rand.shuffle(shuffled_non_static_indices)
        with open(file_path, "rb") as input_file:
            input_file.seek(shuffled_non_static_indices.pop(0))
            training_instance = pickle.load(input_file)

        return training_instance, shuffled_non_static_indices

    @staticmethod
    def running_mean(x, N):
        """
        Get the running mean of an array x over N timepoints.
        Mirrors the beginning of the signal to provide a good running mean from the start.
        """
        cumsum = np.cumsum(np.insert(x, 0, (x[:N])[::-1]))
        return (cumsum[N:] - cumsum[:-N]) / N






