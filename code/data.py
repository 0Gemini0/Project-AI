import numpy as np
import globals
import os
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, lombscargle
import pickle
import random as rand
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
from sklearn.decomposition import FastICA


class Data:
    """Class for handling and pre processing various types of EEG data"""

    def __init__(self, type,
                 load_pkl=False, preprocess=True,
                 notch_freq=50.0, high_freq=1.0, low_freq=43.0, sampl_rate=128.0, quality=15.0,
                 blur_size=150, ch_outlier_dev = 4.0, gy_outlier_dev=2.0,
                 sup_path = globals.SUPER_PATH, uns_path=globals.UNSUP_PATH, fak_path=globals.FAKE_PATH):

        self.type = type
        self.preprocess = preprocess

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

        # motor noise removal settings
        self.blur_size = blur_size
        self.ch_outlier_dev = ch_outlier_dev
        self.gy_outlier_dev = gy_outlier_dev

        # ica settings
        self.loc = globals.LOC

        if load_pkl:
            # this class can be used to load data from a pickle frame
            self._load_pkl()
            if self.preprocess:
                self._preprocess()
            else:
                self.channels_interpolated = np.copy(self.channels)
        else:
            # standard usage, load the data from csv files in a folder and apply simple pre-processing
            self._load()
            if self.preprocess:
                self._preprocess()
            else:
                self.channels_interpolated = np.copy(self.channels)

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

        # load the pandas dataframes
        if self.type == "unsupervised":
            self._load_unsupervised()
        elif self.type == "supervised":
            self._load_supervised()
        else:
            self._load_fake()

        # get gyro head movement information
        self._get_gyro()

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

        # enable interactive plots
        # plt.ion()

        # apply a notch filter and high pass filter to the data, plot the frequency of a random plot as test
        print("commence filtering...")
        index = rand.randint(0, len(self.dataset) - 1)
        self._lomb_scargle(index, 0)
        print("apply high pass filter...")
        self._high_pass()
        print("apply line noise filter...")
        self._line_notch()
        print("apply low pass filter...")
        self._low_pass()
        self._lomb_scargle(index, 0)
        print("filtering complete")

        # remove some motor noise
        print("removing motor noise...")
        channel = self.channels[index][:, 0]
        start = 0
        end = len(channel)
        self._visualize_range(channel, np.mean(channel), np.std(channel), start, end)
        self._motor_noise_removal()
        channel = self.channels[index][:, 0]
        self._visualize_range(channel, np.mean(channel), np.std(channel), start, end)
        self._lomb_scargle(index, 0)
        print("motor noise removal complete")
        print("please close/save all plots now...")
        plt.show()

        # select ICA components to remove
        print("commence ica analysis")
        self._ica()

    def _load_unsupervised(self):
        dataset = []
        names = []

        # first load files taken with the epoc (if exists)
        print("loading old data")
        try:
            for file in os.listdir(self.uns_path + "old\\"):
                if file.endswith(".CSV"):
                    names.append(file)
                    data = pd.read_csv(self.uns_path + "old\\" + file)
                    dataset.append(data)
        except FileNotFoundError:
            print("No epoc files, or files in wrong folder (make sure the folder is old, within current data path)")

        # then load files taken with the epoc+ (different format)
        db1 = 0
        db2 = 0
        i = -1
        print("loading new data")
        for file in os.listdir(self.uns_path):
            if file.endswith(".md.csv"):
                i += 1
                db2 = pd.read_csv(self.uns_path + file, skiprows=[0])
                header = {}
                f = open(self.uns_path + file, 'r')
                for h in f.readline().split(','):
                    hh = h.split(':')
                    header[hh[0].replace(' ', '')] = hh[1].split()
                f.close()
                db2.columns = header['labels']
                db2.TIME_STAMP_ms += db2.TIME_STAMP_s * 1000
            elif file.endswith(".csv"):
                i += 1
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

            if i % 2 == 1 and not file.endswith('old'):
                data = pd.merge(db1, db2, on='TIME_STAMP_ms', how='left', suffixes=['_eeg', '_md'])
                header['labels'] = list(data.columns)
                dataset.append(data.ffill().bfill())

        # store the data, channels array and filenames
        self.names = names
        self.dataset = dataset
        self.channels = []

        for data in self.dataset:
            try:
                self.channels.append(data.loc[:, ' AF3':' AF4'].values)
            except KeyError:
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

    def _get_gyro(self):
        """Get X,Y,Z axis gyro information and combine it into one movement magnitude channel."""
        self.gyro = []
        for j, dat in enumerate(self.dataset):
            try:
                # new data (epoc+)
                gyrox = np.asarray(dat["GYROX_md"])
                gyroy = np.asarray(dat["GYROY_md"])
                gyroz = np.asarray(dat["GYROZ"])
                gyro = np.abs(gyrox - np.mean(gyrox)) + np.abs(gyroy - np.mean(gyroy)) + np.abs(gyroz - np.mean(gyroz))
            except KeyError:
                # old data (epoc)
                gyrox = dat.loc[:, " GYROX"].values
                gyroy = dat.loc[:, " GYROY"].values
                gyro = np.abs(gyrox - np.mean(gyrox)) + np.abs(gyroy - np.mean(gyroy))

            self.gyro.append(gyro)
        print("gyro data collection complete")

    def _line_notch(self):
        """notch filter at specified frequency hz (standard = 50)"""
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
        """low pass filter at specified hz (standard = 43, the epoc effective range)"""
        fs = self.sampl_rate
        f0 = self.low_freq / globals.FREQ_CORRECTION
        w0 = f0 / (fs / 2)
        b, a = butter(4, w0)  # 8th order butter filter for steep roll off (48db/octave)
        for j, data in enumerate(self.channels):
            for i in range(data.shape[1]):
                data[:, i] = filtfilt(b, a, data[:, i])
            self.channels[j] = data

    def _high_pass(self):
        """low pass filter at specified hz (standard = 1)"""
        fs = self.sampl_rate
        f0 = self.high_freq / globals.FREQ_CORRECTION
        w0 = f0 / (fs / 2)
        b, a = butter(4, w0, "highpass")  # 8th order butter filter for steep roll off (48db/octave)
        for j, data in enumerate(self.channels):
            for i in range(data.shape[1]):
                data[:, i] = filtfilt(b, a, data[:, i])
            self.channels[j] = data

    def _lomb_scargle(self, index, index2):
        """Produce a simple lomb scargle frequency density plot"""
        wave = self.channels[index][:, index2] - np.mean(self.channels[index][:, index2])
        wave_len = wave.shape[0]
        steps = np.linspace(0.0, wave_len / self.sampl_rate, wave_len)
        freqs = np.linspace(0.01, 70.0, 140)
        pgram = lombscargle(steps, wave, freqs)
        plt.figure(figsize=[15, 8])
        plt.plot(freqs, np.sqrt(4 * (pgram / wave_len)), "b")
        # plt.show()

    def _motor_noise_removal(self):
        self.mask = []
        self.indices = []
        for j, channel in enumerate(self.channels):
            # get the gyro channel and statistics
            gyro = self.gyro[j]
            gyro_mean = np.mean(gyro)
            gyro_stdev = np.std(gyro, dtype="float64")

            # center channel mean around zero and get channel stdev
            channel = channel - np.mean(channel, axis=0)
            stdev = np.std(channel, dtype="float64", axis=0)

            # create mask based on spurious head movement
            mask = gyro - self.gy_outlier_dev * gyro_stdev - gyro_mean
            mask[mask > 0] = 0.0
            mask[mask < 0] = 1.0

            # create mask based on outliers in the channels
            mask2 = np.abs(channel) - self.ch_outlier_dev * stdev
            mask2[mask2 > 0] = 0.0
            mask2[mask2 < 0] = 1.0

            # combine masks
            mask2 = np.prod(mask2, axis=1)
            mask = mask * mask2

            # blur mask to filter out peaks
            box_filter = np.ones([self.blur_size, ]) / self.blur_size
            mask = np.convolve(mask, box_filter, 'same')
            mask[mask < 1.0] = 0.0

            # get the mask indices by first and last index of region to remove
            indices = np.where(mask == 0.0)[0]
            indices = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
            indices = [[i[0], i[-1]] for i in indices]
            print("percentage of data retained: ", sum(mask) / len(mask))

            # apply mask to data
            for chunk in indices:
                chk_mean = (channel[chunk[0]] + channel[chunk[1]]) / 2
                channel[chunk[0]:chunk[1]] = chk_mean

            self.channels[j] = channel
            self.mask.append(mask)
            self.indices.append(indices)

    def _ica(self):
        self.channels_interpolated = []
        for j, channel in enumerate(self.channels):
            # retain pre ica channel with interpolated noise removed data.
            self.channels_interpolated.append(np.copy(channel))
            channel = channel[self.mask[j] > 0]

            # apply ica to masked channel
            ica = FastICA(n_components=14, max_iter=10000, whiten=True)
            S = ica.fit_transform(channel)  # Reconstruct signals
            A = ica.mixing_

            # lomb scargle prelims
            normval = S.shape[0]
            steps = np.linspace(0.0, normval / self.sampl_rate, normval)
            freqs = np.linspace(0.01, 70.0, 140)

            # plot for every IC separately, reject components based on informative plots
            for i in range(channel.shape[1]):
                # lomb scargle frequency plot
                plt.figure(figsize=[15, 8])
                plt.title("frequency range of independent component: " + str(i))
                pgram = lombscargle(steps, S[:, i], freqs)
                plt.plot(freqs, np.sqrt(4 * (pgram / normval)), "r")
                plt.pause(0.0001)

                # plot of S over a small range, twice
                plt.figure(figsize=[18, 8])
                plt.title("small section of activity of independent component: " + str(i))
                plt.plot(S[20000:21000, i])

                plt.figure(figsize=[18, 8])
                plt.title("small section of activity of channels")
                plt.plot(channel[20000:21000])

                plt.figure(figsize=[18, 8])
                plt.title("small section of activity 2 of independent component: " + str(i))
                plt.plot(S[40000:41000, i])
                plt.pause(0.0001)

                plt.figure(figsize=[18, 8])
                plt.title("small section of activity 2 of channels")
                plt.plot(S[40000:41000, i])
                plt.pause(0.0001)

                plt.draw_all()

                # scalp topomap
                plt.figure(figsize=[8, 8])
                plt.title("scalp activity of independent component: " + str(i))
                plot_topomap(A[:, i], self.loc)

                while True:
                    remove = input("Remove this component? y/n: ")
                    if remove == 'y':
                        print("component removed")
                        A[:, i] = 0
                        break
                    elif remove == 'n':
                        print("component retained")
                        break
                    else:
                        print("invalid input...")

            # replace the channels with ICA corrected data, and plot the effect of the correction
            self.channels[j] = (np.dot(A, S.T)).T
            plt.figure(figsize=[18, 8])
            plt.title("small section of activity of channels, after ica")
            plt.plot(self.channels[j][20000:21000])
            plt.show()

            # replace non-interpolated parts of the interpolated channels with ICA corrected data
            self.channels_interpolated[j][self.mask[j] > 0] = self.channels[j]

    @staticmethod
    def _visualize_range(wave, mean, stdev, start, end):
        xrange = len(wave[start:end])
        plt.figure(figsize=[28, 8])
        plt.plot([0, xrange], [mean, mean], 'b')
        plt.plot([0, xrange], [mean + stdev, mean + stdev], 'g')
        plt.plot([0, xrange], [mean - stdev, mean - stdev], 'g')
        plt.plot([0, xrange], [mean + 2 * stdev, mean + 2 * stdev], 'g')
        plt.plot([0, xrange], [mean - 2 * stdev, mean - 2 * stdev], 'g')
        plt.plot([0, xrange], [mean + 3 * stdev, mean + 3 * stdev], 'g')
        plt.plot([0, xrange], [mean - 3 * stdev, mean - 3 * stdev], 'g')
        plt.plot(wave[start:end], 'r')
        # plt.show()

    def save_csv(self, filename_old, filename_new):
        f_new = open(filename_new, 'a')
        f_old = open(filename_old, 'a')
        i = 0
        j = 0
        for k, data in enumerate(self.dataset):
            try:
                # prepare new (epoc+) data for storage
                check = data["GYROX_md"]
                data.loc[:,"AF3":"AF4"] = self.channels_interpolated[k]
                if self.preprocess:
                    data['mask'] = self.mask[k]

                # append to csv file
                if i == 0:
                    data.to_csv(f_new, index=False)
                else:
                    data.to_csv(f_new, index=False, header=False)
                i += 1
            except KeyError:
                # prepare old (epoc) data for storage
                data.loc[:," AF3":" AF4"] = self.channels_interpolated[k]
                if self.preprocess:
                    data['mask'] = self.mask[k]

                # append to csv file
                if j == 0:
                    data.to_csv(f_old, index=False)
                else:
                    data.to_csv(f_old, index=False, header=False)
                j += 1

    def save_csv_separate(self, folder, tag):
        for k, data in enumerate(self.dataset):
            try:
                # stupid, checks if this a new file (epoc+)
                check = data["GYROX_md"]

                # replace channels with (potentially preprocessed)
                data.loc[:, "AF3":"AF4"] = self.channels_interpolated[k]
                if self.preprocess:
                    data['mask'] = self.mask[k]

                # store new file separate csv file
                f_new = open(folder + self.names[k][:-4] + tag + self.names[k][-4:], "w")
                data.to_csv(f_new, index=False)
            except KeyError:
                # otherwise handle as old file (epoc)
                data.loc[:, " AF3":" AF4"] = self.channels_interpolated[k]
                if self.preprocess:
                    data['mask'] = self.mask[k]

                f_old = open(folder + self.names[k][:-4] + tag + self.names[k][-4:], "w")
                data.to_csv(f_old, index=False)

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






