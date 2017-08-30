"""this project is doing two things:
    1.read csv
    2.process the audio to mfcc"""

import os
import datetime
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import numpy as np
import pandas as pd
import csv

class Datas(object):

    def __init__(self, n_input, batch_size, epoch, data_path, filename):
        """keyword spotting for read csv and wav initial parameters
        Attributes:
            n_input: Number of MFCC features
            n_context: The number of frames in the context
            _num_batch_size: count batch size
            n_label: The number of characters in the target language plus one
            batch_size: input test batch_size
        """
        self._n_input = n_input
        self.global_batch_size = 0
        self._batch_size = batch_size
        self.wav_fn = []
        self.wav_labels = []
        self.batch_x = []
        self.batch_acc_x = []
        self.check_csv(data_path)
        self.read_csv(filename)

    def read_csv(self,filename):
        wav_label_0 = []
        wav_label_1 = []
        df_read = pd.read_csv(filename)
        self.wav_fn = df_read['wav_filename'].values.tolist()
        wav_label_0 = df_read['label_0'].values.tolist()
        wav_label_1 = df_read['label_1'].values.tolist()
        for label in range(len(self.wav_fn)):
            self.wav_labels.append([wav_label_0[label], wav_label_1[label]])

    def check_csv(self, path):
        csv_data = [('wav_filename', 'label_0', 'label_1')]  # title
        if os.path.isfile('record.csv'):
            pass
        else:
            for files in os.listdir(path):
                filename = path + files
                fl = files.split("_", 3)
                verify = fl[3]
                if verify == '1.wav':
                    csv_data.append((filename, '1', '0'))  # Keyword
                elif verify == '0.wav':
                    csv_data.append((filename, '0', '1'))  # Keyword
            f = open('record.csv', "w")  # 'w' means append
            w = csv.writer(f)
            w.writerows(csv_data)
            f.close()

    def dictionary(self, wav_name):
        """Given a WAV audio file at ``audio_filename``, calculates ``numcep`` MFCC features
        at every 0.01s time step with a window length of 0.025s. Appends ``numcontext``
        context frames to the left and right of each time step, and returns this data
        in a numpy array.
        """
        fs, audio = wav.read(wav_name)
        # Get mfcc coefficients
        orig_inputs = mfcc(audio, samplerate=fs, numcep=self._n_input)
        orig_inputs = (orig_inputs - np.mean(orig_inputs)) / np.std(orig_inputs)
        source = orig_inputs
        source = source.ravel()
        return source

    def set_datas(self, global_num):
        start_index = global_num * self._batch_size
        end_index = start_index + self._batch_size
        for i in range(start_index, end_index):
            self.batch_x.append(self.dictionary(self.wav_fn[i]))
            self.label_train = self.wav_labels[start_index: end_index]
        return self.batch_x, self.label_train

    def set_acc_data(self):
        for i in range(len(self.wav_fn)):
            self.batch_acc_x.append(self.dictionary(self.wav_fn[i]))
        return self.batch_acc_x, self.wav_labels
