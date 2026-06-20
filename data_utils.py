import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list  #
    # ---------------------ASVspoof2019PA------------
    elif (is_eval):
        for line in l_meta:
            # key= line.strip()
            _, key, _, label, _ = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list


def pad(x, max_len=8192):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs  # label
        self.labels = labels  # 0、1
        self.base_dir = base_dir  # flac

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = 8192  # take ~4 sec audio (64600 samples)
        X_X_X = []
        X_X_X_X = []
        key = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + 'flac/' + key + '.flac', sr=16000)
        frame_20ms = int(fs * 0.02)  # 20ms
        frame_10ms = int(fs * 0.01)
        frame_512ms = int(fs * 0.512)
        frame_rms = librosa.feature.rms(y=X, frame_length=frame_20ms, hop_length=frame_10ms)[0]
        frame_zero = librosa.feature.zero_crossing_rate(y=X, frame_length=frame_20ms, hop_length=frame_10ms)[0]
        frame_rms_hamm = np.hamming(len(frame_rms)) * frame_rms
        frame_zero_hamm = np.hamming(len(frame_zero)) * frame_zero
        rms_yuzhi = 0.0025   # rms thr
        zero_yuzhi = 0.35  # cross_zero thr
        for i in range(len(frame_rms_hamm)):
            if ((frame_rms_hamm[i] < rms_yuzhi) and (frame_zero_hamm[i] > zero_yuzhi)):
                X_X_X.append(X[i * frame_20ms:(i + 1) * frame_20ms])
        for i in X_X_X:
            for j in i:
                X_X_X_X.append(j)
        if (len(X_X_X_X) >= frame_512ms):
            X_X_X_X = X_X_X_X[:frame_512ms]
        else:
            X_X_X_cha = frame_512ms - len(X_X_X_X)

            for i in X[-X_X_X_cha:]:
                X_X_X_X.append(i)
        X_X_X_X = np.array(X_X_X_X)
        # ------------------------------------------------------------------------------------
        X_pad = pad(X_X_X_X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs,labels, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''
        self.list_IDs = list_IDs  #
        self.base_dir = base_dir  #
        self.labels = labels  # 0,1

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = 8192  # take ~4 sec audio (64600 samples)
        X_X_X = []
        X_X_X_X = []
        key = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + 'flac/' + key + '.flac', sr=16000)
        frame_20ms = int(fs * 0.02)  # 20ms
        frame_10ms = int(fs * 0.01)
        frame_512ms = int(fs * 0.512)
        frame_rms = librosa.feature.rms(y=X, frame_length=frame_20ms, hop_length=frame_10ms)[0]
        frame_zero = librosa.feature.zero_crossing_rate(y=X, frame_length=frame_20ms, hop_length=frame_10ms)[0]
        frame_rms_hamm = np.hamming(len(frame_rms)) * frame_rms
        frame_zero_hamm = np.hamming(len(frame_zero)) * frame_zero
        rms_yuzhi = 0.0025
        zero_yuzhi = 0.35
        for i in range(len(frame_rms_hamm)):
            if ((frame_rms_hamm[i] < rms_yuzhi) and (frame_zero_hamm[i] > zero_yuzhi)):
                X_X_X.append(X[i * frame_20ms:(i + 1) * frame_20ms])
        for i in X_X_X:
            for j in i:
                X_X_X_X.append(j)
        if (len(X_X_X_X) >= frame_512ms):
            X_X_X_X = X_X_X_X[:frame_512ms]
        else:
            X_X_X_cha = frame_512ms - len(X_X_X_X)

            for i in X[-X_X_X_cha:]:
                X_X_X_X.append(i)
        X_X_X_X = np.array(X_X_X_X)
        # ------------------------------------------------------------------------------------
        X_pad = pad(X_X_X_X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, key, y



