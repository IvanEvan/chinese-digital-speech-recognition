#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Evan
import os
import glob
from torch.utils import data
import numpy as np
import torch
import utils.dataPreProcess as dataPreProcess
import random
import librosa


class SpeechData(data.Dataset):

    def __init__(self, dataList):
        self.data = []
        for dt in dataList:
            self.data.append(dt)

    def __getitem__(self, index):
        (dataPath, label) = self.data[index]
        feature = Featurize(dataPath)

        return feature, torch.tensor(int(label))

    def __len__(self):
        """
        返回数据集中所有数据个数
        """
        return len(self.data)


def Featurize(path):
    """

    :param path:
    :return:
    """
    paddedLength = 99000
    y, sr = librosa.load(path, sr=16000, mono=True, res_type='kaiser_best')

    try:
        y = dataPreProcess.padding(y, paddedLength)

    except:
        pass

    feat = dataPreProcess.stft(y, hop_length=512, n_fft=2048)
    # 得到功率谱
    feat = np.abs(feat) ** 2
    # 做梅尔滤波,并且取对数，然后提取能量谱
    feat = dataPreProcess.melspectrogram(Spec=feat, sigRate=sr, n_mels=128)
    # 均一化
    feat = (feat - np.mean(feat)) / (np.sqrt(np.sum(feat ** 2) + 0.0000001))
    feat = feat.reshape(1, feat.shape[0], -1)
    # 转为tensor
    feat = torch.from_numpy(feat).type(torch.float)

    return feat


def featurize_from_nparray(array_data):
    """

    :param path:
    :return:
    """
    paddedLength = 99000
    sr = 16000
    # y, sr = librosa.load(path, sr=16000, mono=True, res_type='kaiser_best')
    y = array_data
    try:
        y = dataPreProcess.padding(array_data, paddedLength)

    except:
        pass

    feat = dataPreProcess.stft(y, hop_length=512, n_fft=2048)
    # 得到功率谱
    feat = np.abs(feat) ** 2
    # 做梅尔滤波,并且取对数，然后提取能量谱
    feat = dataPreProcess.melspectrogram(Spec=feat, sigRate=sr, n_mels=128)
    # 均一化
    feat = (feat - np.mean(feat)) / (np.sqrt(np.sum(feat ** 2) + 0.0000001))
    feat = feat.reshape(1, feat.shape[0], -1)
    # 转为tensor
    feat = torch.from_numpy(feat).type(torch.float)

    return feat


def getFolder(root_dir='D:\\数字\\train_data', fold_n=10):
    """
    return: a list with elements as a list
    """

    wav_gener = glob.iglob(os.path.join(root_dir, '*.wav'))

    speech_data = []
    for i in wav_gener:
        label = str(os.path.basename(i).split('_')[0]).split('.wav')[0]
        speech_data.append((i, label))

    random.shuffle(speech_data)
    total_len = len(speech_data)
    each_folder_size = total_len // fold_n
    folder = [speech_data[i:i + each_folder_size] for i in range(0, total_len, each_folder_size)]
    return folder


def loadDataset(folder, partition, batch_size, fold_n=10):
    """

    :param partition:
    :param batch_size:
    :param train:
    :param fold_n: 10 默认十折交叉验证
    :return:
    """
    from itertools import chain

    trainFolder = folder[:partition] + folder[partition + 1:]
    trainFolder = list(chain(*trainFolder))

    trainLoader = torch.utils.data.DataLoader(SpeechData(trainFolder), shuffle=True, batch_size=batch_size,
                                              pin_memory=True, num_workers=32)
    validFolder = folder[partition:partition + 1]
    validFolder = list(chain(*validFolder))
    validLoader = torch.utils.data.DataLoader(SpeechData(validFolder), shuffle=True, batch_size=batch_size,
                                              pin_memory=True, num_workers=32)

    return trainLoader, validLoader


def dynamic_vad(vid_path, sr=16000):
    wa, sr_ret = librosa.load(vid_path, sr=sr, mono=True)  # 对输入音频读取为单声道，采样平率16000KHz/s
    assert sr_ret == sr

    sig_energy = np.mean((wa ** 2))
    sig_db = librosa.power_to_db(sig_energy)
    # adapt_db = np.abs(sig_db) * 0.75
    adapt_db = np.abs(sig_db) * 0.7
    intervals = librosa.effects.split(wa, top_db=adapt_db)  # 去静音，对响度低于20db的部分
    # wav_output = []
    # for sliced in intervals:
    #     wav_output.extend(wav[sliced[0]:sliced[1]])
    # wav_output = np.array(wav_output)

    # return wav, wav_output
    return wa, intervals


if __name__ == '__main__':
    feat = Featurize('xx.wav')
