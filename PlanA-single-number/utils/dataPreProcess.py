# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Evan
import numpy as np
import scipy.fftpack as fft
import scipy
import scipy.ndimage
import scipy.signal
import scipy.interpolate


def padding(data, size, axis=-1, **kwargs):
    """
    pad a array so the the edge of the singal is zero

    :param data: np.ndarray
    :param size:
    :param axis:
    :param kwargs:
    :return:

    """
    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    padding_len = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (padding_len, int(size - n - padding_len))
    if padding_len < 0:
        raise Exception(('Target size ({:d}) must be '
                         'at least input size ({:d})').format(size, n))

    return np.pad(data, lengths, **kwargs)


def frame(signal, frame_length=1024, hop_length=512):
    from numpy.lib.stride_tricks import as_strided
    """

    #16000 * 0.032 = 512 帧长度为512 每帧长度为0.032

    :param signal: 
    :param frame_length: 帧长度
    :param hop_length: 不同帧之间的重叠的样本个数
    :return: 
    """

    # compute number of frames
    num_frames = 1 + int((len(signal) - frame_length) / hop_length)
    y_frames = as_strided(signal, shape=(frame_length, num_frames),
                          strides=(signal.itemsize, hop_length * signal.itemsize))

    return y_frames


def get_window(name='Hamming', N=1024):
    window = None
    if name == 'Hanning':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Hamming':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])

    # plt.plot(window)
    return window


def preEmphasis(audioSignal, preEmphCof):
    """

    预加重处理其实是将语音信号通过一个高通滤波器：

    :param audioSignal:
    :param preEmphCof: 0.97
    :return:
    """
    import numpy as np
    firstSignal = audioSignal[0]
    firstAfterSignal = audioSignal[1:]
    exceptFirstSignal = audioSignal[:-1]

    result = np.append(firstSignal, firstAfterSignal - preEmphCof * exceptFirstSignal)
    return result


def stft(signal, n_fft=1024, hop_length=None, win_length=1024, center=True,
         window='hann', pad_mode='reflect'):
    """

    短时快速傅里叶变换

    :param signal: [shape=(n,)] time series of input signal
    :param n_fft: FFT WINDOW SIZE
    :param hop_length: 即不同窗口之间样本的重叠个数 number audio of frames between STFT columns deafult: win_length/4
    :param win_length:  = nfft window will be of length 'win_length' and then padded with zeros to match n_fft
    :param center: signal padded in the center
    :param window:
    :param pad_mode:
    :return: np.ndarray[shape=(1+n_fft/2,t)]


    """
    signal = preEmphasis(signal, 0.97)

    if hop_length == None:
        hop_length = int(win_length // 4)

    fft_window = scipy.signal.get_window(window, win_length, fftbins=True)
    # fft_window = get_window(window,win_length)

    # 补 0
    fft_window = padding(fft_window, n_fft)

    # reshape he window so window can broadcast
    fft_window = fft_window.reshape((-1, 1))

    if center:
        signal = np.pad(signal, int(n_fft // 2), mode=pad_mode)

    # 分帧

    signal_frames = frame(signal, frame_length=n_fft, hop_length=hop_length)

    # 预分配内存 the STFT matrix

    stft_matrix = np.empty((int(1 + n_fft // 2), signal_frames.shape[1]), dtype=np.complex64, order='F')

    # how many columns can we fit?
    MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10  # 256MB

    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] * stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        # 加窗并且FFT
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])
        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window * signal_frames[:, bl_s:bl_t], axis=0)[:stft_matrix.shape[0]]

    return stft_matrix


def mel_to_hz(mels):
    mels = np.asanyarray(mels)
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def hz_to_mel(frequencies):
    frequencies = np.asanyarray(frequencies)

    return 2595 * np.log10(1 + frequencies / 700.0)


def mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0):
    # mel = 2595 * np.log10(1.0 + f/ 700.0)

    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels)


def mel(sigRate, n_fft, n_mels=128, fmin=0.0, fmax=None):
    """
    梅尔滤波

    :param sigRate:
    :param n_fft:
    :param n_mels: 滤波器个数
    :param fmin:
    :param fmax:
    :return:
    """

    # initialize
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))
    # 最高频率为采样频率一半
    fmax = float(sigRate) / 2

    # 计算每个FFT bin的中心频率
    fft_frequencies = np.linspace(0, float(sigRate) / 2, int(1 + n_fft // 2), endpoint=True)

    # 计算梅尔频率
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)

    # 实际上是返回这么一个vector
    # array([(mel_f[2]-mel_f[1])....(mel_f[i+1]-mel_f[i])])
    fdiff = np.diff(mel_f)

    # 梅尔标度中心频率与实际上的fft_frequency相减
    ramps = np.subtract.outer(mel_f, fft_frequencies)

    # 三角滤波
    for i in range(n_mels):
        # 这里实际上是ppt 的 (k-f(m-1)) / (f(m) - f(m-1))
        lower = - ramps[i] / fdiff[i]

        # (f(m+1)-k) / (f(m+1) - f(m))
        upper = ramps[i + 2] / fdiff[i + 1]

        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # slaney-style 归一化
    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights


def melspectrogram(sigRate=16000, Spec=None, n_fft=1024, hop_length=512, power=2.0, **kwargs):
    """

    :param sigRate:
    :param Spec:
    :param n_fft:
    :param hop_length:
    :param power:
    :param kwargs:
    :return:
    """
    n_fft = 2 * (Spec.shape[0] - 1)

    mel_basis = mel(sigRate, n_fft)

    # 梅尔滤波
    feat = np.dot(mel_basis, Spec)

    feat = np.asarray(feat)

    # scaling
    # 10 * log10 (10*feat / ref)

    magnitude = np.abs(feat)

    ref_value = np.max(magnitude)

    amin = 1e-10

    top_db = 80  # db: decibel (dB) units

    # 取对数
    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
    log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def mfcc(Spec, n_mfcc=16):
    return scipy.fftpack.dct(Spec, axis=0, type=2, norm='ortho')[:n_mfcc]
