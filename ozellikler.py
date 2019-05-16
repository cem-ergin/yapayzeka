import os
import sys
from typing import Tuple

import numpy
import numpy as np
import scipy.io.wavfile as wav
import librosa
import librosa.display
import matplotlib.pyplot as plt
#CLASS_LABELS = ("Sinirli", "Korku", "Mutlu", "Normal", "Uzuntulu", "Igrenmis")
#NUM_LABELS = len(CLASS_LABELS)

def Radar(data_prob, class_labels: Tuple, num_classes: int):

    angles = np.linspace(0, 2 * np.pi, num_classes, endpoint = False)
    data = np.concatenate((data_prob, [data_prob[0]]))  # 闭合
    angles = np.concatenate((angles, [angles[0]]))  # 闭合

    fig = plt.figure()

    # polar参数
    ax = fig.add_subplot(111, polar = True)
    ax.plot(angles, data, 'bo-', linewidth=2)
    ax.fill(angles, data, facecolor='r', alpha=0.25)
    ax.set_thetagrids(angles * 180 / np.pi, class_labels)
    ax.set_title("Emotion Recognition", va = 'bottom')

    # 设置雷达图的数据最大值
    ax.set_rlim(0, 1)

    ax.grid(True)
    # plt.ion()
    plt.show()
    # plt.pause(4)
    # plt.close()


'''
Waveform(): 
    音频波形图

输入:
    file_path(str): 音频路径
'''

def Waveform(file_path: str):
    data, sampling_rate = librosa.load(file_path)
    plt.figure(figsize=(15, 5))
    librosa.display.waveplot(data, sr=sampling_rate)
    plt.show()

'''
Spectrogram(): 
    频谱图

输入:
    file_path(str): 音频路径
'''
def Spectrogram(file_path: str):
    # sr: 采样率
    # x: 音频数据的numpy数组
    sr,x = wav.read(file_path)

    # step: 10ms, window: 30ms
    nstep = int(sr * 0.01)
    nwin  = int(sr * 0.03)
    nfft = nwin
    window = np.hamming(nwin)

    nn = range(nwin, len(x), nstep)
    X = np.zeros( (len(nn), nfft//2) )

    for i,n in enumerate(nn):
        xseg = x[n-nwin:n]
        z = np.fft.fft(window * xseg, nfft)
        X[i,:] = np.log(np.abs(z[:nfft//2]))

    plt.imshow(X.T, interpolation='nearest', origin='lower', aspect='auto')
    plt.show()


