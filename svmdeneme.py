import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
X, sample_rate = librosa.load('/Users/cemergin/Downloads/emo-DB/wav/03a01Nc.wav')
# Get features
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)  # 40 values
#zcr = np.mean(librosa.feature.zero_crossing_rate)
chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, # tonal centroid features
                      axis=0)


print("mfccs: ",mfccs)
print("chroma: ",chroma)
print("contrast: ",contrast)
print("tonnetz: ",tonnetz)
