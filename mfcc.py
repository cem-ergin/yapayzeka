import librosa
import numpy as np

filepath = '/Users/cemergin/Downloads/emo-DB/wav/11b09Wa.wav'
data, sampling_rate = librosa.load(filepath)
mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=39).T, axis=0)
print(mfccs)