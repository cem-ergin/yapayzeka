import librosa
import numpy as np

#y, sr = librosa.load('/Users/cemergin/Downloads/emo-DB/wav/03b10Wb.wav', offset=30, duration=5)
y, sr = librosa.load('/Users/cemergin/Downloads/emo-DB/wav/03b10Wb.wav')
#array = librosa.feature.mfcc(y=y, sr=sr)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
array = librosa.feature.mfcc(S=librosa.power_to_db(S))

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
mfccs = np.mean(mfccs.T,axis=0)
file1 = open("mfcc.txt","a")
file1.write(str(librosa.feature.mfcc(y=y,sr=sr)))
file1.close

for i in mfccs:
    print(i)

