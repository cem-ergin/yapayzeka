import csv

import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#dataset = pd.read_csv("/Users/cemergin/Documents/Book10.csv")
dataset = pd.read_csv("tesscembirlesik.csv")
#dataset = pd.read_csv("ravdes40W.csv")
print(dataset.head())

#X = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, 40].values


X = dataset.drop('Class', axis=1)
y = dataset['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

#scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(X_train)
#print("norm öncesi: ",X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#print("norm sonrasi: ",X_train)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
score = classifier.score(X_test,y_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("dogruluk: ",score)


'''
dizi = []
data, sampling_rate = librosa.load('/Users/cemergin/Downloads/saykorkuayca.wav')
mfccs = np.array(np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0))

#mfccs = StandardScaler()

myarray = np.asarray(mfccs)
print("standart scalerdan önce: ",myarray)
myarray = myarray.reshape(1, -1)

#myarray = scaler.transform(myarray)

print("standart scalerdan sonra: ",myarray)
#myarray = np.asarray(dizi)
print(myarray)
#print(dizi.shape)
#myarray = np.arange(40).reshape(40,1)

tahmin = classifier.predict(myarray)
print("tahminimiz: ", tahmin)
'''

data, sampling_rate = librosa.load('/Users/cemergin/Downloads/emo-DB/wav/08a05Wa.wav')
# print(f[34:])
# print(f[67:-16])
# mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
stft = np.abs(librosa.stft(data))
mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)  # 40 values
# zcr = np.mean(librosa.feature.zero_crossing_rate)
chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampling_rate).T, axis=0)
contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sampling_rate).T, axis=0)
tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sampling_rate).T,
                  # tonal centroid features
                  axis=0)
dizi = []
#dizi.append(mfccs)
for i in mfccs:
    dizi.append(i)
#dizi.append(chroma)
for i in chroma:
    dizi.append(i)
#dizi.append(mel)
#dizi.append(contrast)
for i in contrast:
    dizi.append(i)
#dizi.append(tonnetz)
for i in tonnetz:
    dizi.append(i)
#mfccs = StandardScaler()
print(dizi)

myarray = np.asarray(dizi)
#print("standart scalerdan önce: ",myarray)
print("shape: ",myarray.shape)
myarray = myarray.reshape(1, -1)

myarray = scaler.transform(myarray)
print(myarray)

#print("standart scalerdan sonra: ",myarray)
#myarray = np.asarray(dizi)
#print(myarray)
#print(dizi.shape)
#myarray = np.arange(40).reshape(40,1)

tahmin = classifier.predict(myarray)
print("tahminimiz: ", tahmin)
