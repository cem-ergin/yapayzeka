import librosa
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#bankdata = pd.read_csv("/Users/cemergin/PycharmProjects/Speech-Emotion-Recognition/bill_authentication.csv")
#bankdata = pd.read_csv("/Users/cemergin/Documents/Book10.csv")
from ozellikler import Radar

bankdata = pd.read_csv("cemtessemodbravdes.csv")
print(bankdata.shape)
print(bankdata.head())


X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

CLASS_LABELS = ("Korku", "Mutlu", "Normal", "Sinirli", "Uzuntulu", "Igrenmis")
NUM_LABELS=len(CLASS_LABELS)

class_names = y  # Reads all the folders in which images are present
class_names = sorted(class_names) # Sorting them
name_id_map = dict(zip(class_names, range(len(class_names))))
print(name_id_map)


#X = bankdata.drop('label',axis=1)
#y = bankdata['label']
#X = StandardScaler().fit_transform(X)
#standart scaler
scaler = StandardScaler()
#min-max scaler
#scaler = MinMaxScaler()
# PCA
# n : Number of principal components
#n = 40
#pca = PCA(n_components = n)
#X = pca.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#normalizasyon
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#print(X_train)
#print(y_train)

svclassifier = SVC(kernel='rbf',probability=True)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)
#print(y_pred)
score = svclassifier.score(X_test,y_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("dogruluk: ",score)
'''
dizi = []
data, sampling_rate = librosa.load('/Users/cemergin/Downloads/emo-DB/wav/03a01Wa.wav')
mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
for aa in mfccs:
    dizi.append(aa)

myarray = np.asarray(dizi)
#print(dizi.shape)
#myarray = np.arange(40).reshape(40,1)
myarray = myarray.reshape(1, -1)
tahmin = svclassifier.predict(myarray)
print("tahminimiz: ", tahmin)
'''

''' 16 mayıs 19:06
dizi = []
data, sampling_rate = librosa.load('/Users/cemergin/Downloads/sayigrenmisayca.wav')
mfccs = np.array(np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0))

#mfccs = StandardScaler()

myarray = np.asarray(mfccs)
print("standart scalerdan önce: ",myarray)
myarray = myarray.reshape(1, -1)

myarray = scaler.transform(myarray)

print("standart scalerdan sonra: ",myarray)
#myarray = np.asarray(dizi)
print(myarray)
#print(dizi.shape)
#myarray = np.arange(40).reshape(40,1)

tahmin = svclassifier.predict(myarray)
print("tahminimiz: ", tahmin)

'''

data, sampling_rate = librosa.load('/Users/cemergin/Downloads/aycanormal.wav')
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

tahmin = svclassifier.predict(myarray)
print("tahminimiz: ", tahmin)

result = np.argmax(svclassifier.predict(myarray))
result_prob = svclassifier.predict(myarray)
print('Recogntion: ', result)
print('Probability: ', result_prob)
proba = svclassifier.predict_proba(myarray)
print("result proba: ",proba)
proba_list = []
for i in proba:
    proba_list.append(i)

proba_array = np.asarray(proba_list)
print("proba_array: ",proba_array)
print("proba_array[0]: ",proba_array[0])
Radar(proba_array[0], CLASS_LABELS, NUM_LABELS)