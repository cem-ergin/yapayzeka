import librosa
import pandas as pd
import numpy as np
bankdata = pd.read_csv("hepsibirlesik.csv")

# Assign data from first four columns to X variabl
X = bankdata.drop('Class', axis=1)
# Assign data from first fifth columns to y variable
y = bankdata['Class']


#from sklearn import preprocessing
#le = preprocessing.LabelEncoder()

#y = y.apply(le.fit_transform)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

#scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train.values.ravel())

predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


dizi = []
data, sampling_rate = librosa.load('/Users/cemergin/Downloads/aycanormal.wav')
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

tahmin = mlp.predict(myarray)
print("tahminimiz: ", tahmin)
score = mlp.score(X_test,y_test)
print("dogruluk: ",score)