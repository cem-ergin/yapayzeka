# Load DATA
import librosa
import pandas as pd
import numpy as np
#from matplotlib.mlab import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df = pd.read_csv('ravdes40Wemodb40.csv')

print("aloooo")
# Header for Features without Labels
features = [str(i) for i in range(0,40)]

# Standarize the DATA
#X = df.loc[features,:].values

#Y = df.loc[:,'Class'].values

X = df.drop('Class',axis=1)
Y = df['Class']

X = StandardScaler().fit_transform(X)

# PCA
# n : Number of principal components
n = 40
pca = PCA(n_components = n)

X = pca.fit_transform(X)

#Split data to train and test
X_train, X_test, Y_train, Y_test = train_test_split (X,Y,test_size=0.2)

#Create a Support Vector Classifier instance
classifier = svm.SVC(kernel='linear')

#Fit the classifier
classifier.fit(X_train,Y_train)

#Calculate the score (Accuracy)
score = classifier.score(X_test,Y_test)

predict = classifier.predict(X_test)

#Printing the score

print(score)

dizi = []
data, sampling_rate = librosa.load('/Users/cemergin/Downloads/emo-DB/wav/10a02Lb.wav')
mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
for aa in mfccs:
    dizi.append(aa)

myarray = np.asarray(dizi)
#print(dizi.shape)
#myarray = np.arange(40).reshape(40,1)
myarray = myarray.reshape(1, -1)
tahmin = classifier.predict(myarray)
print("tahminimiz: ", tahmin)