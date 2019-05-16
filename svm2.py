#Import scikit-learn dataset library
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

#Load dataset
sesTanima = pd.read_csv("ravdes40neutralanger.csv")

X = sesTanima.drop('Class', axis=1)
y = sesTanima['Class']

#print(X)
#print(y)
# Split dataset into training set and test se
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=109) # 70% training and 30% test

#Create a svm Classifier
clf = svm.SVC(kernel='rbf', random_state=1, gamma=0.1, C=100000) # Linear Kernel

my_svc = svm.SVC(probability=True, C=1000)
my_svc.fit(X_train,y_train)

p_classification = my_svc.predict(X_test)

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("p_accuracy: ",metrics.accuracy_score(y_test,p_classification))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

