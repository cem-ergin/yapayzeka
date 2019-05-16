import numpy
from tensorflow_estimator.python.estimator.canned.timeseries import model
from cffi import model

from ML_Model import SVM_Model
from SER import LSTM, DATA_PATH
from SER import CNN
from SER import SVM
from SER import MLP
from Utilities import load_model, get_feature_svm
# load_model: the type of model (DNN / ML)

from DNN_Model import LSTM_Model
from DNN_Model import CNN_Model



from Utilities import get_data
from Utilities import get_feature

model.recognize_one(get_feature("/Users/cemergin/PycharmProjects/Speech-Emotion-Recognition/Models/DataSet/RAVDESS/angry/03-01-05-01-01-01-01.wav"))
DATA_PATH="/Users/cemergin/PycharmProjects/Speech-Emotion-Recognition/Models/DataSet/RAVDESSson"
#DATA_PATH="/Users/cemergin/PycharmProjects/Speech-Emotion-Recognition/Models/DataSet/RAVDESS"
#SVM("/Users/cemergin/PycharmProjects/Speech-Emotion-Recognition/Models/DataSet/RAVDESS")

# When using SVM, _svm = True, or _svm = False
#asdf = get_feature_svm("/Users/cemergin/PycharmProjects/Speech-Emotion-Recognition/Models/DataSet/RAVDESSson/angry/03-01-05-01-01-01-01.wav", 39)
#numpy.savetxt("asdf.txt",asdf)
#print("yazdım")
#print(asdf)
x_train, x_test, y_train, y_test = get_data(DATA_PATH,_svm=True)
#print(len(x_train))
#print(len(x_test))
#print(x_train.shape)
#print(x_test.shape)
#print(x_train)
#print(x_test)

numpy.savetxt("x_train.txt",x_train)
numpy.savetxt("x_test.txt",x_test)
numpy.savetxt("y_train.txt",y_train)
numpy.savetxt("y_test.txt",y_test)
model_svm = SVM_Model()
model_svm.save_model("svmmodeli")
model_svm.train(x_train, y_train)
model_svm.evaluate(x_test, y_test)
#model.recognize_one(get_feature("/Users/cemergin/PycharmProjects/Speech-Emotion-Recognition/Models/DataSet/RAVDESS/angry/03-01-05-01-01-01-01.wav"))


#a = get_feature("/Users/cemergin/PycharmProjects/audio-classification/data/03a01Nc.wav",39,flatten=1)
#print(a)
#print(type(a))
#print(len(a))
#numpy.savetxt("feature.txt",a)
#-36.04365339 03a01nc
#1134 -5.689264606224537779e-01 fa