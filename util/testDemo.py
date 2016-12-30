#coding=utf-8
'''
Created on 2016年10月16日

@author: kylin
'''
'''
from softmaxUtil import prob2Label
import numpy as np
a = np.asarray([0.1, 0.2,0.3])
print prob2Label(a)
'''

from data.data import getData
from keras.models import load_model
from util.softmaxUtil import prob2LabelBinary
import sys
sys.setrecursionlimit(1000000)
X_train,Y_train,X_test,Y_test = getData()

model = load_model("../pickleModel/imagenet_model_cnn_softmax_imagenetclf3.h5")
print model.summary()

y_pred = model.predict(X_test)
y_pred = prob2LabelBinary(y_pred)
print y_pred