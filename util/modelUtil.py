#coding=utf-8
import matplotlib.pyplot as plt

from scipy.misc import imshow,imresize
from PIL import Image
import numpy as np
from keras import backend as K

#Get features function
def get_features(model, layer, X_batch):
    get_feature = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    features = get_feature([0,X_batch])
    return features

#Hyper-column extraction:
def extract_hypercolumn(model, layer_indexes, instance):
    layers = [K.function([K.learning_phase(),model.layers[0].input],[model.layers[li].output])([instance])[0] for li in layer_indexes]
    feature_maps = get_features(model,layers,instance)
    hypercolumns = []
    for convmap in feature_maps:
        for fmap in convmap[0]:
            upscaled = imresize(fmap, size=(224, 224),mode="F", interp='bilinear')
            hypercolumns.append(upscaled)
    return np.asarray(hypercolumns)

'''
import cPickle
model = cPickle.load(open(r"../modelSave/merge_imagenet2.pkl","rb"))

#after necessary processing of input to get im
from data.data import getData
X_train,Y_train,X_test,Y_test = getData()

print"====predict:", model.predict(X_test)

get_feature = K.function([K.learning_phase(),model.layers[0].input],[model.layers[10].output,])
feat = get_feature(0,X_test)
plt.imshow(feat[0][2])
'''





