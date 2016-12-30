#coding=utf-8
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense,Flatten,Dropout,merge
from keras.models import Model
from keras.callbacks import EarlyStopping  

from sklearn.preprocessing import LabelBinarizer
import numpy as np
import keras.backend as K  
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation


def getModel(modelName="imagenetclf3"):
    
        if modelName== "imagenetclf3":
            return imageNetClf3()
        if modelName=="mergeImagenet":
            return mergeImagenet()
     
        

########################################################       
def mergeImagenet():
    
    input_img = Input(shape=(3, 256, 256),name='main_input')

    ###############
    x1 = Convolution2D(36, 11, 11,border_mode='same')(input_img)
    x1= BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    
    
    x1 =Convolution2D(36,5,5,border_mode='same')(x1)
    x1= BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    
    x1 =Convolution2D(36,5,5,border_mode='same')(x1)
    x1= BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    

    
    x2 = Convolution2D(36, 7, 7,  border_mode='same')(input_img)
    x2= BatchNormalization()(x2)
    x2 = Activation("relu")(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    
    x2 =Convolution2D(36,5,5,border_mode='same')(x2)
    x2= BatchNormalization()(x2)
    x2 = Activation("relu")(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    
    x2 =Convolution2D(36,3,3,border_mode='same')(x2)
    x2= BatchNormalization()(x2)
    x2 = Activation("relu")(x2)
    x2 = MaxPooling2D((2, 2))(x2)

    x3 = Convolution2D(36, 5, 5, border_mode='same')(input_img)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)
    x3 = MaxPooling2D((2, 2))(x3)

    x3 = Convolution2D(36, 5, 5, border_mode='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)
    x3 = MaxPooling2D((2, 2))(x3)

    x3 = Convolution2D(36, 5, 5, border_mode='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)
    x3 = MaxPooling2D((2, 2))(x3)

    x = merge([x1,x2,x3],mode='concat')
    
    
    flatten = Flatten()(x)
    x= Dense(200,activation='relu')(flatten)
    # x = Dropout(0.5)(x)
    output= Dense(67,activation='softmax')(x)
    
    model = Model(input=input_img,output=output)
    # convo=Model(input=input_img,output=flatten)
    model.summary()
    
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model


########################################################
def imageNetClf3():
    #参考论文建立卷积模型  ImageNet Classification with Deep Convolutional Neural Netowrk
    '''
    '''

    input_img = Input(shape=(3, 256, 256),name='main_input')
    x1 = Convolution2D(36, 11, 11,border_mode='same')(input_img)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    
    x1 =Convolution2D(36,5,5,border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1) 
    x1 = MaxPooling2D((2, 2))(x1)
    
    x1 =Convolution2D(36,3,3,border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1) 
    x1 = MaxPooling2D((2, 2))(x1)
    
    x1 =Convolution2D(36,3,3,border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    
    x1 =Convolution2D(36,3,3,border_mode='same')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1) 
    # x1 = MaxPooling2D((2, 2))(x1)
    
    flatten = Flatten()(x1)
    x= Dense(200,activation='relu')(flatten)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    output= Dense(67,activation='softmax')(x)
    model = Model(input=input_img,output=output)
    # convo=Model(input=input_img,output=flatten)
    model.summary()
    model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
    return model

