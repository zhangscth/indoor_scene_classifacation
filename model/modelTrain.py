#coding=utf-8
'''
Created on 2016年9月24日
@author: kylin
'''
import  cPickle
import keras
from keras.callbacks import EarlyStopping  , Callback
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Flatten, Dropout, merge, BatchNormalization
from keras.models import Model
from keras.optimizers import SGD 
from keras.utils.visualize_util import plot
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from readData import readData
import keras.backend as K  
import numpy as np
import numpy as np
from softmaxUtil import prob2LabelBinary
# import xgboost as xgb
import sys
import time
#设置递归深度,防止Cpickle保存模型的时候出错:   RuntimeError: maximum recursion depth exceeded while pickling an object
sys.setrecursionlimit(1000000) #例如这里设置为一百万  

# 用于记录损失率的回调函数
class AccuracyHistory(keras.callbacks.Callback):
    def __init__(self,result_file):
        self.result_file = result_file
    def on_train_begin(self, logs={}):
        self.result_file.write("acc \t  val_acc \n")
        self.result_file.flush()
#         self.acc = []
 
    def on_epoch_end(self, epoch,logs={}):
        print "===========",str(logs.get("acc")) +"\t"+str(logs.get("val_acc"))
        self.result_file.write(str(logs.get("acc")) +"\t"+str(logs.get("val_acc"))+"\n")
        self.result_file.flush()
#         self.acc.append(logs.get('acc'))

def trainModel(X_train,Y_train,X_test,Y_test,modelName="imagenetclf3",nb_epoch=50,batch_size=32,patience=5,early_stop=True,model_save_file="classifier",result_file="result.txt"):
    
    np.random.seed(1024)
    #读取文件图片并获取label
    
    
#     X_train_flatten = [i.reshape((96*96)) for i in X_train]
#     X_train_flatten = np.asarray(X_train_flatten)
#     print X_train_flatten.shape
#     X_test_flatten = [i.reshape((96*96)) for i in X_test]
#     X_test_flatten = np.asarray(X_test_flatten)
#     print X_test_flatten.shape
    
    #结果保存在文件中
    result_file=open(result_file,"a+")
    
    #加载模型
    from modelFactory import getModel
    # modelName=""
    model = getModel(modelName)
    #模型可视化
    plot(model,to_file='imagenet_model_'+modelName+'.png')
    
    lb = LabelBinarizer()#将类二值化
    y_train=lb.fit_transform(Y_train)
#     y_validate = lb.fit_transform(Y_validate)
    #使用early stopping返回最佳epoch对应的model  
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)  
    
    layers_len = len(model.layers)
    print"=================", len(model.layers)
    acchistory= AccuracyHistory(result_file)
    if early_stop:
        hist=model.fit(X_train, y_train, batch_size=batch_size,  nb_epoch=nb_epoch, validation_split=0.2,  shuffle=True,callbacks=[acchistory,early_stopping])
    else:
        
        hist=model.fit(X_train, y_train, batch_size=batch_size,  nb_epoch=nb_epoch, validation_split=0.2,  shuffle=True,callbacks=[acchistory])
    # model.fit(X, y, batch_size=100,  nb_epoch=30, validation_split=0.2,  shuffle=True)
#     print acchistory
    
    
    #保存模型
    print "保存模型"
    #model.save('../pickleModel/imagenet_model_cnn_softmax_'+model_save_file+'.h5')  # creates a HDF5 file 'my_model.h5'
    # cPickle.dump(model,open('../pickleModel/imagenet_model_cnn_softmax_'+modelName+'.pkl',"wb"))  
    
    
    #####################   softmax     ###################
    
    
    predict_fc = model.predict(X_test)
    predict_fc = prob2LabelBinary(predict_fc)
    print "分类结果报告fc+softmax"
    y_test_lb = lb.fit_transform(Y_test)
    report= classification_report(y_test_lb,predict_fc)
    print report
    # print "",confusion_matrix(y_test_lb, predict_fc)
    #将结果写入到文件中
    result_file.write("分类结果报告fc+softmax\n")
    result_file.write(report)
    result_file.flush()
    
    
    ##################### conv+svm      ######################
    
    get_feature = K.function([K.learning_phase(),model.layers[0].input],model.layers[layers_len-3].output)  
    feature = get_feature([0,X_train[:600]])
    feature = np.concatenate((feature,get_feature([0,X_train[600:1000]])))
    feature = np.concatenate((feature,get_feature([0,X_train[1000:]])))
    # cPickle.dump(get_feature,open("./merge_imagenet_get_feature.pkl","wb"))
    
    #保存模型
    print "保存模型get_feature"
    #cPickle.dump(get_feature,open('imagenet_model_cnn_getFeature_'+model_save_file+'.pkl',"wb"))  
    try:get_feature.save('../pickleModel/imagenet_model_cnn_getFeature_'+model_save_file+'.h5')  # creates a HDF5 file 'my_model.h5'
    except Exception,e:
        print Exception,e
    
    ##############   cnn        svm ##############
    
    
    
    
    scaler = MinMaxScaler()
    print "============="
    # feature = scaler.fit_transform(feature)
    print feature.shape
    
    svc =  SVC(C=1.0,kernel="rbf",cache_size=900)  
    
    
    svc.fit(feature,Y_train)
    feature_test = get_feature([0,X_test])
    y_predict=svc.predict(feature_test)
    
    #保存模型
    print "保存模型cnn+svm"
    #cPickle.dump(svc,open('imagenet_model_cnn_svm_'+model_save_file+'.pkl',"wb"))  
    try:svc.save('../pickleModel/imagenet_model_cnn_svm_'+model_save_file+'.h5')  # creates a HDF5 file 'my_model.h5'
    except Exception,e:
        print Exception,e
    
    
    print "分类结果报告conv+svm"
    report= classification_report(Y_test,y_predict)
    confusion=confusion_matrix(Y_test, y_predict)
    
    print report
    print confusion
    
    #将结果写入到文件中
    result_file.write("\n分类结果报告conv+svm\n")
    result_file.write(report)
    result_file.write(confusion)
    result_file.flush()


    return 
    
    #######################gradient_boosting_classifier###################
    print "gradient_boosting_classifier.........."
    from sklearn_model import gradient_boosting_classifier
    clf = gradient_boosting_classifier(X_train_flatten,Y_train)
    print "training..."
    y_pred = clf.predict(X_test_flatten)
    
    print "分类结果报告svm"
    report=classification_report(Y_test,y_pred)
    confusion=confusion_matrix(Y_test, y_pred)
    
    print report
    print confusion
    
    #将结果写入到文件中
    result_file.write("\n分类结果报告boost..\n")
    result_file.write(report)
    result_file.write(confusion)
    result_file.flush()
    
    
    
    #############################################################
    
    ###############                ################
    #直接使用svm分类
    
    return
    
    print X_train_flatten.shape,Y_train.shape
    
    svc2 =  SVC(C=1.0,kernel="rbf",cache_size=900)  
    svc2.fit(X_train_flatten,Y_train)
    y_predict2 = svc2.predict(X_test_flatten)
    
    print "分类结果报告svm"
    report=classification_report(Y_test,y_predict2)
    confusion=confusion_matrix(Y_test, y_predict2)
    
    print report
    print confusion
    
    #将结果写入到文件中
    result_file.write("\n分类结果报告svm\n")
    result_file.write(report)
    result_file.write(confusion)
    result_file.flush()

    #保存模型
    print "保存模型cnn+svm"
    #cPickle.dump(svc2,open('imagenet_model_cnn_svm_'+model_save_file+'.pkl',"wb"))  
    try:svc2.save('../pickleModel/imagenet_model_svm2_'+model_save_file+'.h5')  # creates a HDF5 file 'my_model.h5'
    except Exception,e:
        print Exception,e
        
    
    '''
    
    #####################      ######################
    #使用xgboost进行分类
    
    # read in data
    
    dtrain = xgb.DMatrix( feature, label=y_train)
    # specify parameters via map
    param={
    'booster':'gbtree',
    # 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
    'objective': 'multi:softmax', 
    'num_class':8, # 类数，与 multisoftmax 并用
    'gamma':0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    'max_depth':12, # 构建树的深度 [1:]
    #'lambda':450,  # L2 正则项权重
    'subsample':0.4, # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    'colsample_bytree':0.7, # 构建树树时的采样比率 (0:1]
    #'min_child_weight':12, # 节点的最少特征数
    'silent':1 ,
    'eta': 0.005, # 如同学习率
    'seed':710,
    'nthread':4,# cpu 线程数,根据自己U的个数适当调整
    }
    num_round = 2
    bst = xgb.train(param, dtrain, num_round)
    # make prediction
    preds = bst.predict(feature[-300:])
    
    print "分类结果报告xgboost"
    print classification_report(Y_test,preds)
    print "",confusion_matrix(Y_test, preds)
    
    #####################      ######################
    #使用adboost算法
    n_split = -300
    
    X_train, X_test = feature, feature_test
    y_train, y_test = Y_train, Y_test
    
    bdt_discrete = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=12),
        n_estimators=600,
        learning_rate=1.5,
        algorithm="SAMME")
    
    
    bdt_discrete.fit(X_train, y_train)
    y_predict=bdt_discrete.predict(X_test)
    
    print "分类结果报告svm"
    print classification_report(Y_test,y_predict)
    print "",confusion_matrix(Y_test, y_predict)
    '''
    
    
    
if __name__=="__main__":
    X_train,Y_train,X_test,Y_test= readData()
    for nb_epoch in [50]:#1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50
        print "epoch:",nb_epoch
#         nb_epoch=10
        batch_size=20#32
        patience=1#
        early_stop=False
        modelName="mergeImagenet" #使用的模型 imagenetclf3
        model_save_file=modelName+"_"+str(nb_epoch)#模型保存的文件名
        result_file="../modelEvaluate/result_"+model_save_file+".txt"#模型评估的结果保存的文件名
        trainModel(X_train,Y_train,X_test,Y_test,modelName=modelName,nb_epoch=nb_epoch,batch_size=batch_size,patience=patience,early_stop=early_stop,model_save_file=model_save_file,result_file=result_file)
