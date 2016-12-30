#coding=utf-8
import numpy as np

#将softmax的结果labelBinarize
def prob2LabelBinary(proMatrix):
    assert type(proMatrix)==np.ndarray
    len = proMatrix.shape[1]
    print len
    results=[]
    
    for i in xrange(proMatrix.shape[0]):
        cl = np.argmax(proMatrix[i])
        result = np.zeros(len,dtype="int8")
        result[cl]=1
        results.append(result)
    results = np.asarray(results)
    return results


def labelBinary2Label(labelMatrix):
    labels=[]
    for i in xrange(labelMatrix.shape[0]):
        cl = np.argmax(labelMatrix[i])
        labels.append(cl)
    return np.asarray(labels)
    

# print prob2Label(np.asarray([[1,2,3],[2,3,1]]))