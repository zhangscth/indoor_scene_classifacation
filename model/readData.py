#coding=utf-8
'''
    读取训练数据/测试数据

'''

from PIL import Image
import numpy as np

import os

class_label=['elevator', 'trainstation', 'office', 'hospitalroom', 'movietheater', 'children_room', 'inside_subway', 'library', 'nursery', 'greenhouse', 'bedroom', 'florist', 'bathroom', 'livingroom', 'closet', 'artstudio', 'tv_studio', 'computerroom', 'clothingstore', 'videostore', 'winecellar', 'bakery', 'restaurant_kitchen', 'corridor', 'warehouse', 'poolinside', 'inside_bus', 'kindergarden', 'hairsalon', 'fastfood_restaurant', 'deli', 'gym', 'bowling', 'airport_inside', 'buffet', 'church_inside', 'gameroom', 'laboratorywet', 'pantry', 'jewelleryshop', 'locker_room', 'studiomusic', 'auditorium', 'lobby', 'kitchen', 'classroom', 'waitingroom', 'bar', 'restaurant', 'concert_hall', 'casino', 'stairscase', 'meeting_room', 'shoeshop', 'dentaloffice', 'garage', 'mall', 'bookstore', 'grocerystore', 'subway', 'operating_room', 'cloister', 'toystore', 'laundromat', 'dining_room', 'museum', 'prisoncell']

def readData():
    path="/home/zsc/workspace/datasets/indoorCVPR_09/"
    imageRoot = path+"Images/"
    img_matrixs_train=[]
    img_labels_train=[]
#     img_matrixs_validate=[]
#     img_labels_validate=[]
    img_matrixs_test=[]
    img_labels_test=[]


    '''
    
    #读取train data 
    train_data_list_file = path+"TrainImages.txt"
    import numpy as np
    i=0
    for line in open(train_data_list_file):
#         i=i+1
#         if i>1000:
#             break
        line = line.strip()
        image_label  = line.split('/')[0] 
        image_label = class_label.index(image_label )#将类别数值化
        print line 
        image = Image.open(imageRoot+line)
        image = image.resize((256,256))
        print np.asarray(image).size
        print image.format,image.size,image.mode
        try:
            image=np.asarray(image).reshape((3,256,256))
            img_matrixs_train.append(image)
            img_labels_train.append(image_label)
        except Exception,e:
            print e
            
    '''
    # 读取 test data
    test_data_list_file = path + "TestImages.txt"
    test_file=[]
    i = 0
    for line in open(test_data_list_file):
        #         i=i+1
        #         if i>1000:
        #             break
        i=i+1
        line = line.strip()
        image_label = line.split('/')[0]
        image_label = class_label.index(image_label)  # 将类别数值化
        #print line
        image = Image.open(imageRoot + line)
        test_file.append(imageRoot+line)
        image = image.resize((256, 256))
        #print image.format, image.size, image.mode
        #print np.asarray(image).size
        try:
            image = np.asarray(image).reshape((3, 256, 256))
            img_matrixs_test.append(image)
            img_labels_test.append(image_label)
        except Exception, e:
            print e
    print i

    #获取训练集
    label_list = os.listdir(imageRoot)
    i = j=0
    for label_item in label_list:
        img_label = label_item
        image_label_lists = os.listdir(imageRoot + "/" + str(img_label))
        for imgname in image_label_lists:
            i = i + 1
            img =imageRoot+img_label+"/"+imgname
            #print img
            if img not in test_file:
                j=j+1
                image = Image.open(img)
                image = image.resize((256, 256))
                #print image.format, image.size, image.mode
                #print np.asarray(image).size
                try:
                    image = np.asarray(image).reshape((3, 256, 256))
                    img_matrixs_train.append(image)
                    label = class_label.index(img_label)
                    img_labels_train.append(label)
                except Exception, e:
                    print e

    print i,j


    
    
     #将数据打乱
#     import numpy as np
#     index = [i for i in range(len(img_matrixs_train))]  
#     np.random.shuffle(index) 
#     img_matrixs_train= img_matrixs_train[index]
#     img_labels_train = img_labels_train[index]

#     img_labels_train = np.random.shuffle(img_labels_train)
    
    # img_matrixs_train_ran=[]
    # for i in xrange(len(img_matrixs_train)):
    #     k = np.random.randint(10)
    #     if k<6:
    #         img_matrixs_train_ran.append(img_matrixs_train[i])
        
    
    # img_matrixs_train = np.asarray(img_matrixs_train[:3000])
    # img_matrixs_test = np.asarray(img_matrixs_test[:500])
    #
    # img_labels_train = np.asarray(img_labels_train[:3000])
    # img_labels_test = np.asarray(img_labels_test[:500])

    # 统计各个类出现的出现
    class_count = {}
    print class_count.items()
    for i in img_labels_train:
        if i not in class_count:
            class_count[i] = 1
        else:
            class_count[i] = class_count[i] + 1

    print "train count:", class_count

    # 统计各个类出现的出现
    class_count = {}
    print class_count.items()
    for i in img_labels_test:
        if i not in class_count:
            class_count[i] = 1
        else:
            class_count[i] = class_count[i] + 1

    print "test count:", class_count

    img_matrixs_train = np.asarray(img_matrixs_train)
    img_matrixs_test = np.asarray(img_matrixs_test)

    img_labels_train = np.asarray(img_labels_train)
    img_labels_test = np.asarray(img_labels_test)
    
#     print "==================",len(img_matrixs_train),len(img_matrixs_test),len(img_labels_train),len(img_labels_test)
    return img_matrixs_train,img_labels_train,img_matrixs_test,img_labels_test
        
    
    
    
# readData()

    