import sys
import os
import numpy as np
import random
import pickle
from collections import Counter
    
# preprocess data by remove the comment row
def load_Data(dataSample):
    contents = np.loadtxt(dataSample,delimiter=',')
    return contents
    
# sample_handling, turn the data in to feature set matrix for nn computing
def sample_handling(sample,classification):
    featureSet = []
    # set sample to list for easy handle
    sample = sample.tolist()
    # iterate though all the feature in income data set
    for feature in sample:
        featureSet.append([feature,classification])
    return featureSet
    
# function that load all necessary data
def load_and_process_training_Data(targ_f_train, nontarg_f_train):
    # check if the file exist in the directory
    if os.path.exists(targ_f_train)and os.path.exists(nontarg_f_train):
        # load data
        # load train data, we are loading events 1640*60 so 60 row of 1640 dataPoint
        print('loading train data ' + targ_f_train)
        targEvent_train = load_Data(targ_f_train)
        print('loading train data ' + nontarg_f_train)
        nontargetEvent_train = load_Data(nontarg_f_train)
        # finish load and begin process
        print('processing data')
        # mark targ feature with [1,0] and non targ with [0,1]
        # train featureSet
        features_train = []
        features_train += sample_handling(targEvent_train,[1,0])
        features_train += sample_handling(nontargetEvent_train,[0,1])
        # they said it good to shuffle
        # shuffle the train data
        random.shuffle(features_train)
        features_train = np.array(features_train)
        # train data
        train_x = list(features_train[:,0])
        train_y = list(features_train[:,1])
        #print(train_y)
        #firstOccur = 0
        #count = 0
        #for i in range(len(train_y)):
        #    if train_y[i] ==[1,0]:
        #        firstOccur = i
        #        count+=1
        #print(count)
        #
        #print(train_x[firstOccur])
        #print(train_y[firstOccur])
        # finished loading
        print('preprocess finished')
    else:
        print('fail to read file, file not exist or be moved from directory');
        sys.exit(0)
    return train_x,train_y
def load_and_process_test_data(test_file,labels):
    if os.path.exists(test_file)and os.path.exists(labels):
        print('loading test data ' + test_file)
        print('loading test label ' + labels)
        test_data = load_Data(test_file)
        test_labels = load_Data(labels)
        print('processing data')
        labels = []
        for l in test_labels:
            if(l==1):
                labels.append([1,0])
            else :
                labels.append([0,1])
        print('preprocess finished')
        return test_data,labels





''' 
# for test
# if execute file directly
if __name__ =='__main__':
    # ask targ file name and non targ file name to preprocess
    # we can pre process the data and store in a file so we do not have to go though preprocess again
    # however, preprocess are huge take to many disk space, so not using it, but if you want, you can use it
    targ_f_train = input('Enter a train targfeatures file: ')
    nontarg_f_train = input('Enter a train nontargetFeatures file: ')
    test_data = input('Enter the test dataset file: ')
    labels = input('Enter the labels file for test data: ')
    train_x,train_y= load_and_process_training_Data(targ_f_train,nontarg_f_train)
    test_x,test_y= load_and_process_test_data(test_data,labels)
    # here where you can use to execute the save preprocess data in to file
    #with open('preprocessedData.pickle','wb') as f:
    #	pickle.dump([train_x,train_y,test_x,test_y],f)
'''