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
    
# function that load all necessary data
def load_and_process_training_Data(train_data, train_label):
    # check if the file exist in the directory
    if os.path.exists(train_data)and os.path.exists(train_label):
        print('loading train data ' + train_data)
        print('loading train label ' + train_label)
        features_train = load_Data(train_data)
        features_labels = load_Data(train_label)
        # trim half channel
        channelLength = len(features_train[0])
        channelFirstHalfLength = channelLength//2
        channelFirstHalf_test = [item[:channelFirstHalfLength] for item in features_train]
        features_train = channelFirstHalf_test
        # process data
        print('processing data')
        labels = []
        for l in features_labels:
            if(l==1):
                labels.append([1,0])
            else :
                labels.append([0,1])
        print('preprocess finished')
    else:
        print('fail to read file, file not exist or be moved from directory');
        sys.exit(0)
    return features_train,labels
def load_and_process_test_data(test_file,labels):
    if os.path.exists(test_file)and os.path.exists(labels):
        print('loading test data ' + test_file)
        print('loading test label ' + labels)
        test_data = load_Data(test_file)
        test_labels = load_Data(labels)
        # trim half channel
        channelLength = len(test_data[0])
        channelFirstHalfLength = channelLength//2
        channelFirstHalf_test = [item[:channelFirstHalfLength] for item in test_data]
        test_data = channelFirstHalf_test
        # process data
        print('processing data')
        labels = []
        for l in test_labels:
            if(l==1):
                labels.append([1,0])
            else :
                labels.append([0,1])
        print('preprocess finished')
        return test_data,labels
    
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
