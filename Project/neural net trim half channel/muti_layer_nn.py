import tensorflow as tf
import numpy as np
from math import exp
from PreProcessDataSetWithTrim import load_and_process_training_Data,load_and_process_test_data
tf.set_random_seed(0)

# Get all pre processed data
# load training data, test data
#train_x,train_y= load_and_process_training_Data('targfeatures_train.txt','nontargetFeatures_train.txt')
#test_x,test_y = load_and_process_test_data('testfeatures.txt','testlabels.txt') 
train_x,train_y= load_and_process_training_Data('targfeatures_compress.txt','nontargetFeatures_compress.txt')
test_x,test_y = load_and_process_test_data('testfeatures_compress.txt','testlabels_compress.txt')

# set up parameters we need for nn model
# trained neural network path
save_path = "nn_saved_model/model_compress_samenode/model.ckpt"
# The number of class you want to have in NN. In this case we want NN to determine which dataset belone 
# to target signal or non_target signal
n_classes = 2 
# Number of node each hidden layer will have
n_nodes_hl1 = 100
n_nodes_hl2 = 100
n_nodes_hl3 = 100
# number of times we iterate through training data
num_epochs = 100
# computer may not have enough memory, so we divide the train into batch each batch have 100 data features.
batch_size = 100

# These are placeholders for some values in graph
# tf.placeholder(dtype, shape=None(optional), name=None(optional))
# It's a tensor to hold our datafeatures 
x = tf.placeholder(tf.float32, [None,len(train_x[0])])
# Every row has either [1,0] for targ or [0,1] for non_target. placeholder to hold one hot value
Y_C = tf.placeholder(tf.int8, [None, n_classes])
# variable learning rate
lr = tf.placeholder(tf.float32)

# neural network model
def neural_network_model(data):
    # layers contain weights and bias for case like all neurons fired a 0 into the layer, we will need result out
    # When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),'bias':tf.Variable(tf.ones([n_nodes_hl1])/10)}
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),'bias':tf.Variable(tf.ones([n_nodes_hl2])/10)}
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),'bias':tf.Variable(tf.ones([n_nodes_hl3])/10)}
    # no more bias when come to the output layer
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),'bias':tf.Variable(tf.zeros([n_classes]))}
    
    # multiplication of the raw input data multipled by their unique weights (starting as random, but will be optimized)
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)
    # We repeat this process for each of the hidden layers, all the way down to our output, where we have the final values still being the multiplication of the input and the weights, plus the output layer's bias values.
    Ylogits = tf.matmul(l3,output_layer['weights']) + output_layer['bias']
    return Ylogits

# set up the training process
def train_neural_network(x):
    # produce the prediction base on output of nn model
    Ylogits = neural_network_model(x)
    # measure the error use build in cross entropy function, the value that we want to minimize
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_C))
    # To optimize our cost (cross_entropy), reduce error, default learning_rate is 0.001, but you can change it, this case we use default
    # optimizer = tf.train.GradientDescentOptimizer(0.003)
    optimizer = tf.train.AdamOptimizer(lr)
    train_step = optimizer.minimize(cross_entropy)
    # start the session
    with tf.Session() as sess:
        # We initialize all of our variables first before start
        sess.run(tf.global_variables_initializer())
        # iterate epoch count time (cycles of feed forward and back prop), each epoch means neural see through all train_data once
        for epoch in range(num_epochs):
            # count the total cost per epoch, declining mean better result
            epoch_loss=0
            i=0
            # learning rate decay
            max_learning_rate = 0.003
            min_learning_rate = 0.0001
            decay_speed = 150
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * exp(-epoch/decay_speed)
            # divide the dataset in to dataset/batch_size in case run out of memory
            while i < len(train_x):
                # load train data
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                train_data = {x: batch_x, Y_C: batch_y, lr: learning_rate}
                # train
                # sess.run(train_step,feed_dict=train_data)
				# run optimizer and cost against batch of data.
                _, c = sess.run([train_step, cross_entropy], feed_dict=train_data)
                epoch_loss += c
                i+=batch_size
            print('Epoch', epoch, 'completed out of',num_epochs,'loss:',epoch_loss)
        # how many predictions we made that were perfect matches to their labels
        # test model
        # test data 
        test_data = {x:test_x, Y_C:test_y}
        # calculate accuracy
        correct_prediction = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y_C, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print('Accuracy:',accuracy.eval(test_data))
        # result matrix, return the position of 1 in array
        result = (sess.run(tf.argmax(Ylogits.eval(feed_dict=test_data),1)))
        answer = []
        for i in range(len(test_y)):
            if test_y[i] == [0,1]:
                answer.append(1)
            elif test_y[i]==[1,0]:
                answer.append(0)
        answer = np.array(answer)
        printResultandCorrectMatrix(result,answer)
        #np.savetxt('nn_prediction.txt', Ylogits.eval(feed_dict={x: test_x}), delimiter=',',newline="\r\n")
        # save the nn model for later use again
        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()
        saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

# load the trained neural network model 
def test_loaded_neural_network():
    Ylogits = neural_network_model(x)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # load saved model
        saver.restore(sess, save_path)
        print("Loading variables from ‘%s’." % save_path)
        np.savetxt('nn_prediction.txt', Ylogits.eval(feed_dict={x: test_x}), delimiter=',',newline="\r\n")
        # test model
        # calculate accuracy
        correct_prediction = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y_C, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x, Y_C:test_y}))
        # result matrix
        result = (sess.run(tf.argmax(Ylogits.eval(feed_dict={x:test_x}),1)))
        # answer matrix
        answer = []
        for i in range(len(test_y)):
            if test_y[i] == [0,1]:
                answer.append(1)
            elif test_y[i]==[1,0]:
                answer.append(0)
        answer = np.array(answer)
        printResultandCorrectMatrix(result,answer)
        print(Ylogits.eval(feed_dict={x: test_x}).shape)
def printResultandCorrectMatrix(result,answer):
    print("Result matrix: ")
    print(result)
    # counter for positive and negative reflection
    positiveCount = 0
    negativeCount = 0
    for i in np.nditer(result):
        if i == 0:
            positiveCount+=1
        elif i == 1:
            negativeCount+=1
    print("Positive count ", positiveCount)
    print("Negative count ", negativeCount)
    print("Answer matrix: ")
    print(answer)
    countCorrectMatch = 0
    for i in range(len(answer)):
        if answer[i]==0:
            if result[i]==0:
                countCorrectMatch+=1
    print("Correct match labels is ", countCorrectMatch)
    
''' plot result
def plotGraph(s,prediction):
    import matplotlib.pyplot as plt
    xx = [v[0] for v in test_x]
    yy = [v[1] for v in test_y]
    x_min, x_max = min(xx) - 0.5, max(xx) + 0.5 
    y_min, y_max = min(yy) - 0.5, max(yy) + 0.5 
    xxx, yyy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    pts = np.c_[xxx.ravel(), yyy.ravel()].tolist()
    # ---> Important
    z = s.run(tf.argmax(prediction, 1), feed_dict = {x: pts})
    z = np.array(z).reshape(xxx.shape)
    plt.pcolormesh(xxx, yyy, z)
    plt.scatter(xx, yy, c=['r' if v[0] == 1 else 'b' for v in y_data], edgecolor='k', s=50)
    plt.show()
'''

#train_neural_network(x)
test_loaded_neural_network()
