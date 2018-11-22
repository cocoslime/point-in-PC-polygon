import matplotlib.pyplot as plt
import shutil
import tensorflow as tf


def model1(inputs, keep_prob):
    W1 = tf.get_variable("W1", shape=[3, 3, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
    L1 = tf.nn.conv2d(inputs, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
    '''
    Tensor("Conv2D:0", shape=(?, 32, 32, 32), dtype=float32)
    Tensor("Relu:0", shape=(?, 32, 32, 32), dtype=float32)
    Tensor("MaxPool:0", shape=(?, 16, 16, 32), dtype=float32)
    Tensor("dropout/mul:0", shape=(?, 16, 16, 32), dtype=float32)
    '''

    W2 = tf.get_variable("W2", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    '''
    Tensor("Conv2D_1:0", shape=(?, 16, 16, 64), dtype=float32)
    Tensor("Relu_1:0", shape=(?, 16, 16, 64), dtype=float32)
    Tensor("MaxPool_1:0", shape=(?, 8, 8, 64), dtype=float32)
    Tensor("dropout_1/mul:0", shape=(?, 8, 8, 64), dtype=float32)
    '''

    W3 = tf.get_variable("W3", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
    '''
    '''

    W4 = tf.get_variable("W4", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
    L4_flat = tf.reshape(L4, [-1, 8 * 8 * 128])


    # FC 5x5x128 inputs -> 625 outputs
    W_FC1 = tf.get_variable("W_FC1", shape=[8 * 8 * 128, 625], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([625]))
    L_FC1 = tf.nn.relu(tf.matmul(L4_flat, W_FC1) + b1)
    L_FC1 = tf.nn.dropout(L_FC1, keep_prob=keep_prob)
    '''
    Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
    Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
    '''

    w = tf.get_variable(name="W_FC2", shape=[625, 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([1]))
    hypothesis = tf.sigmoid(tf.matmul(L_FC1, w) + b)
    hypothesis = hypothesis * 0.999998 + 0.000001
    return hypothesis


def model2(inputs, keep_prob):
    weight1 = tf.get_variable("W1", shape=[3, 3, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
    layer1 = tf.nn.conv2d(inputs, weight1, strides=[1, 1, 1, 1], padding='SAME')
    layer1 = tf.nn.relu(layer1)
    layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)
    '''
    Tensor("Conv2D:0", shape=(?, 32, 32, 32), dtype=float32)
    Tensor("Relu:0", shape=(?, 32, 32, 32), dtype=float32)
    Tensor("MaxPool:0", shape=(?, 16, 16, 32), dtype=float32)
    Tensor("dropout/mul:0", shape=(?, 16, 16, 32), dtype=float32)
    '''

    W2 = tf.get_variable("W2", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    layer2 = tf.nn.conv2d(layer1, W2, strides=[1, 1, 1, 1], padding='SAME')
    layer2 = tf.nn.relu(layer2)
    layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)
    '''
    Tensor("Conv2D_1:0", shape=(?, 16, 16, 64), dtype=float32)
    Tensor("Relu_1:0", shape=(?, 16, 16, 64), dtype=float32)
    Tensor("MaxPool_1:0", shape=(?, 8, 8, 64), dtype=float32)
    Tensor("dropout_1/mul:0", shape=(?, 8, 8, 64), dtype=float32)
    '''

    W3 = tf.get_variable("W3", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    layer3 = tf.nn.conv2d(layer2, W3, strides=[1, 1, 1, 1], padding='SAME')
    layer3 = tf.nn.relu(layer3)
    layer3 = tf.nn.dropout(layer3, keep_prob=keep_prob)
    '''
    '''

    W4 = tf.get_variable("W4", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    layer4 = tf.nn.conv2d(layer3, W4, strides=[1, 1, 1, 1], padding='SAME')
    layer4 = tf.nn.relu(layer4)
    layer4 = tf.nn.dropout(layer4, keep_prob=keep_prob)
    '''
    '''

    weight5 = tf.get_variable("W5", shape=[3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    layer5 = tf.nn.conv2d(layer4, weight5, strides=[1, 1, 1, 1], padding='SAME')
    layer5 = tf.nn.relu(layer5)
    layer5 = tf.nn.max_pool(layer5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer5_flat = tf.reshape(layer5, [-1, 4 * 4 * 256])
    '''
    '''

    W_FC1 = tf.get_variable("W_FC1", shape=[4 * 4 * 256, 1000], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([1000]))
    L_FC1 = tf.nn.relu(tf.matmul(layer5_flat, W_FC1) + b1)
    L_FC1 = tf.nn.dropout(L_FC1, keep_prob=keep_prob)
    '''
    Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
    Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
    '''

    weight_fc2 = tf.get_variable("W_FC2", shape=[1000, 100], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([100]))
    layer_fc2 = tf.nn.relu(tf.matmul(L_FC1, weight_fc2) + b2)
    layer_fc2 = tf.nn.dropout(layer_fc2, keep_prob=keep_prob)

    w = tf.get_variable(name="weight", shape=[100, 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([1]))
    hypothesis = tf.sigmoid(tf.matmul(layer_fc2, w) + b)
    hypothesis = hypothesis * 0.999998 + 0.000001
    return hypothesis
