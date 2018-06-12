# Solve problem2 with Neural network
import csv
import shutil
import os
import tensorflow as tf
from func2 import *
from func1 import *
import random
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.set_random_seed(777)  # reproducibility

BOUNDARY_POINTS_NUM = 20 * 20
TRAINING_NUM = 1000
TEST_NUM = 300
LEARNING_RATE = 0.001
MAX_STEP = 2001

DATA_FOLDER = "buffer_001"

for index in range(1000, 20000, 500):
    tf.reset_default_graph()
    TRAINING_NUM = index
    train_x_data, train_y_data = load_data("../data/problem2/" + DATA_FOLDER + "/training_", TRAINING_NUM)
    test_x_data, test_y_data = load_data("../data/problem2/" + DATA_FOLDER + "/test_", TEST_NUM)

    X = tf.placeholder(tf.float32, [None, (BOUNDARY_POINTS_NUM + 1) * 2], name='x-input')
    Y = tf.placeholder(tf.float32, [None, 1], name='y-input')

    keep_prob = tf.placeholder(tf.float32)

    W1, b1, L1 = make_layer_relu("W1", X, (BOUNDARY_POINTS_NUM + 1) * 2, (BOUNDARY_POINTS_NUM + 1) * 2)
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    # Hidden Layer
    W2, b2, L2 = make_layer_relu("W2", L1, (BOUNDARY_POINTS_NUM + 1) * 2, (BOUNDARY_POINTS_NUM + 1))
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    W3, b3, L3 = make_layer_relu("W3", L2, (BOUNDARY_POINTS_NUM + 1), (BOUNDARY_POINTS_NUM + 1))
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
    W4, b4, L4 = make_layer_relu("W4", L3, (BOUNDARY_POINTS_NUM + 1), 200)
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
    W5, b5, L5 = make_layer_relu("W5", L4, 200, 10)
    L5 = tf.nn.dropout(L5, keep_prob=keep_prob)

    W_hypo, b_hypo, hypothesis = make_layer_sigmoid("W6", L5, 10, 1)
    hypothesis = hypothesis * 0.998 + 0.001
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Calculate accuracy
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    # print("\n\n ======= Training Start =======\n\n")

    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())

        for step in range(MAX_STEP):
            c, _ = sess.run([cost, optimizer], feed_dict={X: train_x_data, Y: train_y_data, keep_prob: 0.7})
            # writer.add_summary(summary, global_step=step)

            if step % int(MAX_STEP / 10) == 0:
                print(step, c)
                assert (c == c)

        # print("Learning finished\n")

        # print("\nTrain Accracy : ", sess.run(accuracy, feed_dict={X:train_x_data, Y:train_y_data, keep_prob:1.0}))
        h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: test_x_data, Y: test_y_data, keep_prob:1.0})
        print(str(index) + "," + str(a))



