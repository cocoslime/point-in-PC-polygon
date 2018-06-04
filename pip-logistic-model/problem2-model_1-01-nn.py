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

tf.set_random_seed(777)  # reproducibility

BOUNDARY_POINTS_NUM = 32 * 32
TRAINING_NUM = 1000
TEST_NUM = 100
LEARNING_RATE = 0.0005
MAX_STEP = 501

DATA_FOLDER = "buffer_010"

train_x_data, train_y_data = load_data("../data/problem2/" + DATA_FOLDER + "/training_", TRAINING_NUM)
test_x_data, test_y_data = load_data("../data/problem2/" + DATA_FOLDER + "/test_", TEST_NUM)

X = tf.placeholder(tf.float32, [None, (BOUNDARY_POINTS_NUM + 1) * 2], name='x-input')
Y = tf.placeholder(tf.float32, [None, 1], name='y-input')

W1, b1, L1 = make_layer_relu("W1", X, (BOUNDARY_POINTS_NUM + 1) * 2, 1024)

# Hidden Layer
W2, b2, L2 = make_layer_relu("W2", L1, 1024, 1024)
W3, b3, L3 = make_layer_relu("W3", L2, 1024, 512)
W4, b4, L4 = make_layer_relu("W4", L3, 512, 256)
W5, b5, L5 = make_layer_relu("W5", L4, 256, 10)

W_hypo, b_hypo, hypothesis = make_layer_sigmoid("W6", L5, 10, 1)
hypothesis = hypothesis * 0.998 + 0.001
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
tf.summary.scalar("cost", cost)

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Calculate accuracy
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)

print("\n\n ======= Training Start =======\n\n")

with tf.Session() as sess:
    # tensorboard --logdir=./logs/xor_logs
    merged_summary = tf.summary.merge_all()
    log_file_path = "../logs/problem2"
    if os.path.exists(log_file_path):
        shutil.rmtree(log_file_path)
    writer = tf.summary.FileWriter(log_file_path)
    writer.add_graph(sess.graph)  # Show the graph

    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(MAX_STEP):
        summary, c, _ = sess.run([merged_summary, cost, optimizer], feed_dict={X: train_x_data, Y: train_y_data})
        writer.add_summary(summary, global_step=step)

        if step % int(MAX_STEP / 10) == 0:
            print(step, c)
            # print("TEST accuarcy ", sess.run(accuracy, feed_dict={X: test_x_data, Y: test_y_data}))
            assert (c == c)

    print("Learning finished\n")
    tf.summary.FileWriterCache.clear()
    writer.close()

    print("\nTrain Accracy : ", sess.run(accuracy, feed_dict={X:train_x_data, Y:train_y_data}))
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: test_x_data, Y: test_y_data})
    print("\nAccuracy: ", a)

    result_filename = "../result/problem2_01_" + DATA_FOLDER + ".txt"
    os.makedirs(os.path.dirname(result_filename), exist_ok=True)
    result = open(result_filename, 'w')
    result.write("%f\n" % a)
    for item1, item2 in zip(h, c):
        result.write("%s %s\n" % (item1[0], item2[0]))

tf.reset_default_graph()
