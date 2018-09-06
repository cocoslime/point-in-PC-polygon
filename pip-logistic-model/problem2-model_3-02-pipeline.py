# each model has each vertices num

import time
import matplotlib.pyplot as plt
import csv
import shutil
import os
import tensorflow as tf
from loaddata import *
from func1 import *
import random
header = __import__("problem2-model_3-header")

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.set_random_seed(777)  # reproducibility

HAS_SUMMARY = False

test_x_data = []
test_y_data = []

train_file_list = []
test_file_list = []

start_time = time.time()
tf.reset_default_graph()

for numSides in header.NUM_SIDES:
    test_file_name = "../data/problem2/convex/raster_pc/" + header.BUFFER_OPT + "/test_" + str(numSides) + ".csv"
    test_file_list.append(test_file_name)
    train_file_name = "../data/problem2/convex/raster_pc/" + header.BUFFER_OPT + "/training_" + str(numSides) + ".csv"
    train_file_list.append(train_file_name)

test_x_data, test_y_data = load_raster_data(test_file_list)

record_defaults = [[0.]] * (header.WIDTH_NUM * header.HEIGHT_NUM + 1)
train_xy_data = make_decode_CSV_list(train_file_list, record_defaults)

print("=========== BATCH ===========")
train_x_data = train_xy_data[0:-1]
train_y_data = train_xy_data[-1]
train_y_data = tf.reshape(train_y_data, [1])
batch_train_x, batch_train_y = \
    tf.train.shuffle_batch([train_x_data, train_y_data], min_after_dequeue=10000, capacity=50000, enqueue_many=False,
                           batch_size=header.BATCH_SIZE, num_threads=8)

print("=========== BUILD GRAPH ===========")

# input place holders
X = tf.placeholder(tf.float32,  [None, header.WIDTH_NUM * header.HEIGHT_NUM])
X_img = tf.reshape(X, [-1, header.WIDTH_NUM, header.HEIGHT_NUM, 1])
Y = tf.placeholder(tf.float32, [None, 1])

keep_prob = tf.placeholder(tf.float32)

# W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
W1 = tf.get_variable("W1", shape=[3, 3, 1, 32],
                     initializer=tf.contrib.layers.xavier_initializer())
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
# max_pool : (?, 20, 20, 32) -> (?, 10, 10, 32)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
W2 = tf.get_variable("W2", shape=[3, 3, 32, 64],
                     initializer=tf.contrib.layers.xavier_initializer())
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
# max_pool : (?, 10, 10, 64) -> (?, 5, 5, 64)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 5 * 5 * 64])
L2_flat = tf.nn.dropout(L2_flat, keep_prob=keep_prob)

# FC 7x7x64 inputs -> 10 outputs
W3 = tf.get_variable("W3", shape=[5 * 5 * 64, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
L3 = tf.nn.relu(tf.matmul(L2_flat, W3) + b)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W_hypo, b_hypo, hypothesis = make_layer_sigmoid("W4", L3, 10, 1)
hypothesis = hypothesis * 0.999998 + 0.000001
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=header.LEARNING_RATE).minimize(cost)

# Calculate accuracy
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

if HAS_SUMMARY:
    tf.summary.scalar("cost", cost)
    tf.summary.scalar("accuracy", accuracy)

print("=========== LEARNING START ===========")

step_arr = []
cost_arr = []
train_accuracy_arr = []
test_accuracy_arr = []

# initialize
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for epoch in range(header.TRAINING_EPOCHS):
        batch_xs, batch_ys = sess.run([batch_train_x, batch_train_y])
        # batch_ys = np.reshape(batch_ys, (-1, 1))
        print(epoch, " LOAD FILE BATCH DONE")
        _cost, _opt, _accuracy = sess.run([cost, optimizer, accuracy], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})

        cost_arr.append(_cost)
        step_arr.append(epoch)
        train_accuracy_arr.append(_accuracy)
        test_accuracy = sess.run(accuracy, feed_dict={X: test_x_data, Y: test_y_data, keep_prob: 1.0})
        test_accuracy_arr.append(test_accuracy)

        print('cost = ', _cost)
        print('accuracy = ', _accuracy)
        assert (_cost == _cost)  # check nan

    print("Learning finished\n")

    # print("\nTrain Accracy : ", sess.run(accuracy, feed_dict={X: train_x_data, Y: train_y_data, keep_prob: 1.0}))
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: test_x_data, Y: test_y_data, keep_prob: 1.0})
    print("\nAccuracy: ", a)
    now = time.time()
    print("\n\nTime : ", now - start_time)
    # write result file
    result_filename = "../result/problem2/model3/02_" + header.BUFFER_OPT + ".txt"
    os.makedirs(os.path.dirname(result_filename), exist_ok=True)
    result = open(result_filename, 'w')
    result.write("%f\n" % a)
    for item1, item2 in zip(h, c):
        result.write("%s %s\n" % (item1[0], item2[0]))

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(step_arr, cost_arr)
ax2.plot(step_arr, train_accuracy_arr)
ax2.plot(step_arr, test_accuracy_arr)

ax1.set_xlabel('cost')
ax2.set_ylabel('accuracy')

plt.show()



