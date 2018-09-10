print("problem3-exturded")

import csv
import time
import matplotlib.pyplot as plt
import shutil
import os
import tensorflow as tf
from loaddata import *
from func1 import *
import random
from pathlib import Path

MIN_AFTER_DEQUEUE = 10000
BATCH_SIZE = 2
CAPACITY = MIN_AFTER_DEQUEUE + BATCH_SIZE * 10
LEARNING_RATE = 0.001

CONVEX_OPT = "convex"
TRAINING_SET = 5
TRAINING_EPOCHS = 100
TEST_EPOCH = 10

tf.set_random_seed(777)  # reproducibility

DATA_DIR = "../data/problem3/extruded/" + CONVEX_OPT + "/raster/"

VOXEL = 20

start_time = time.time()
tf.reset_default_graph()
record_defaults = [[0.]] * (VOXEL * VOXEL * VOXEL + 1)

print("=========== BATCH - TEST ===========")

test_xy_data = make_decode_CSV_list([DATA_DIR + str(VOXEL) + "/test.csv"], record_defaults)

test_x_data = test_xy_data[0:-1]
test_y_data = test_xy_data[-1]
test_y_data = tf.reshape(test_y_data, [1])

batch_test_x, batch_test_y = tf.train.batch([test_x_data, test_y_data], enqueue_many=False, batch_size=BATCH_SIZE, num_threads=8)

print("=========== BATCH - TRAIN ===========")
train_file_list = []
for i in range(TRAINING_SET):
    train_file_list.append(DATA_DIR + str(VOXEL) + "/training_" + str(i) + ".csv")
train_xy_data = make_decode_CSV_list(train_file_list, record_defaults)

train_x_data = train_xy_data[0:-1]
train_y_data = train_xy_data[-1]
train_y_data = tf.reshape(train_y_data, [1])

batch_train_x, batch_train_y, = tf.train.shuffle_batch([train_x_data, train_y_data],
min_after_dequeue=MIN_AFTER_DEQUEUE, capacity=CAPACITY, enqueue_many=False, batch_size=BATCH_SIZE, num_threads=8)
# batch_train_x, batch_train_y, = tf.train.batch([train_x_data, train_y_data], enqueue_many=False, batch_size=BATCH_SIZE, num_threads=8)

print("=========== BUILD GRAPH ===========")

# input place holders
X = tf.placeholder(tf.float32,  [None, VOXEL * VOXEL * VOXEL])
X_img = tf.reshape(X, [-1, VOXEL, VOXEL, VOXEL, 1])
Y = tf.placeholder(tf.float32, [None, 1])

keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", shape=[3, 3, 3, 1, 32],
                     initializer=tf.contrib.layers.xavier_initializer())
L1 = tf.nn.conv3d(X_img, W1, strides=[1, 1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool3d(L1, ksize=[1, 2, 2, 2, 1],
                    strides=[1, 2, 2, 2, 1], padding='SAME')

L1 = tf.nn.dropout(L1, keep_prob=keep_prob, name="dr1")
'''
Tensor("Conv3D:0", shape=(?, 20, 20, 20, 32), dtype=float32)
Tensor("MaxPool3D:0", shape=(?, 10, 10, 10, 32), dtype=float32)

'''


W2 = tf.get_variable("W2", shape=[3, 3, 3, 32, 64],
                     initializer=tf.contrib.layers.xavier_initializer())
L2 = tf.nn.conv3d(L1, W2, strides=[1, 1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool3d(L2, ksize=[1, 2, 2, 2, 1],
                    strides=[1, 2, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob, name="dr2")
'''
Tensor("Conv3D_1:0", shape=(?, 10, 10, 10, 64), dtype=float32)
Tensor("MaxPool3D_1:0", shape=(?, 5, 5, 5, 64), dtype=float32)
'''

W3 = tf.get_variable("W3", shape=[3, 3, 3, 64, 128],
                     initializer=tf.contrib.layers.xavier_initializer())
L3 = tf.nn.conv3d(L2, W3, strides=[1, 1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool3d(L3, ksize=[1, 2, 2, 2, 1],
                    strides=[1, 2, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob, name="dr3")
'''
Tensor("Conv3D_2:0", shape=(?, 5, 5, 5, 128), dtype=float32)
Tensor("MaxPool3D_2:0", shape=(?, 3, 3, 3, 128), dtype=float32)
'''

W4 = tf.get_variable("W4", shape=[3, 3, 3, 128, 128],
                     initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.conv3d(L3, W4, strides=[1, 1, 1, 1, 1], padding='SAME')

L4 = tf.nn.relu(L4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob, name="dr4")
L4_flat = tf.reshape(L4, [-1, 3 * 3 * 3 * 128])

'''
Tensor("Conv3D_3:0", shape=(?, 3, 3, 3, 128), dtype=float32)

'''

W_FC1 = tf.get_variable("W_FC1", shape=[3 * 3 * 3 * 128, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([625]))
L_FC1 = tf.nn.relu(tf.matmul(L4_flat, W_FC1) + b1)
L_FC1 = tf.nn.dropout(L_FC1, keep_prob=keep_prob)

W_FC2 = tf.get_variable("W_FC2", shape=[625, 100],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([100]))
L_FC2 = tf.nn.relu(tf.matmul(L_FC1, W_FC2) + b2)
L_FC2 = tf.nn.dropout(L_FC2, keep_prob=keep_prob)

W_hypo, b_hypo, hypothesis = make_layer_sigmoid("W_FC3", L_FC2, 100, 1)
hypothesis = hypothesis * 0.999998 + 0.000001
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Calculate accuracy
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

saver = tf.train.Saver()


print("=========== LEARNING START ===========")

step_arr = []
cost_arr = []
train_accuracy_arr = []

# initialize
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print("RESTORE VARIABLE")
    # saver.restore(sess, "../tmp/problem3-extruded/model.ckpt")

    for epoch in range(TRAINING_EPOCHS):
        batch_xs, batch_ys = sess.run([batch_train_x, batch_train_y])
        print(epoch, " LOAD FILE BATCH DONE")
        _cost, _opt, _accuracy = sess.run([cost, optimizer, accuracy], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})

        if epoch % 5 == 0:
            cost_arr.append(_cost)
            step_arr.append(epoch)
            train_accuracy_arr.append(_accuracy)
            print('cost = ', _cost)
            print('accuracy = ', _accuracy)

        assert (_cost == _cost)  # check nan

    print("Learning finished\n")

    # test
    total_accuracy = 0
    for epoch in range(TEST_EPOCH):
        batch_xs, batch_ys = sess.run([batch_test_x, batch_test_y])
        h, c, a = sess.run([hypothesis, predicted, accuracy],
                           feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0})
        total_accuracy += a
    total_accuracy /= TEST_EPOCH

    print("\nAccuracy: ", total_accuracy)
    now = time.time()
    print("\n\nTime : ", int((now - start_time) / 60), "m ",  int(now - start_time) % 60, "s")

    # write result file
    result_filename = "../result/problem3/extruded/result.txt"
    os.makedirs(os.path.dirname(result_filename), exist_ok=True)
    result = open(result_filename, 'w')
    result.write("%f\n" % total_accuracy)
    for item1, item2 in zip(h, c):
        result.write("%s %s\n" % (item1[0], item2[0]))

    os.makedirs(os.path.dirname("../tmp/problem3-extruded/model.ckpt"), exist_ok=True)
    save_path = saver.save(sess, "../tmp/problem3-extruded/model.ckpt")

    coord.request_stop()
    coord.join(threads)
    sess.close()

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(step_arr, cost_arr)
ax2.plot(step_arr, train_accuracy_arr)

ax1.set_ylabel('cost')
ax2.set_ylabel('accuracy')
ax2.set_xlabel('step')

plt.show()
