import time
import matplotlib.pyplot as plt
import shutil
import os
import tensorflow as tf
from loaddata import *
from func1 import *
import random
from pathlib import Path

DENS_ARR = [50, 75, 700]
for DENSITY_OPT in DENS_ARR:
    DENSITY_OPT = "p" + str(DENSITY_OPT)
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.set_random_seed(777)  # reproducibility
    tf.reset_default_graph()

    PIXEL = 32

    LEARNING_RATE = 0.001
    TRAINING_EPOCHS = 1000

    MIN_AFTER_DEQUEUE = 1000
    BATCH_SIZE = 10
    CAPACITY = MIN_AFTER_DEQUEUE + BATCH_SIZE * 10

    TEST_EPOCHS = 100

    # DENSITY_OPT = "p200"
    BUFFER_OPT = "001"
    DATA_DIR = "../data/problem2/non_convex/raster_pc/"

    TEST_FILEPATH = DATA_DIR + DENSITY_OPT + "_test_"+BUFFER_OPT + ".csv"
    TRAINING_FILEPATH = DATA_DIR + DENSITY_OPT + "_training_"+BUFFER_OPT + ".csv"
    SAVER_FILEPATH = "../tmp/problem2/different-density/" + DENSITY_OPT + "_" + BUFFER_OPT + "/model.ckpt"

    record_defaults = [[0.]] * (PIXEL * PIXEL + 2)

    test_xy_data = make_decode_CSV_list([TEST_FILEPATH], record_defaults)

    print("=========== BATCH - TEST ===========")

    test_x_data = test_xy_data[1:-1]
    test_y_data = test_xy_data[-1]
    test_y_data = tf.reshape(test_y_data, [1])
    test_index_data = test_xy_data[0]

    batch_test_x, batch_test_y, batch_test_index = tf.train.batch([test_x_data, test_y_data, test_index_data], enqueue_many=False,
                                                           batch_size=BATCH_SIZE, num_threads=8)
    # tf.train.shuffle_batch([test_x_data, test_y_data, test_index_data], min_after_dequeue=MIN_AFTER_DEQUEUE, capacity=CAPACITY, enqueue_many=False, batch_size=BATCH_SIZE, num_threads=8)

    train_xy_data = make_decode_CSV_list([TRAINING_FILEPATH], record_defaults)

    print("=========== BATCH - TRAIN ===========")

    train_x_data = train_xy_data[1:-1]
    train_y_data = train_xy_data[-1]
    train_y_data = tf.reshape(train_y_data, [1])
    train_index_data = train_xy_data[0]

    batch_train_x, batch_train_y, = tf.train.shuffle_batch([train_x_data, train_y_data],
                                                           min_after_dequeue=MIN_AFTER_DEQUEUE, capacity=CAPACITY, enqueue_many=False,
                                                           batch_size=BATCH_SIZE, num_threads=8)

    print("=========== BUILD GRAPH ===========")

    # input place holders
    X = tf.placeholder(tf.float32,  [None, PIXEL * PIXEL])
    X_img = tf.reshape(X, [-1, PIXEL, PIXEL, 1])
    Y = tf.placeholder(tf.float32, [None, 1])

    keep_prob = tf.placeholder(tf.float32)

    W1 = tf.get_variable("W1", shape=[3, 3, 1, 32],
                         initializer=tf.contrib.layers.xavier_initializer())
    L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
    '''
    Tensor("Conv2D:0", shape=(?, 32, 32, 32), dtype=float32)
    Tensor("Relu:0", shape=(?, 32, 32, 32), dtype=float32)
    Tensor("MaxPool:0", shape=(?, 16, 16, 32), dtype=float32)
    Tensor("dropout/mul:0", shape=(?, 16, 16, 32), dtype=float32)
    '''

    W2 = tf.get_variable("W2", shape=[3, 3, 32, 64],
                         initializer=tf.contrib.layers.xavier_initializer())
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

    W3 = tf.get_variable("W3", shape=[3, 3, 64, 128],
                         initializer=tf.contrib.layers.xavier_initializer())

    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
    '''
    '''

    W4 = tf.get_variable("W4", shape=[3, 3, 128, 128],
                         initializer=tf.contrib.layers.xavier_initializer())
    L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
    L4 = tf.nn.relu(L4)
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
    L4_flat = tf.reshape(L4, [-1, 8 * 8 * 128])


    # FC 5x5x128 inputs -> 625 outputs
    W_FC1 = tf.get_variable("W_FC1", shape=[8 * 8 * 128, 625],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([625]))
    L_FC1 = tf.nn.relu(tf.matmul(L4_flat, W_FC1) + b1)
    L_FC1 = tf.nn.dropout(L_FC1, keep_prob=keep_prob)
    '''
    Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
    Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
    '''

    W_hypo, b_hypo, hypothesis = make_layer_sigmoid("W_FC2", L_FC1, 625, 1)
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
        # saver.restore(sess, SAVER_FILEPATH)
        start_time = time.time()

        # Training
        for epoch in range(TRAINING_EPOCHS):
            batch_xs, batch_ys = sess.run([batch_train_x, batch_train_y])
            # batch_ys = np.reshape(batch_ys, (-1, 1))
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

        now = time.time()
        print("\n\nLearning Time : ", int((now - start_time) / 60), "m ",  int(now - start_time) % 60, "s")

        # Evaluation
        result = np.zeros((2, 2)) # [True condition][Predicted condition]

        total_accuracy = 0
        for epoch in range(TEST_EPOCHS):
            batch_xs, batch_ys, batch_index = sess.run([batch_test_x, batch_test_y, batch_test_index])
            _h, _c, _a = sess.run([hypothesis, predicted, accuracy],
                               feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0})
            for index, value in enumerate(batch_ys):
                result[int(value[0])][int(_c[index][0])] += 1

            total_accuracy += _a
        total_accuracy /= TEST_EPOCHS

        print("\nAccuracy: ", total_accuracy)
        print(result)

        # write result file
        # result_filename = "../result/problem2/model4/03_" + BUFFER_OPT + ".txt"
        # os.makedirs(os.path.dirname(result_filename), exist_ok=True)
        # result_file = open(result_filename, 'w')
        # result_file.write("%f\n" % a)
        # for item1, item2 in zip(h, c):
        #     result_file.write("%s %s\n" % (item1[0], item2[0]))
        os.makedirs(os.path.dirname(SAVER_FILEPATH), exist_ok=True)
        # save_path = saver.save(sess, SAVER_FILEPATH)

        coord.request_stop()
        coord.join(threads)
        sess.close()




