import os
from loaddata import *
from func1 import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DENS_ARR = [50, 75, 100]
for DENSITY_OPT in DENS_ARR:
    DENSITY_OPT = "p" + str(DENSITY_OPT)
    tf.reset_default_graph()

    PIXEL = 32

    TEST_EPOCHS = 100
    BATCH_SIZE = 100

    # DENSITY_OPT = "p100"
    BUFFER_OPT = "001"
    DATA_DIR = "../data/problem2/non_convex/raster_pc/"

    TEST_FILEPATH = DATA_DIR + DENSITY_OPT + "_test_" + BUFFER_OPT + ".csv"
    SAVER_FILEPATH = "../tmp/problem2/everycase/model.ckpt"

    record_defaults = [[0.]] * (PIXEL * PIXEL + 2)

    test_xy_data = make_decode_CSV_list([TEST_FILEPATH], record_defaults)

    print("=========== BATCH - TEST ===========")

    test_x_data = test_xy_data[1:-1]
    test_y_data = test_xy_data[-1]
    test_y_data = tf.reshape(test_y_data, [1])
    test_index_data = test_xy_data[0]

    batch_test_x, batch_test_y, batch_test_index = tf.train.batch([test_x_data, test_y_data, test_index_data], enqueue_many=False,
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

    # Calculate accuracy
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print("RESTORE VARIABLE")
        saver.restore(sess, SAVER_FILEPATH)

        # Evaluation
        result = np.zeros((2, 2))  # [True condition][Predicted condition]

        total_accuracy = 0
        for epoch in range(TEST_EPOCHS):
            batch_xs, batch_ys, batch_index = sess.run([batch_test_x, batch_test_y, batch_test_index])
            _h, _c, _a = sess.run([hypothesis, predicted, accuracy],
                                  feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0})
            for index, value in enumerate(batch_ys):
                result[int(value[0])][int(_c[index][0])] += 1

            total_accuracy += _a
        total_accuracy /= TEST_EPOCHS

        print(DENSITY_OPT, BUFFER_OPT)
        print("\nAccuracy: ", total_accuracy)
        print(result)

        coord.request_stop()
        coord.join(threads)
        sess.close()



# polygon
# for test_i in TEST_TARGET:
#     data = raster_test_x_data[test_i]
#     data = np.reshape(data, [header.WIDTH_NUM, header.HEIGHT_NUM])
#     result = result_data[test_i]
#     if result[1] > 0.5:
#         print("IN")
#         cmap = colors.ListedColormap(['white', 'green', 'blue'])
#     else:
#         print("OUT")
#         cmap = colors.ListedColormap(['white', 'green', 'red'])
#
#     # create discrete colormap
#
#     bounds = [-0.5, 0.5, 1.5, 2.5]
#     norm = colors.BoundaryNorm(bounds, cmap.N)
#
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1,2,1)
#     ax1.imshow(data, cmap=cmap, norm=norm)
#
#     # draw gridlines
#     ax1.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
#     ax1.set_xticks(np.arange(WHOLE_RANGE[0], WHOLE_RANGE[1], WIDTH_NUM))
#     ax1.set_yticks(np.arange(WHOLE_RANGE[2], WHOLE_RANGE[3], HEIGHT_NUM))
#
#     # draw vector data
#     coords_x = []
#     coords_y = []
#
#     for index, row in enumerate(vector_test_x_data[test_i]):
#         if index <= 0:
#             continue
#         coords_x.append(row[0])
#         coords_y.append(row[1])
#
#     ax2 = fig.add_subplot(1,2,2)
#     ax2.scatter(coords_x, coords_y, c='b', alpha=.4)
#     if result[1] > 0.5:
#         ax2.scatter(vector_test_x_data[test_i][0][0], vector_test_x_data[test_i][0][1], s=100, c='g')
#     else:
#         ax2.scatter(vector_test_x_data[test_i][0][0], vector_test_x_data[test_i][0][1], s=100, c='r')
#
#     plt.show()