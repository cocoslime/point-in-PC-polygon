import time
import os
from loaddata import *
from func1 import *
import importlib
import random
from pathlib import Path

models = __import__("problem2-models")
importer = __import__("problem2-import")

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.set_random_seed(777)  # reproducibility
tf.reset_default_graph()

PIXEL = 32

LEARNING_RATE = 0.001
TRAINING_EPOCHS = 1000

MIN_AFTER_DEQUEUE = 10000
BATCH_SIZE = 100
CAPACITY = MIN_AFTER_DEQUEUE + BATCH_SIZE * 10

TEST_EPOCHS = 100

BUFFER_OPT = "001"
DATA_DIR = "../data/problem2/non_convex/raster_pc/"

TRAINING_FILE_ARR, TEST_FILE_ARR = importer.load_density_dir(DATA_DIR, "density", BUFFER_OPT)

# load_points_dir()

SAVER_FILEPATH = "../tmp/problem2/model2/model.ckpt"

record_defaults = [[0.]] * (PIXEL * PIXEL + 3)

test_xy_data = make_decode_CSV_list(TEST_FILE_ARR, record_defaults)

print("=========== BATCH - TEST ===========")

test_x_data = test_xy_data[2:-1]
test_y_data = test_xy_data[-1]
test_y_data = tf.reshape(test_y_data, [1])
test_index_data = test_xy_data[0]
test_density_data = test_xy_data[1]

# batch_test_x, batch_test_y, batch_test_index = tf.train.batch([test_x_data, test_y_data, test_index_data], enqueue_many=False, batch_size=BATCH_SIZE, num_threads=8)
batch_test_x, batch_test_y, batch_test_index = tf.train.shuffle_batch([test_x_data, test_y_data, test_index_data], min_after_dequeue=MIN_AFTER_DEQUEUE, capacity=CAPACITY, enqueue_many=False, batch_size=BATCH_SIZE, num_threads=8)

train_xy_data = make_decode_CSV_list(TRAINING_FILE_ARR, record_defaults)

print("=========== BATCH - TRAIN ===========")

train_x_data = train_xy_data[2:-1]
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

hypothesis = models.model2(X_img, keep_prob)
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

    if os.path.isfile(SAVER_FILEPATH):
        print("RESTORE VARIABLE")
        saver.restore(sess, SAVER_FILEPATH)
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
    result = np.zeros((2, 2))  # [][Predicted condition]

    total_accuracy = 0
    for epoch in range(TEST_EPOCHS):
        batch_xs, batch_ys, batch_index = sess.run([batch_test_x, batch_test_y, batch_test_index])
        _h, _c, _a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0})
        for index, value in enumerate(batch_ys):
            result[int(value[0])][int(_c[index][0])] += 1
        total_accuracy += _a

    total_accuracy /= TEST_EPOCHS

    print("\nAccuracy: ", total_accuracy)
    print(result)

    os.makedirs(os.path.dirname(SAVER_FILEPATH), exist_ok=True)
    save_path = saver.save(sess, SAVER_FILEPATH)

    coord.request_stop()
    coord.join(threads)
    sess.close()




