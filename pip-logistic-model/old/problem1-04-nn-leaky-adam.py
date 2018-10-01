import random
import shutil
import os
from func import *

tf.set_random_seed(777)  # for reproducibility

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
HIDDEN_LAYER_NUM = 8
ONE_HIDDEN_LAYER_INPUT = 20
HIDDEN_LAYER_INPUTS = [ONE_HIDDEN_LAYER_INPUT] * (HIDDEN_LAYER_NUM + 1)
OPT_NAME = "r00001_hl" + str(HIDDEN_LAYER_NUM) + "_" + str(ONE_HIDDEN_LAYER_INPUT) + "_adam_leaky"
MAX_STEP = 30001
PROJECT_FOLDER = "../"
DATASET_NUM = 3

tf.reset_default_graph()

for index in range(0, DATASET_NUM):
    if index != 2:
        continue
    training_file_name = PROJECT_FOLDER + "data/problem1/training_" + str(index) + ".csv"
    test_file_name = PROJECT_FOLDER + "data/problem1/test_" + str(index) + ".csv"

    xy = np.loadtxt(training_file_name, delimiter=',', dtype=np.float32, skiprows=1)
    x_data = xy[:, 0:-1]
    y_data = xy[:, [-1]]
    feature_num = x_data.shape[1]

    X = tf.placeholder(tf.float32, [None, feature_num], name='x-input')
    Y = tf.placeholder(tf.float32, [None, 1], name='y-input')

    layer_parameters = []
    with tf.name_scope("inputlayer") as scope:
        W_input, b_input, layer_input = make_layer_simple(X, feature_num, HIDDEN_LAYER_INPUTS[0], "_input")
        layer_parameters.append([W_input, b_input, layer_input])
        w_input_hist, b_input_hist, layer_input_hist = make_summary(W_input, b_input, layer_input, "_input")

    # Hidden Layer
    for hd_index in range(0, HIDDEN_LAYER_NUM):
        with tf.name_scope("layer" + str(hd_index)) as scope:
            W, b, layer = make_layer_simple(layer_parameters[-1][2], HIDDEN_LAYER_INPUTS[hd_index],
                                          HIDDEN_LAYER_INPUTS[hd_index+1], str(hd_index))
            W_hist, b_hist, layer_hist = make_summary(W, b, layer, str(hd_index))

    with tf.name_scope("hypothesis") as scope:
        W_hypo, b_hypo, hypothesis = make_layer_sigmoid(layer_parameters[-1][2], HIDDEN_LAYER_INPUTS[-1], 1, "_hypo")
        w_hypo_hist, b_hypo_hist, hypothesis_hist = make_summary(W_hypo, b_hypo, hypothesis, "_hypo")

    with tf.name_scope("cost") as scope:
        cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
        tf.summary.scalar("cost", cost)

    with tf.name_scope("train") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Calculate accuracy
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
    accuracy_summ = tf.summary.scalar("accuracy", accuracy)

    with tf.Session() as sess:
        # tensorboard --logdir=./logs/xor_logs
        merged_summary = tf.summary.merge_all()
        log_file_path = PROJECT_FOLDER + "logs/problem1/data_" + str(index) + "/" + OPT_NAME
        if os.path.exists(log_file_path):
            shutil.rmtree(log_file_path)
        writer = tf.summary.FileWriter(log_file_path)
        writer.add_graph(sess.graph)  # Show the graph

        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())

        last_cost = -1.0

        for step in range(MAX_STEP):
            summary, _ = sess.run([merged_summary, optimizer], feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, global_step=step)

            if step % 1000 == 0:
                curr_cost = sess.run(cost, feed_dict={X: x_data, Y: y_data});
                print(step, curr_cost)
                # cost convergence
                # if abs(last_cost - curr_cost) < 0.005:
                    # break
                # result with train data has 99% accuracy
                if sess.run(accuracy, feed_dict={X: x_data, Y: y_data}) > 0.99:
                    break
                last_cost = curr_cost

        print("%d, Learning finished\n" % step)
        tf.summary.FileWriterCache.clear()

        # Accuracy report
        # for test
        test_xy = np.loadtxt(test_file_name, delimiter=',', dtype=np.float32, skiprows=1)
        test_x_data = test_xy[:, 0:-1]
        test_y_data = test_xy[:, [-1]]

        h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: test_x_data, Y: test_y_data})
        # print('\nHypothesis: %d', h)
        # print("\nCorrect (Y): ", c)
        print("\nAccuracy: ", a)

        result_filename = PROJECT_FOLDER + "/result/problem1/data_" + str(index) + "/" + OPT_NAME + '.txt'
        os.makedirs(os.path.dirname(result_filename), exist_ok=True)
        result = open(result_filename, 'w')
        result.write("%f\n" % a)
        for item1, item2 in zip(h, c):
            result.write("%s %s\n" % (item1[0], item2[0]))

    print(str(index) + " : Done")
    tf.reset_default_graph()

