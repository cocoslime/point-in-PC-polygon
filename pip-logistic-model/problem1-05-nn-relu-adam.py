import random
import shutil
import os
from func import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.set_random_seed(777)  # for reproducibility

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
OPT_NAME = "problem1-05"
MAX_STEP = 10001

DATASET_NUM = 3

tf.reset_default_graph()

for index in range(0, DATASET_NUM):
    training_file_name = "../data/problem1/training_" + str(index) + ".csv"
    test_file_name = "../data/problem1/test_" + str(index) + ".csv"

    xy = np.loadtxt(training_file_name, delimiter=',', dtype=np.float32, skiprows=1)
    x_data = xy[:, 0:-1]
    y_data = xy[:, [-1]]
    feature_num = x_data.shape[1]

    X = tf.placeholder(tf.float32, [None, feature_num], name='x-input')
    Y = tf.placeholder(tf.float32, [None, 1], name='y-input')

    W1, b1, L1 = make_layer_relu("W1", X, feature_num, 10)

    # Hidden Layer
    W2, b2, L2 = make_layer_relu("W2", L1, 10, 10)
    W3, b3, L3 = make_layer_relu("W3", L2, 10, 10)
    W4, b4, L4 = make_layer_relu("W4", L3, 10, 10)

    W_hypo, b_hypo, hypothesis = make_layer_sigmoid("W5", L4, 10, 1)
    hypothesis = hypothesis * 0.998 + 0.001
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    tf.summary.scalar("cost", cost)

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Calculate accuracy
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
    accuracy_summ = tf.summary.scalar("accuracy", accuracy)

    print("\n\n ======= " + str(index) + " Training Start =======\n\n")

    with tf.Session() as sess:
        # tensorboard --logdir=./logs/xor_logs
        merged_summary = tf.summary.merge_all()
        log_file_path = "../logs/problem1/data_" + str(index) + "/" + OPT_NAME
        if os.path.exists(log_file_path):
            shutil.rmtree(log_file_path)
        writer = tf.summary.FileWriter(log_file_path)
        writer.add_graph(sess.graph)  # Show the graph

        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())

        for step in range(MAX_STEP):
            summary, _ = sess.run([merged_summary, optimizer], feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, global_step=step)

            if step % 1000 == 0:
                curr_cost = sess.run(cost, feed_dict={X: x_data, Y: y_data})
                print(step, curr_cost)
                assert (curr_cost == curr_cost)

                # result with train data has 99% accuracy
                # if sess.run(accuracy, feed_dict={X: x_data, Y: y_data}) > 0.99:
                #     break

        print("Learning finished\n")
        tf.summary.FileWriterCache.clear()
        writer.close()

        # Accuracy report
        # for test
        test_xy = np.loadtxt(test_file_name, delimiter=',', dtype=np.float32, skiprows=1)
        test_x_data = test_xy[:, 0:-1]
        test_y_data = test_xy[:, [-1]]

        h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: test_x_data, Y: test_y_data})
        # print('\nHypothesis: %d', h)
        # print("\nCorrect (Y): ", c)
        print("\nAccuracy: ", a)

        result_filename = "../result/problem1/data_" + str(index) + "/" + OPT_NAME + '.txt'
        os.makedirs(os.path.dirname(result_filename), exist_ok=True)
        result = open(result_filename, 'w')
        result.write("%f\n" % a)
        for item1, item2 in zip(h, c):
            result.write("%s %s\n" % (item1[0], item2[0]))

    print(str(index) + " : Done")
    tf.reset_default_graph()

