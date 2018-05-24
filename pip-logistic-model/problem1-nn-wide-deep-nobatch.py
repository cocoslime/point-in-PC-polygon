import tensorflow as tf
import numpy as np
import random
tf.set_random_seed(777)  # for reproducibility

BATCH_SIZE = 32
N_FEATURES = 2
LEARNING_RATE = 0.5
HIDDEN_LAYER_INPUTS = [4, 4]

index = 0
file_name = "../data/problem1/points_" + str(index) + ".csv"
points_file = open(file_name, "r")
data_num = int(points_file.readline())

# for test
xy = np.loadtxt(file_name, delimiter=',', dtype=np.float32, skiprows=1)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, [None, N_FEATURES], name='x-input')
Y = tf.placeholder(tf.float32, [None, 1], name='y-input')

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([N_FEATURES, HIDDEN_LAYER_INPUTS[0]]), name='weight1')
    b1 = tf.Variable(tf.random_normal([HIDDEN_LAYER_INPUTS[0]]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist = tf.summary.histogram("biases1", b1)
    layer1_hist = tf.summary.histogram("layer1", layer1)

with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([HIDDEN_LAYER_INPUTS[0], HIDDEN_LAYER_INPUTS[1]]), name='weight2')
    b2 = tf.Variable(tf.random_normal([HIDDEN_LAYER_INPUTS[1]]), name='bias2')
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    w2_hist = tf.summary.histogram("weights2", W2)
    b2_hist = tf.summary.histogram("biases2", b2)
    layer2_hist = tf.summary.histogram("layer2", layer2)

with tf.name_scope("hypothesis") as scope:
    Wh = tf.Variable(tf.random_normal([HIDDEN_LAYER_INPUTS[1], 1]), name='weight_h')
    bh = tf.Variable(tf.random_normal([1]), name='bias_h')
    hypothesis = tf.sigmoid(tf.matmul(layer2, Wh) + bh)

    wh_hist = tf.summary.histogram("weights_h", Wh)
    bh_hist = tf.summary.histogram("biases_h", bh)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Calculate accuracy
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
    # tensorboard --logdir=./logs/xor_logs
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("../logs/problem1_r05_h44_d3")
    writer.add_graph(sess.graph)  # Show the graph

    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        summary, _ = sess.run([merged_summary, optimizer], feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, global_step=step)

        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={
                  X: x_data, Y: y_data}), sess.run([W1, W2]))

    print("Learning finished")

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print('\nHypothesis: %d', h)
    print("\nCorrect (Y): ", c, "\nAccuracy: ", a)


