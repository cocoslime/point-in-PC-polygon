import tensorflow as tf
import numpy as np
import random
tf.set_random_seed(777)  # for reproducibility

BATCH_SIZE = 10
N_FEATURES = 2

index = 0
file_name = "../data/problem1/points_" + str(index) + ".csv"

# for test
xy = np.loadtxt(file_name, delimiter=',', dtype=np.float32, skiprows=1)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

points_file = open(file_name, "r")
DATA_SIZE = int(points_file.readline())

filename_queue = tf.train.string_input_producer([file_name], shuffle = False, name='filename_queue')

reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

record_defaults = [[0.0], [0.0], [0]]
data = tf.decode_csv(value, record_defaults=record_defaults, field_delim=',')

train_x_batch, train_y_batch = tf.train.batch([data[0:-1], data[-1:]], batch_size=BATCH_SIZE)

X = tf.placeholder(tf.float32, [None, N_FEATURES])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_normal([N_FEATURES, 1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
is_correct = tf.equal(predicted, tf.cast(Y > 0.5, dtype=tf.float32))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(DATA_SIZE / BATCH_SIZE)

        for i in range(total_batch):
            x_batch, y_batch = sess.run([train_x_batch, train_y_batch])

            c, _ = sess.run([cost, optimizer], feed_dict={
                            X: x_batch, Y: y_batch})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test the model using test sets

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print('\nHypothesis: %d', h)
    print("\nCorrect (Y): ", c, "\nAccuracy: ", a)

    coord.request_stop()
    coord.join(threads)

