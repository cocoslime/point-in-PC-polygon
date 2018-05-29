import tensorflow as tf
import numpy as np


def make_layer_sigmoid(name, input_layer, input_num, output_num):
    w = tf.get_variable(name=name, shape=[input_num, output_num], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([output_num]))
    layer = tf.sigmoid(tf.matmul(input_layer, w) + b)
    return w, b, layer


def make_layer_relu(name, input_layer, input_num, output_num):
    w = tf.get_variable(name=name, shape=[input_num, output_num], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([output_num]))
    layer = tf.nn.relu(tf.matmul(input_layer, w) + b)
    return w, b, layer


def make_layer_simple(name, input_layer, input_num, output_num):
    w = tf.get_variable(name=name, shape=[input_num, output_num], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([output_num]))
    layer = tf.nn.leaky_relu(tf.matmul(input_layer, w) + b)
    return w, b, layer


def make_summary(w, b, layer, _id):
    w_hist = tf.summary.histogram("weights" + _id, w)
    b_hist = tf.summary.histogram("biases" + _id, b)
    layer_hist = tf.summary.histogram("layer" + _id, layer)
    return w_hist, b_hist, layer_hist
