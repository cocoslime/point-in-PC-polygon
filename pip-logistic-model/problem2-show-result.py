import matplotlib.pyplot as plt
import numpy as np
import csv
from func2 import *
import matplotlib.cm as cm

TEST_NUM = 1000
TEST_TARGET = list(range(10, 30))
NUM_SIDES = "4"
BUFFER_OPT = "buffer_001"
RESULT_FILE = "problem2/model2/" + NUM_SIDES + "_01_" + BUFFER_OPT + ".txt"

test_x_data, test_y_data = load_data("../data/problem2/" + BUFFER_OPT + "/test_" + NUM_SIDES + "_", TEST_NUM)
RESULT_FILE_PATH = "../result/" + RESULT_FILE

# result file
print("Accuraccy : " + open(RESULT_FILE_PATH, 'r').readline());
result_data = np.loadtxt(RESULT_FILE_PATH, dtype=np.float32, skiprows=1)
# result_data = result_data.reshape(result_data.shape[0], 1)

# polygon
for test_i in TEST_TARGET:
    coords_x = []
    coords_y = []

    for index, row in enumerate(test_x_data[test_i]):
        if index <= 1:
            continue
        if index % 2 == 0:
            coords_x.append(row)
        if index % 2 == 1:
            coords_y.append(row)

    plt.scatter(coords_x, coords_y, c='b', alpha=.4)
    if result_data[test_i][0] > 0.5 :
        plt.scatter(test_x_data[test_i][0], test_x_data[test_i][1], s=100, c='g')
    else:
        plt.scatter(test_x_data[test_i][0], test_x_data[test_i][1], s=100, c='r')
    plt.show()
