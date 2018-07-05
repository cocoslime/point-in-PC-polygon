import matplotlib.pyplot as plt
import numpy as np
import csv
from func2 import *
import matplotlib.cm as cm

WIDTH_NUM = 20
HEIGHT_NUM = 20

TEST_NUM = 1000
TEST_TARGET = list(range(10, 30))
NUM_SIDES = "5"
BUFFER_OPT = "buffer_001"
RESULT_FILE = "problem2/model2/" + NUM_SIDES + "_01_" + BUFFER_OPT + ".txt"

test_x_data, test_y_data, test_pid = load_vector_data(["../data/problem2/convex/vector_pc/" + BUFFER_OPT + "/test_" + NUM_SIDES + ".csv"], TEST_NUM)
grid_test_x_data = grid(test_x_data, WIDTH_NUM, HEIGHT_NUM, [-5.0, 5., -5., 5.])

RESULT_FILE_PATH = "../result/" + RESULT_FILE

# result file
print("Accuraccy : " + open(RESULT_FILE_PATH, 'r').readline())
result_data = np.loadtxt(RESULT_FILE_PATH, dtype=np.float32, skiprows=1)

# polygon data
polygons_csv = open("../data/problem2/convex/polygon/" + NUM_SIDES + ".csv", newline='')
polygons_reader = csv.reader(polygons_csv, quoting=csv.QUOTE_NONNUMERIC)

for test_i in TEST_TARGET:
    coords_x = []
    coords_y = []

    for index, row in enumerate(test_x_data[test_i]):
        if index <= 0:
            continue
        coords_x.append(row[0])
        coords_y.append(row[1])

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(coords_x, coords_y, c='b', alpha=.4)
    if result_data[test_i][0] > 0.5:
        ax1.scatter(test_x_data[test_i][0][0], test_x_data[test_i][0][1], s=100, c='g')
    else:
        ax1.scatter(test_x_data[test_i][0][0], test_x_data[test_i][0][1], s=100, c='r')

    plt.show()
