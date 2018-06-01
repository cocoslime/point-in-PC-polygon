import matplotlib.pyplot as plt
import numpy as np
import csv
from func2 import *

TEST_NUM = 100
RESULT_FILE = "problem2"

DATA_PATH = "../data/problem2/"
test_x_data, test_y_data = load_data("../data/problem2/test_", TEST_NUM)
RESULT_FILE_PATH = "../result/" + RESULT_FILE + ".txt"

# result file
print("Accuraccy : " + open(RESULT_FILE_PATH, 'r').readline());
result_data = np.loadtxt(RESULT_FILE_PATH, dtype=np.float32, skiprows=1)
# result_data = result_data.reshape(result_data.shape[0], 1)

# polygon
for test_i in range(TEST_NUM):

polygons_csv = open(DATA_PATH + "polygons.csv", newline='')
polygons_reader = csv.reader(polygons_csv, quoting=csv.QUOTE_NONNUMERIC)
for index, row in enumerate(polygons_reader):
    if index != DATASET_INDEX:
        continue

    polygon_coords_x = []
    polygon_coords_y = []

    for i in range(0, len(row), 2):
        polygon_coords_x.append(row[i])
        polygon_coords_y.append(row[i+1])

    polygon_coords_x.append(row[0])
    polygon_coords_y.append(row[1])

    color_data = result_data[:, 0].reshape(result_data.shape[0], 1)
    plt.scatter(test_data_x, test_data_y, c=color_data)
    # plt.scatter(train_data_x, train_data_y, c=train_data_label)
    plt.plot(polygon_coords_x, polygon_coords_y)
    plt.show()

polygons_csv.close()