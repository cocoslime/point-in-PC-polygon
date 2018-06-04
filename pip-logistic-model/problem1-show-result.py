import matplotlib.pyplot as plt
import numpy as np
import csv

PROBLEM_INDEX = 1
DATASET_INDEX = 2
RESULT_FILE = "problem1-05"

DATA_PATH = "../data/problem"+str(PROBLEM_INDEX)+"/"
TEST_FILE_PATH = DATA_PATH + "test_" + str(DATASET_INDEX) + ".csv"
RESULT_FILE_PATH = "../result/problem" + str(PROBLEM_INDEX) + "/data_" + str(DATASET_INDEX) + "/" + RESULT_FILE + ".txt"

test_data = np.loadtxt(TEST_FILE_PATH, delimiter=',', dtype=np.float32, skiprows=1)
test_data_x = test_data[:, 0:1]
test_data_y = test_data[:, 1:2]

# train_data = np.loadtxt(DATA_PATH + "training_" + str(DATASET_INDEX) + ".csv", delimiter=',',  dtype=np.float32, skiprows=1)
# train_data_x = train_data[:, 0:1]
# train_data_y = train_data[:, 1:2]
# train_data_label = train_data[:, 2:3]

# result file
print("Accuraccy : " + open(RESULT_FILE_PATH, 'r').readline());
result_data = np.loadtxt(RESULT_FILE_PATH, dtype=np.float32, skiprows=1)
# result_data = result_data.reshape(result_data.shape[0], 1)

# polygon
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