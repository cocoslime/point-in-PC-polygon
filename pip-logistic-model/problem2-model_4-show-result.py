import matplotlib.pyplot as plt
import numpy as np
import csv
from func2 import *
import matplotlib.cm as cm
from matplotlib import colors

WIDTH_NUM = 20
HEIGHT_NUM = 20

TEST_NUM = 1000
TEST_TARGET = list(range(1, 1000))
NUM_SIDES = list(range(3, 6))

WHOLE_RANGE = [-5.0, 5., -5., 5.]

BUFFER_OPT = "buffer_001"
RASTER_DATA_DIR = "../data/problem2/simple/raster_pc/" + BUFFER_OPT
VECTOR_DATA_DIR = "../data/problem2/simple/vector_pc/" + BUFFER_OPT
RESULT_FILE = "problem2/model4/02_" + BUFFER_OPT + ".txt"
RESULT_FILE_PATH = "../result/" + RESULT_FILE

# load_raster
def load_raster_data_in_array(file_path):
    x_data = []
    y_data = []
    convex_rate = []
    num_sides = []
    csvfile = open(file_path, newline='')
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for index, row in enumerate(reader):
        i_x_data = row[0:-3]
        i_y_data = [row[-3]]
        i_c_data = row[-2]
        i_sides_data = row[-1]

        x_data.append(i_x_data)
        y_data.append(i_y_data)
        convex_rate.append(i_c_data)
        num_sides.append(i_sides_data)

    return x_data, y_data, convex_rate, num_sides


raster_test_x_data, raster_test_y_data, test_convex_rate, test_num_sides = \
        load_raster_data_in_array(RASTER_DATA_DIR + "/test.csv")
vector_test_x_data, vector_test_y_data, _ = load_vector_data([VECTOR_DATA_DIR + "/test.csv"], 5000)

tf.reset_default_graph()

# result file
print("Accuraccy : " + open(RESULT_FILE_PATH, 'r').readline())
result_data = np.loadtxt(RESULT_FILE_PATH, dtype=np.float32, skiprows=1)

# polygon
for test_i in TEST_TARGET:
    data = raster_test_x_data[test_i]
    data = np.reshape(data, [20, 20])
    result = result_data[test_i]
    if result[1] > 0.5:
        print("IN")
        cmap = colors.ListedColormap(['white', 'green', 'blue'])
    else:
        print("OUT")
        cmap = colors.ListedColormap(['white', 'green', 'red'])

    # create discrete colormap

    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(data, cmap=cmap, norm=norm)

    # draw gridlines
    ax1.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax1.set_xticks(np.arange(WHOLE_RANGE[0], WHOLE_RANGE[1], WIDTH_NUM))
    ax1.set_yticks(np.arange(WHOLE_RANGE[2], WHOLE_RANGE[3], HEIGHT_NUM))

    # draw vector data
    coords_x = []
    coords_y = []

    for index, row in enumerate(vector_test_x_data[test_i]):
        if index <= 0:
            continue
        coords_x.append(row[0])
        coords_y.append(row[1])

    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(coords_x, coords_y, c='b', alpha=.4)
    if result[1] > 0.5:
        ax2.scatter(vector_test_x_data[test_i][0][0], vector_test_x_data[test_i][0][1], s=100, c='g')
    else:
        ax2.scatter(vector_test_x_data[test_i][0][0], vector_test_x_data[test_i][0][1], s=100, c='r')

    plt.show()