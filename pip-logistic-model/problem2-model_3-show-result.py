import matplotlib.pyplot as plt
import numpy as np
import csv
from func2 import *
import matplotlib.cm as cm
from matplotlib import colors

WIDTH_NUM = 20
HEIGHT_NUM = 20

TEST_NUM = 1000
TEST_TARGET = list(range(5010, 5030))
NUM_SIDES = list(range(3, 6))
BUFFER_OPT = "buffer_001"
RESULT_FILE = "problem2/model3/01_" + BUFFER_OPT + ".txt"
RESULT_FILE_PATH = "../result/" + RESULT_FILE

WHOLE_RANGE = [-5.0, 5., -5., 5.]

test_x_data = []
test_y_data = []

tf.reset_default_graph()

for numSides in NUM_SIDES:
    print(str(numSides) + " =========== LOAD DATA ===========")
    _test_x_data, _test_y_data = load_raster_data("../data/problem2/raster_pc/" + BUFFER_OPT + "/test_" + str(numSides) + ".csv")
    test_x_data.extend(_test_x_data)
    test_y_data.extend(_test_y_data)

# result file
print("Accuraccy : " + open(RESULT_FILE_PATH, 'r').readline())
result_data = np.loadtxt(RESULT_FILE_PATH, dtype=np.float32, skiprows=1)
# result_data = result_data.reshape(result_data.shape[0], 1)

# polygon
for test_i in TEST_TARGET:
    coords_x = []
    coords_y = []

    

    # create discrete colormap
    cmap = colors.ListedColormap(['red', 'blue'])
    bounds = [0, 10, 20]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)

    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(WHOLE_RANGE[0], WHOLE_RANGE[1], WIDTH_NUM));
    ax.set_yticks(np.arange(WHOLE_RANGE[2], WHOLE_RANGE[3], HEIGHT_NUM));

    plt.show()