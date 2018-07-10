import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

NUM_SIDES = list(range(3, 10))
POLYGON_NUMBER = 30000
WHOLE_RANGE = [-5.0, 5., -5., 5.]
WIDTH_NUM = 32
HEIGHT_NUM = 32
DATA_DIR = "../data/problem2/simple/"


def draw_raster_row(row):
    label = row[-1]
    data = np.reshape(row[1:-1], [WIDTH_NUM, HEIGHT_NUM])
    if label > 0.5:
        print("IN")
        cmap = colors.ListedColormap(['white', 'green', 'blue'])
    else:
        print("OUT")
        cmap = colors.ListedColormap(['white', 'green', 'red'])

    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.imshow(data, cmap=cmap, norm=norm)

    # draw gridlines
    ax1.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax1.set_xticks(np.arange(WHOLE_RANGE[0], WHOLE_RANGE[1], WIDTH_NUM))
    ax1.set_yticks(np.arange(WHOLE_RANGE[2], WHOLE_RANGE[3], HEIGHT_NUM))

    plt.show()