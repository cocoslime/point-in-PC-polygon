import matplotlib.pyplot as plt
import random
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
import scipy.stats as stats
import csv

POINTS_PER_LINE = 40
max_buffer_dist = 0.05
SIGMA = 0.1
NUMBER = 500

plt.gcf().clear()
for data_index in range(NUMBER):
    polygons_csv = open("../data/problem2/polygon/" + str(data_index) + ".csv", newline='')
    polygons_reader = csv.reader(polygons_csv, quoting=csv.QUOTE_NONNUMERIC)

    x_data = []
    y_data = []
    for index, row in enumerate(polygons_reader):
        x_data.append(row[0])
        y_data.append(row[1])

    x_data.append(x_data[0])
    y_data.append(y_data[0])

    m_values = []
    b_values = []
    c_values = []

    for i in range(len(x_data) - 1):
        x_coords = [x_data[i], x_data[i + 1]]
        y_coords = [y_data[i], y_data[i+1]]
        assert ((x_coords[0] != x_coords[1]) | (y_coords[0] != y_coords[1]))
        # x = c
        if x_coords[0] == x_coords[1]:
            m_values.append(0)
            b_values.append(-1)
            c_values.append(x_coords[0])
            continue
        A = vstack([x_coords, ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        b_values.append(0)
        m_values.append(m)
        c_values.append(c)

    x_pc = []
    y_pc = []
    for step in range(POINTS_PER_LINE * (len(x_data) - 1)):
        r_index = random.randrange(0, len(x_data) - 1)
        x_coords = [x_data[r_index], x_data[r_index+1]]
        y_coords = [y_data[r_index], y_data[r_index+1]]
        x_random = np.random.uniform(min(x_coords) - max_buffer_dist, max(x_coords) + max_buffer_dist, 1)[0]

        y_random = 0

        if b_values[r_index] == -1:
            low = min(y_coords) - max_buffer_dist
            upp = max(y_coords) + max_buffer_dist
            y_random = np.random.uniform(low, upp, 1)
        else:
            m = m_values[r_index]
            c = c_values[r_index]
            y_target = m * x_random + c
            low = y_target - max_buffer_dist
            upp = y_target + max_buffer_dist
            # if upp > max(y_coords) + max_buffer_dist:
            #     upp = max(y_coords) + max_buffer_dist
            # if low < min(y_coords) - max_buffer_dist:
            #     low = min(y_coords) - max_buffer_dist
            y_random = np.random.normal(y_target, SIGMA, 1)[0]

        x_pc.append(x_random)
        y_pc.append(y_random)


    # plt.plot(x_pc, y_pc, 'o')
    # plt.plot(x_data, y_data)
    # plt.show()