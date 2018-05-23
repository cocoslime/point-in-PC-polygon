import matplotlib.pyplot as plt
import random
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
import scipy.stats as stats

POINTS_PER_LINE = 40
max_buffer_dist = 0.1
SD = 2

if __name__ == '__main__' :
    x_data = [1.0, 2.0, 3.0, 1.5, 0, 1.0]
    y_data = [0.0, 0.0, 1.0, 2., 1.0, 0.0]

    a_values = []
    b_values = []
    c_values = []

    for i in range(len(x_data) - 1):
        x_coords = [x_data[i], x_data[i + 1]]
        y_coords = [y_data[i], y_data[i+1]]
        if x_coords[0] == x_coords[1]:
            a_values.append(1)
            b_values.append(0)
            c_values.append(x_coords[0])
            continue
        if y_coords[0] == y_coords[1]:
            a_values.append(0)
            b_values.append(-1)
            c_values.append(y_coords[0])
            continue
        A = vstack([x_coords, ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords, rcond=None)[0]
        b_values.append(-1)
        a_values.append(m)
        c_values.append(c)

    x_pc = []
    y_pc = []
    for step in range(POINTS_PER_LINE * (len(x_data) - 1)):
        r_index = random.randrange(0, len(x_data) - 1)
        x_coords = [x_data[r_index], x_data[r_index+1]]
        y_coords = [y_data[r_index], y_data[r_index+1]]
        x_random = np.random.uniform(min(x_coords) - max_buffer_dist, max(x_coords) + max_buffer_dist, 1)

        y_random = 0

        if b_values[r_index] == 0:
            low = min(y_coords) - max_buffer_dist
            upp = max(y_coords) + max_buffer_dist
            y_random = np.random.uniform(low, upp, 1)
        elif b_values[r_index] == -1:
            m = a_values[r_index]
            c = c_values[r_index]
            y_target = m * x_random[0] + c
            low = y_target - max_buffer_dist
            upp = y_target + max_buffer_dist
            if upp > max(y_coords) + max_buffer_dist:
                upp = max(y_coords) + max_buffer_dist
            if low < min(y_coords) - max_buffer_dist:
                low = min(y_coords) - max_buffer_dist
            y_random = [stats.truncnorm((low - y_target) / SD, (upp - y_target) / SD, loc=y_target, scale=SD).rvs()]

        x_pc.append(x_random[0])
        y_pc.append(y_random[0])

    plt.plot(x_pc, y_pc, 'o')
    plt.plot(x_data, y_data)
    plt.show()