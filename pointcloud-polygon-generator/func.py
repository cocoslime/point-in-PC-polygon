import random
import math
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import collections
from numpy.linalg import lstsq
from numpy import ones, vstack

# ================== Problem 2 ===================

Equation = collections.namedtuple('Equation', 'a b c')  # ax + by + c = 0
PCP_Point = collections.namedtuple('PCP_Point', 'x y type') # type -  0 : boundary, 1 : test


def make_equation_list(x_data,  y_data):
    equations = []
    for i in range(len(x_data) - 1):
        x_coords = [x_data[i], x_data[i + 1]]
        y_coords = [y_data[i], y_data[i+1]]
        equ = make_equation(x_coords, y_coords)
        equations.append(equ)
    return equations


def make_equation(x_coords, y_coords):
    assert ((x_coords[0] != x_coords[1]) | (y_coords[0] != y_coords[1]))
    # x = c
    if x_coords[0] == x_coords[1]:
        equ = Equation(1, 0, -x_coords[0])
        return equ

    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0] # y = mx + c
    equ = Equation(m, -1, c)
    return equ


def make_random_polygon(num_sides, rand_range):
    sides = 0
    coords = []
    x_random_start = float("{0:.3f}".format(np.random.uniform(rand_range[0], rand_range[1])))
    y_random_start = float("{0:.3f}".format(np.random.uniform(rand_range[2], rand_range[3])))
    coords.append([x_random_start, y_random_start])

    while sides + 1 < num_sides:
        x_random = float("{0:.3f}".format(np.random.uniform(rand_range[0], rand_range[1])))
        y_random = float("{0:.3f}".format(np.random.uniform(rand_range[2], rand_range[3])))
        coords.append([x_random, y_random])
        sides += 1

    polygon = Polygon(coords)
    if polygon.is_simple:
        return coords
    else:
        print("one more random")
        return make_random_polygon(num_sides, rand_range)


def create_polygon(x_coords, y_coords):
    xy = zip(x_coords, y_coords)
    return Polygon(xy)


def generate_points_along_sides(x_data, y_data, max_buffer_dist, equations, point_num):
    pc = []
    for step in range(point_num):
        r_index = random.randrange(0, len(equations))
        x_coords = [x_data[r_index], x_data[r_index+1]]
        y_coords = [y_data[r_index], y_data[r_index+1]]
        equ = equations[r_index]

        min_x = min(x_coords)
        max_x = max(x_coords)
        x_random = np.random.uniform(min_x - max_buffer_dist, max_x + max_buffer_dist, 1)[0]
        b = equ.b
        if b == 0:
            low = min(y_coords) - max_buffer_dist
            upp = max(y_coords) + max_buffer_dist
            y_random = np.random.uniform(low, upp, 1)
        elif b == -1:
            y_min_x = equ.a * min_x + equ.c
            y_max_x = equ.a * max_x + equ.c
            if x_random < min_x:
                max_y = math.sqrt(max_buffer_dist * max_buffer_dist - (x_random - min_x) * (x_random - min_x)) + y_min_x
                min_y = -math.sqrt(max_buffer_dist * max_buffer_dist - (x_random - min_x) * (x_random - min_x)) + y_min_x
            elif x_random > max_x:
                max_y = math.sqrt(max_buffer_dist * max_buffer_dist - (x_random - max_x) * (x_random - max_x)) + y_max_x
                min_y = -math.sqrt(max_buffer_dist * max_buffer_dist - (x_random - max_x) * (x_random - max_x)) + y_max_x
            else:
                y_target = equ.a * x_random + equ.c
                max_y = y_target + max_buffer_dist
                min_y = y_target - max_buffer_dist
            y_random = np.random.uniform(min_y, max_y, 1)[0]
        else:
            return
        pc.append(PCP_Point(x_random, y_random, 0))
    return pc


def make_test_point_list(polygon, rand_range, point_num):
    is_in_num = 0
    pc = []
    label = []
    while len(pc) < point_num:
        x_random = float("{0:.3f}".format(np.random.uniform(rand_range[0], rand_range[1])))
        y_random = float("{0:.3f}".format(np.random.uniform(rand_range[2], rand_range[3])))
        point = Point(x_random, y_random)
        is_in = int(polygon.contains(point))
        is_in_num += is_in
        pc.append(PCP_Point(x_random, y_random, 1))
        label.append(is_in)
    if (is_in_num == 5) | (is_in_num == 0):
        return make_test_point_list(polygon, rand_range, point_num)
    return pc, label
