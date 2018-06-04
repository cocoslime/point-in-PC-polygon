import random
import math
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import collections
from scipy.stats import truncnorm

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
    random_start = make_random_point(rand_range)
    x_random_start = random_start.x
    y_random_start = random_start.y
    coords.append([x_random_start, y_random_start])

    while sides + 1 < num_sides:
        rand_xy = make_random_point(rand_range)
        x_random = rand_xy.x
        y_random = rand_xy.y
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


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def generate_random_point_in_rectangle(center, buffer):
    x_value = random.uniform(center[0]-buffer, center[0]+buffer)
    y_value = random.uniform(center[1]-buffer, center[1]+buffer)
    return x_value, y_value


def generate_random_point_in_circle(center, buffer):
    min_x = center[0] - buffer
    max_x = center[0] + buffer
    x_value = get_truncated_normal(center[0], buffer/3.0, min_x, max_x).rvs()
    max_y = math.sqrt(buffer * buffer -
                      (x_value - center[0]) * (x_value - center[0])) + center[1]
    min_y = -math.sqrt(buffer * buffer -
                       (x_value - center[0]) * (x_value - center[0])) + center[1]
    y_value = np.random.uniform(min_y, max_y, 1)[0]
    return x_value, y_value


def generate_points_along_sides(x_data, y_data, max_buffer_dist, equations, point_num):
    pc = []
    for step in range(point_num):
        r_index = random.randrange(0, len(equations))
        x_coords = [x_data[r_index], x_data[r_index+1]]
        y_coords = [y_data[r_index], y_data[r_index+1]]
        equ = equations[r_index]

        min_x = min(x_coords)
        max_x = max(x_coords)
        x_random = random.uniform(min_x, max_x)
        b = equ.b
        if b == 0:
            low = min(y_coords) - max_buffer_dist
            upp = max(y_coords) + max_buffer_dist
            y_random = random.uniform(low, upp)
        elif b == -1:
            y_random = equ.a * x_random + equ.c
        else:
            return
        result_x, result_y = generate_random_point_in_rectangle([x_random, y_random], max_buffer_dist)
        result_x = float("{0:.5f}".format(result_x))
        result_y = float("{0:.5f}".format(result_y))
        pc.append(PCP_Point(result_x, result_y, 0))
    return pc


def make_random_point(rand_range):
    x_random = float("{0:.5f}".format(np.random.uniform(rand_range[0], rand_range[1])))
    y_random = float("{0:.5f}".format(np.random.uniform(rand_range[2], rand_range[3])))
    point = Point(x_random, y_random)
    return point


def make_test_point_list(polygon, rand_range, point_num):
    in_pc = []
    not_in_pc = []
    label = []
    while True:
        point = make_random_point(rand_range)
        is_in = int(polygon.contains(point))
        if is_in:
            in_pc.append(PCP_Point(point.x, point.y, 1))
        else:
            not_in_pc.append(PCP_Point(point.x, point.y, 1))
        if len(in_pc) + len(not_in_pc) == point_num:
            if len(in_pc) <= 1:
                not_in_pc = []
            else:
                break
    label = [1] * len(in_pc) + [0] * len(not_in_pc)
    pc = in_pc + not_in_pc
    return pc, label
