import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def make_random_polygon(numSides, rand_range):
    sides = 0
    coords = []
    x_random_start = float("{0:.3f}".format(np.random.uniform(rand_range[0], rand_range[1])))
    y_random_start = float("{0:.3f}".format(np.random.uniform(rand_range[2], rand_range[3])))
    coords.append([x_random_start, y_random_start])

    while sides + 1 < numSides:
        x_random = float("{0:.3f}".format(np.random.uniform(rand_range[0], rand_range[1])))
        y_random = float("{0:.3f}".format(np.random.uniform(rand_range[2], rand_range[3])))
        coords.append([x_random, y_random])
        sides += 1

    polygon = Polygon(coords)
    if polygon.is_simple:
        return coords
    else:
        print("one more random")
        return make_random_polygon(numSides, rand_range)


def make_point_list(polygon, rand_range, point_num):
    pc = []
    for step in range(point_num):
        x_random = float("{0:.3f}".format(np.random.uniform(rand_range[0], rand_range[1])))
        y_random = float("{0:.3f}".format(np.random.uniform(rand_range[2], rand_range[3])))
        point = Point(x_random, y_random)
        is_in = int(polygon.contains(point))
        pc.append([x_random, y_random, is_in])
    return pc