import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def make_point_list(polygon, rand_range, point_num):
    pc = []
    for step in range(point_num):
        x_random = float("{0:.3f}".format(np.random.uniform(rand_range[0], rand_range[1])))
        y_random = float("{0:.3f}".format(np.random.uniform(rand_range[2], rand_range[3])))
        point = Point(x_random, y_random)
        is_in = int(polygon.contains(point))
        pc.append([x_random, y_random, is_in])
    return pc