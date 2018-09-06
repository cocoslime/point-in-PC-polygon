import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing
from shapely.geometry import LineString
import random
import numpy as np
import csv
from func_2 import *
import scipy.stats as stats
import os
header = __import__("problem2-header")

index = 0

polygon_csv = open("../data/problem2/non_convex/" + "polygon.csv",
                   'w', encoding='utf-8', newline='')
polygon_writer = csv.writer(polygon_csv)

# To keep convex ratio balance
cr_num_arr = np.zeros(21)

while index < header.POLYGON_NUMBER:
    numSides = random.randrange(header.NUM_SIDES[0], header.NUM_SIDES[-1])
    polygon_coords = make_random_point_list(numSides, header.WHOLE_RANGE)
    polygon = Polygon(polygon_coords)
    ring = LinearRing(polygon_coords)
    if not ring.is_simple:
        continue

    convex_hull = polygon.convex_hull
    if convex_hull.area < 2.0:
        continue

    convex_hull_ratio = float(polygon.area) / float(convex_hull.area)
    if convex_hull_ratio < 0.5:
        continue

    cr_index = int(convex_hull_ratio / 0.05)
    if cr_num_arr[cr_index] >= 5000:
        continue
    cr_num_arr[cr_index] += 1

    '''
    convex_ratio, numOfSides, [polygon_coords]
    '''
    polygon_row = ["{0:.4f}".format(convex_hull_ratio), numSides]
    polygon_row.extend([int(item) for sublist in polygon_coords for item in sublist])
    polygon_writer.writerow(polygon_row)
    index += 1
    if index % 1000 == 0:
        print("=======", index, "======")
polygon_csv.close()
