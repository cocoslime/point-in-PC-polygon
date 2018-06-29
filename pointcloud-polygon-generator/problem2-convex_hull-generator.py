import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing
from shapely.geometry import LineString
import random
import numpy as np
import csv
from func import *
import scipy.stats as stats
header = __import__("problem2-header")

NUMBER = 10000

for numSides in header.NUM_SIDES:
    index = 0
    while index < NUMBER:
        polygon_coords = make_random_polygon(numSides, header.WHOLE_RANGE)
        polygon = Polygon(polygon_coords)
        convex_hull = polygon.convex_hull
        if convex_hull.area < 2.0:
            continue

        polygon_csv = open("../data/problem2/polygon/" + str(numSides) + "_" + str(index) + ".csv",
                           'w', encoding='utf-8', newline='')
        polygon_writer = csv.writer(polygon_csv)
        for xy in list(convex_hull.exterior.coords)[:-1]:
            polygon_writer.writerow(xy)
        polygon_csv.close()
        index += 1
