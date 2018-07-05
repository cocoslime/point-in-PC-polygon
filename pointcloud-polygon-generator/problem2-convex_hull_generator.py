import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing
from shapely.geometry import LineString
import random
import numpy as np
import csv
from func import *
import scipy.stats as stats
header = __import__("problem2-convex-header")

for numSides in header.NUM_SIDES:
    index = 0
    polygon_csv = open("../data/problem2/convex/polygon/" + str(numSides) + ".csv",
                           'w', encoding='utf-8', newline='')
    polygon_writer = csv.writer(polygon_csv)
    while index < header.DATA_NUMBER:
        polygon_coords = make_random_polygon(numSides, header.WHOLE_RANGE)
        polygon = Polygon(polygon_coords)
        convex_hull = polygon.convex_hull
        if convex_hull.area < 2.0:
            continue

        convex_coords = list(convex_hull.exterior.coords)[:-1]
        if len(convex_coords) != numSides:
            continue

        convex_coords = [element for tupl in convex_coords for element in tupl]

        polygon_writer.writerow(convex_coords)

        index += 1
    polygon_csv.close()
