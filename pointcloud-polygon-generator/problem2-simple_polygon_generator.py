import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing
from shapely.geometry import LineString
import random
import numpy as np
import csv
from func import *
import scipy.stats as stats
import os
header = __import__("problem2-simple-header")

index = 0

if not os.path.exists(header.POLYGON_DIRECTORY):
    os.makedirs(header.POLYGON_DIRECTORY)

while index < header.DATA_NUMBER:
    numSides = random.randrange(header.NUM_SIDES[0], header.NUM_SIDES[-1])
    polygon_coords = make_random_polygon(numSides, header.WHOLE_RANGE)
    polygon = Polygon(polygon_coords)
    ring = LinearRing(polygon_coords)
    if not ring.is_simple:
        continue

    convex_hull = polygon.convex_hull
    if convex_hull.area < 2.0:
        continue

    # draw_polygon(polygon)

    polygon_csv = open(header.POLYGON_DIRECTORY + str(index) + ".csv",
                       'w', encoding='utf-8', newline='')
    polygon_writer = csv.writer(polygon_csv)
    polygon_writer.writerow([polygon.area / convex_hull.area, numSides])
    for xy in polygon_coords:
        polygon_writer.writerow(xy)
    polygon_csv.close()
    index += 1
