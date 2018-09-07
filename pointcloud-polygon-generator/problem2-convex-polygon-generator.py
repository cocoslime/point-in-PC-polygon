import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing
from shapely.geometry import LineString
import random
import numpy as np
import csv
import scipy.stats as stats
from func_2 import *
header = __import__("problem2-header")

index = 0
polygon_csv = open("../data/problem2/convex/polygon.csv",
                       'w', encoding='utf-8', newline='')
polygon_writer = csv.writer(polygon_csv)

while index < header.POLYGON_NUMBER:
    numSides = random.randrange(header.NUM_SIDES[0], header.NUM_SIDES[-1])
    polygon_coords = make_random_coordinate_list(numSides, header.WHOLE_RANGE)
    polygon = Polygon(polygon_coords)
    convex_hull = polygon.convex_hull
    if convex_hull.area < 2000:
        continue
    convex_coords = list(convex_hull.exterior.coords)[:-1]

    '''
    convex_ratio = 1.0, numOfSides, [polygon_coords]
    '''
    polygon_row = [1.0, len(convex_coords)]
    polygon_row.extend([int(item) for sublist in convex_coords for item in sublist])
    polygon_writer.writerow(polygon_row)
    index += 1
    if index % 1000 == 0:
        print("=======", index, "======")

polygon_csv.close()
