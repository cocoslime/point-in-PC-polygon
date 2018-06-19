import matplotlib.pyplot as plt
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
        if Polygon(polygon_coords).area < 0.5:
            continue
        polygon_csv = open("../data/problem2/polygon/" +str(numSides) + "_" + str(index) + ".csv",
                           'w', encoding='utf-8', newline='')
        polygon_writer = csv.writer(polygon_csv)
        for xy in polygon_coords:
            polygon_writer.writerow(xy)
        polygon_csv.close()
        index += 1
