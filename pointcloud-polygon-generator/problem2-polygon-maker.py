import matplotlib.pyplot as plt
import random
import numpy as np
import csv
from func import *
import scipy.stats as stats

index = 0
NUMBER = 500
MAX_NUM_SIDES = 3

for numSides in range(3, MAX_NUM_SIDES + 1):
    for i in range(NUMBER):
        polygon_coords = make_random_polygon(numSides, [-1.0, 1., -1., 1.])
        if Polygon(polygon_coords).area < 0.5:
            i -= 1
            continue
        polygon_csv = open("../data/problem2/polygon/" + str(index) + ".csv", 'w', encoding='utf-8', newline='')
        polygon_writer = csv.writer(polygon_csv)
        for xy in polygon_coords:
            polygon_writer.writerow(xy)
        polygon_csv.close()
        index += 1
