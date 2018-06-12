import matplotlib.pyplot as plt
import random
import numpy as np
import csv
from func import *
import scipy.stats as stats

index = 0
NUMBER = 10000
MAX_NUM_SIDES = 3

for numSides in range(3, MAX_NUM_SIDES + 1):
    while True:
        if index >= NUMBER:
            break
        polygon_coords = make_random_polygon(numSides, [-5.0, 5., -5., 5.])
        if Polygon(polygon_coords).area < 0.5:
            continue
        polygon_csv = open("../data/problem2/polygon/" +str(numSides) + "_" + str(index) + ".csv", 'w', encoding='utf-8', newline='')
        polygon_writer = csv.writer(polygon_csv)
        for xy in polygon_coords:
            polygon_writer.writerow(xy)
        polygon_csv.close()
        index += 1
