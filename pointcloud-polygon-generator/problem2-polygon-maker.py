import matplotlib.pyplot as plt
import random
import numpy as np
import csv
from func import *
import scipy.stats as stats

index = 0
for numSides in range(3, 5):
    for i in range(5):
        polygon_coords = make_random_polygon(numSides, [-1.0, 1., -1., 1.])
        polygon_csv = open("../data/problem2/polygon_" + str(index) + ".csv", 'w', encoding='utf-8', newline='')
        polygon_writer = csv.writer