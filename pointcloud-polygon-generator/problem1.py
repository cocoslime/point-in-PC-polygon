import numpy as np
import csv
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from func import *

TRAINING_DATA_NUM = 10000
TEST_DATA_NUM = 500

# polygons = np.loadtxt(, delimiter=',', dtype=np.float32);
polygons_csv = open("../data/problem1/polygons.csv", newline='')
polygons_reader = csv.reader(polygons_csv, quoting=csv.QUOTE_NONNUMERIC);

for index, line in enumerate(polygons_reader):
    polygon_coords = []
    for i in range(0, len(line), 2):
        polygon_coords.append([line[i], line[i+1]])
    polygon = Polygon(polygon_coords)
    bounds = polygon.bounds
    x_buffer = abs(bounds[0] - bounds[2]) / 5.0
    y_buffer = abs(bounds[1] - bounds[3]) / 5.0
    x_min = bounds[0] - x_buffer
    y_min = bounds[1] - y_buffer
    x_max = bounds[2] + x_buffer
    y_max = bounds[3] + y_buffer

    xy = [x_min, x_max, y_min, y_max]
    # training data
    training_pc = make_point_list(polygon, xy, TRAINING_DATA_NUM)

    # test data
    test_pc = make_point_list(polygon, xy, TEST_DATA_NUM)

    training_csv = open("../data/problem1/training_" + str(index) + ".csv", 'w', encoding='utf-8', newline='')
    training_writer = csv.writer(training_csv)
    training_writer.writerow([TRAINING_DATA_NUM])
    for row in training_pc:
        training_writer.writerow(row)
    training_csv.close()

    test_csv = open("../data/problem1/test_" + str(index) + ".csv", 'w', encoding='utf-8', newline='')
    test_writer = csv.writer(test_csv)
    test_writer.writerow([TEST_DATA_NUM])
    for row in test_pc:
        test_writer.writerow(row)
    test_csv.close()

    index += 1

polygons_csv.close()




