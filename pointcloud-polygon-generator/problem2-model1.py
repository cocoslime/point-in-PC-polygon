import matplotlib.pyplot as plt
import random
import numpy as np
import csv
import math
from func import *

POINTS_PER_POLYGON = 32 * 32
BUFFER = 0.05
SIGMA = 0.1
TRAINING_NUMBER = 500
TEST_NUMBER = 100

NUM_SIDES = 3
DATA_NUMBER = 500
ONE_POLYGON_TESTNUM = 5

# MAKE TRAINING DATA
train_i = 0

while train_i < TRAINING_NUMBER:
    data_index = random.randrange(0, DATA_NUMBER)
    polygons_csv = open("../data/problem2/polygon/"
                        + str(NUM_SIDES) + "_" + str(data_index)
                        + ".csv", newline='')
    polygons_reader = csv.reader(polygons_csv, quoting=csv.QUOTE_NONNUMERIC)

    x_data = []
    y_data = []
    for index, row in enumerate(polygons_reader):
        x_data.append(row[0])
        y_data.append(row[1])

    x_data.append(x_data[0])
    y_data.append(y_data[0])

    pg = create_polygon(x_data, y_data)
    equations = make_equation_list(x_data, y_data)

    # pointcloud polygon
    pcp_list = generate_points_along_sides(x_data, y_data, BUFFER, equations, POINTS_PER_POLYGON)
    # target points
    tp_list, labels = make_test_point_list(pg, [-1.0, 1., -1., 1.], ONE_POLYGON_TESTNUM)

    training_csv = open("../data/problem2/training_" + str(train_i) + ".csv", 'w', encoding='utf-8', newline='')
    training_writer = csv.writer(training_csv)
    training_writer.writerow(labels)
    for train_point in tp_list:
        training_writer.writerow(train_point)
    for pt in pcp_list:
        training_writer.writerow(pt)
    training_csv.close()
    train_i += 1

    # for drawing

    # tp_x = []
    # tp_y = []
    # for tp in tp_list:
    #     tp_y.append(tp.y)
    #     tp_x.append(tp.x)
    # plt.plot(tp_x, tp_y, 'o')
    # plt.plot(x_data, y_data)
    # plt.show()

# MAKE TEST DATA
test_i = 0

while test_i < TEST_NUMBER:
    data_index = random.randrange(0, DATA_NUMBER)
    polygons_csv = open("../data/problem2/polygon/" + str(NUM_SIDES) + "_" + str(data_index) + ".csv", newline='')
    polygons_reader = csv.reader(polygons_csv, quoting=csv.QUOTE_NONNUMERIC)

    x_data = []
    y_data = []
    for index, row in enumerate(polygons_reader):
        x_data.append(row[0])
        y_data.append(row[1])

    x_data.append(x_data[0])
    y_data.append(y_data[0])

    pg = create_polygon(x_data, y_data)
    equations = make_equation_list(x_data, y_data)

    # pointcloud polygon
    pcp_list = generate_points_along_sides(x_data, y_data, BUFFER, equations, POINTS_PER_POLYGON)
    # target points
    tp_list, labels = make_test_point_list(pg, [-1.0, 1., -1., 1.], ONE_POLYGON_TESTNUM)

    test_csv = open("../data/problem2/test_" + str(test_i) + ".csv", 'w', encoding='utf-8', newline='')
    writer = csv.writer(test_csv)
    writer.writerow(labels)
    for testpoint in tp_list:
        writer.writerow(testpoint)
    for pt in pcp_list:
        writer.writerow(pt)
    test_csv.close()
    test_i += 1

    # for drawing

    # tp_x = []
    # tp_y = []
    # for tp in tp_list:
    #     tp_y.append(tp.y)
    #     tp_x.append(tp.x)
    # plt.plot(tp_x, tp_y, 'o')
    # plt.plot(x_data, y_data)
    # plt.show()