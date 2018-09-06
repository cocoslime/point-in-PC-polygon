import matplotlib.pyplot as plt
import random
import numpy as np
import csv
import math
from func_2 import *
header = __import__("problem2-header")

POINTS_PER_POLYGON = 200
TRAINING_NUMBER = 10000
TEST_NUMBER = 1000
ONE_POLYGON_TESTNUM = 5

BUFFER = 0.01
DATA_OPT = '001'

for numsides in header.NUM_SIDES:
    polygons_csv = open("../data/problem2/convex/polygon/" + str(numsides) + ".csv", newline='')
    polygons_reader = csv.reader(polygons_csv, quoting=csv.QUOTE_NONNUMERIC)

    # MAKE TRAINING DATA
    train_i = 0
    training_file = open("../data/problem2/convex/vector_pc/buffer_" + DATA_OPT + "/training_" + str(numsides)
                         + ".csv", 'w', encoding='utf-8', newline='')
    training_writer = csv.writer(training_file)

    while train_i < TRAINING_NUMBER:
        data_index = random.randrange(0, header.DATA_NUMBER)
        polygons_csv.seek(0)
        x_data = []
        y_data = []
        for rid, row in enumerate(polygons_reader):
            if rid == data_index:
                for x, y in pairwise(row):
                    x_data.append(x)
                    y_data.append(y)
                break

        x_data.append(x_data[0])
        y_data.append(y_data[0])

        pg = create_polygon(x_data, y_data)
        equations = make_equation_list(x_data, y_data)

        # pointcloud polygon
        pcp_list = generate_points_along_sides(x_data, y_data, BUFFER, equations, POINTS_PER_POLYGON)
        # target points
        tp_list, labels = make_test_point_list(pg, header.WHOLE_RANGE, ONE_POLYGON_TESTNUM)

        '''
        polygon_id : numSides + "_" + data_index 
        polygon_id, target_point_x, target_point_y, boundary_points, ... , label
        '''
        pcp_flatten_list = [element for tupl in pcp_list for element in tupl]
        for tp, label in zip(tp_list, labels):
            row_data = [str(numsides) + "_" + str(data_index), tp.x, tp.y]
            row_data.extend(pcp_flatten_list)
            row_data.append(label)
            training_writer.writerow(row_data)
            train_i += 1

        if train_i % 100 == 0:
            print(train_i)

    training_file.close()

    # MAKE TEST DATA
    test_i = 0

    test_csv = open(
        "../data/problem2/convex/vector_pc/buffer_" + DATA_OPT + "/test_" + str(numsides)
                         + ".csv", 'w', encoding='utf-8', newline='')
    test_writer = csv.writer(test_csv)

    while test_i < TEST_NUMBER:
        data_index = random.randrange(0, header.DATA_NUMBER)
        polygons_csv.seek(0)

        x_data = []
        y_data = []
        for rid, row in enumerate(polygons_reader):
            if rid == data_index:
                for x, y in pairwise(row):
                    x_data.append(x)
                    y_data.append(y)
                break

        x_data.append(x_data[0])
        y_data.append(y_data[0])

        pg = create_polygon(x_data, y_data)
        equations = make_equation_list(x_data, y_data)

        # pointcloud polygon
        pcp_list = generate_points_along_sides(x_data, y_data, BUFFER, equations, POINTS_PER_POLYGON)
        # target points
        tp_list, labels = make_test_point_list(pg, header.WHOLE_RANGE, ONE_POLYGON_TESTNUM)

        # Write CSV FILE

        '''
        polygon_id : numSides + "_" + data_index 
        polygon_id, target_point_x, target_point_y, boundary_points, ... , label
        '''
        pcp_flatten_list = [element for tupl in pcp_list for element in tupl]
        for tp, label in zip(tp_list, labels):
            row_data = [str(numsides) + "_" + str(data_index), tp.x, tp.y]
            row_data.extend(pcp_flatten_list)
            row_data.append(label)
            test_writer.writerow(row_data)

        test_i += 1

        if test_i % 100 == 0:
            print(test_i)
    test_csv.close()
