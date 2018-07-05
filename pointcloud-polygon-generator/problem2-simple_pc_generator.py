import matplotlib.pyplot as plt
import random
import numpy as np
import csv
import math
from func import *
import copy
header = __import__("problem2-simple-header")

POINTS_PER_POLYGON = 200
TRAINING_NUMBER = 20000
TEST_NUMBER = 5000
ONE_POLYGON_TESTNUM = 5

BUFFER = 0.01
DATA_OPT = '001'
RASTER_DIR = "../data/problem2/simple/raster_pc/buffer_" + DATA_OPT
VECTOR_DIR = "../data/problem2/simple/vector_pc/buffer_" + DATA_OPT
# MAKE TRAINING DATA
train_i = 0

raster_training_file = open(RASTER_DIR + "/training.csv", 'w', encoding='utf-8', newline='')
raster_training_writer = csv.writer(raster_training_file)

vector_training_file = open(VECTOR_DIR + "/training.csv", 'w', encoding='utf-8', newline='')
vector_training_writer = csv.writer(vector_training_file)

while train_i < TRAINING_NUMBER:
    data_index = random.randrange(0, header.DATA_NUMBER)
    polygons_csv = open(header.POLYGON_DIRECTORY + str(data_index)
                        + ".csv", newline='')
    polygons_reader = csv.reader(polygons_csv, quoting=csv.QUOTE_NONNUMERIC)
    headers = next(polygons_reader, None)
    x_data = []
    y_data = []
    for rid, row in enumerate(polygons_reader):
        x_data.append(row[0])
        y_data.append(row[1])

    x_data.append(x_data[0])
    y_data.append(y_data[0])

    pg = create_polygon(x_data, y_data)
    equations = make_equation_list(x_data, y_data)

    # pointcloud polygon
    pcp_list = generate_points_along_sides(x_data, y_data, BUFFER, equations, POINTS_PER_POLYGON)
    # target points
    tp_list, labels = make_test_point_list(pg, header.WHOLE_RANGE, ONE_POLYGON_TESTNUM)
    pcp_flatten_list = [element for tupl in pcp_list for element in tupl]

    basis_image = grid(pcp_list, header.WIDTH_NUM, header.HEIGHT_NUM, header.WHOLE_RANGE)
    for target, label in zip(tp_list, labels):
        temp_image = copy.deepcopy(basis_image)
        x_index = find_index(target.x, header.WHOLE_RANGE[1], header.WHOLE_RANGE[0], header.WIDTH_NUM)
        y_index = find_index(target.y, header.WHOLE_RANGE[3], header.WHOLE_RANGE[2], header.HEIGHT_NUM)
        temp_image[x_index][y_index] = 2
        flat_list = [item for sublist in temp_image for item in sublist]
        flat_list.append(label)
        flat_list.extend(headers)
        raster_training_writer.writerow(flat_list)

        vector_data = [str(data_index), target.x, target.y]
        vector_data.extend(pcp_flatten_list)
        vector_data.append(label)
        vector_training_writer.writerow(vector_data)

        train_i += 1
        if train_i % 500 == 0:
            print(train_i)
        if train_i >= TRAINING_NUMBER:
            break

raster_training_file.close()
vector_training_file.close()

# MAKE TEST DATA
test_i = 0

raster_test_file = open(RASTER_DIR + "/test.csv", 'w', encoding='utf-8', newline='')
raster_test_writer = csv.writer(raster_test_file)

vector_test_file = open(VECTOR_DIR + "/test.csv", 'w', encoding='utf-8', newline='')
vector_test_writer = csv.writer(vector_test_file)

while test_i < TEST_NUMBER:
    data_index = random.randrange(0, header.DATA_NUMBER)
    polygons_csv = open(header.POLYGON_DIRECTORY + str(data_index) + ".csv", newline='')
    polygons_reader = csv.reader(polygons_csv, quoting=csv.QUOTE_NONNUMERIC)
    headers = next(polygons_reader, None)
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
    tp_list, labels = make_test_point_list(pg, header.WHOLE_RANGE, ONE_POLYGON_TESTNUM)
    pcp_flatten_list = [element for tupl in pcp_list for element in tupl]

    # Write CSV FILE
    basis_image = grid(pcp_list, header.WIDTH_NUM, header.HEIGHT_NUM, header.WHOLE_RANGE)
    for target, label in zip(tp_list, labels):
        temp_image = copy.deepcopy(basis_image)
        x_index = find_index(target.x, header.WHOLE_RANGE[1], header.WHOLE_RANGE[0], header.WIDTH_NUM)
        y_index = find_index(target.y, header.WHOLE_RANGE[3], header.WHOLE_RANGE[2], header.HEIGHT_NUM)
        temp_image[x_index][y_index] = 2
        flat_list = [item for sublist in temp_image for item in sublist]
        flat_list.append(label)
        flat_list.extend(headers)

        vector_data = [str(data_index), target.x, target.y]
        vector_data.extend(pcp_flatten_list)
        vector_data.append(label)
        vector_test_writer.writerow(vector_data)

        raster_test_writer.writerow(flat_list)
        test_i += 1
        if test_i % 500 == 0:
            print(test_i)
        if test_i >= TEST_NUMBER:
            break
raster_test_file.close()
vector_test_file.close()
