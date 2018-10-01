import matplotlib.pyplot as plt
import random
import numpy as np
import csv
import math
from func_2 import *
import copy
header = __import__("problem2-header")

POINTS_PER_POLYGON = 200
ONE_POLYGON_TESTNUM = 5
TRAINING_NUM = 10000
TEST_NUM = 1000

BUFFER = 0.01
DATA_OPT = '001'

for numsides in header.NUM_SIDES:
    # MAKE TRAINING DATA
    train_i = 0
    vector_file = open("../data/problem2/convex/vector_pc/buffer_" + DATA_OPT + "/training_" + str(numsides)
                         + ".csv", newline='')
    vector_reader = csv.reader(vector_file, quoting=csv.QUOTE_NONNUMERIC)

    training_file = open("../data/problem2/convex/raster_pc/buffer_" + DATA_OPT + "/training_" + str(numsides)
                         + ".csv", 'w', encoding='utf-8', newline='')
    training_writer = csv.writer(training_file)

    headers = next(vector_reader, None)
    for rid, pl_vector_data in enumerate(vector_reader):
        label = pl_vector_data[-1]
        coords = pl_vector_data[3:-1]
        target = PCP_Point(pl_vector_data[1], pl_vector_data[2])

        pcp_list = []
        for x, y in pairwise(coords):
            pcp_list.append(PCP_Point(x, y))

        basis_image = grid(pcp_list, header.WIDTH_NUM, header.HEIGHT_NUM, header.WHOLE_RANGE)
        x_index = find_index(target.x, header.WHOLE_RANGE[1], header.WHOLE_RANGE[0], header.WIDTH_NUM)
        y_index = find_index(target.y, header.WHOLE_RANGE[3], header.WHOLE_RANGE[2], header.HEIGHT_NUM)
        basis_image[x_index][y_index] = 2
        flat_list = [item for sublist in basis_image for item in sublist]
        flat_list.append(label)
        training_writer.writerow(flat_list)
        train_i += 1
        if train_i % 100 == 0:
            print(train_i)
        if train_i == TRAINING_NUM:
            break;

    training_file.close()

    # MAKE TEST DATA

    test_i = 0
    vector_file = open("../data/problem2/convex/vector_pc/buffer_" + DATA_OPT + "/test_" + str(numsides)
                       + ".csv", newline='')
    vector_reader = csv.reader(vector_file, quoting=csv.QUOTE_NONNUMERIC)

    test_file = open("../data/problem2/convex/raster_pc/buffer_" + DATA_OPT + "/test_" + str(numsides)
                         + ".csv", 'w', encoding='utf-8', newline='')
    test_writer = csv.writer(test_file)

    headers = next(vector_reader, None)
    for rid, pl_vector_data in enumerate(vector_reader):
        label = pl_vector_data[-1]
        coords = pl_vector_data[3:-1]
        target = PCP_Point(pl_vector_data[1], pl_vector_data[2])

        pcp_list = []
        for x, y in pairwise(coords):
            pcp_list.append(PCP_Point(x, y))

        basis_image = grid(pcp_list, header.WIDTH_NUM, header.HEIGHT_NUM, header.WHOLE_RANGE)
        x_index = find_index(target.x, header.WHOLE_RANGE[1], header.WHOLE_RANGE[0], header.WIDTH_NUM)
        y_index = find_index(target.y, header.WHOLE_RANGE[3], header.WHOLE_RANGE[2], header.HEIGHT_NUM)
        basis_image[x_index][y_index] = 2
        flat_list = [item for sublist in basis_image for item in sublist]
        flat_list.append(label)
        test_writer.writerow(flat_list)
        test_i += 1
        if test_i % 100 == 0:
            print(test_i)
        if test_i == TEST_NUM:
            break
    test_file.close()

