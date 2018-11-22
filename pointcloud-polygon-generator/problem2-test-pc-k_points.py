import matplotlib.pyplot as plt
import random
import numpy as np
import csv
import math
from problem2_func import *
import copy

PIXEL = 32

POLYGON_NUMBER = 30000
WHOLE_RANGE = [0, 100, 0, 100]

TRAINING_NUMBER = 30000
TEST_NUMBER = 20000
ONE_POLYGON_TESTNUM = 4

BUFFER = 1
CONVEX_OPT = 'non_convex'

RASTER_DIR = "../data/problem2/" + CONVEX_OPT + "/raster_pc/"
VECTOR_DIR = "../data/problem2/" + CONVEX_OPT + "/vector_pc/"

polygons_csv = open("../data/problem2/" + CONVEX_OPT + "/polygon.csv", newline='')
polygons_reader = csv.reader(polygons_csv, quoting=csv.QUOTE_NONNUMERIC)

polygon_rows = []
for i in range(0, 5000):
    if i % 50 == 0:
        print(i)
    polygons_csv.seek(0)
    data_index = random.randrange(0, POLYGON_NUMBER)
    for rid, row in enumerate(polygons_reader):
        if rid == data_index:
            polygon_rows.append(row)
            break


def generate(raster_writer, vector_writer, points_per_polygon):
    for polygon_row in polygon_rows:
        x_data = []
        y_data = []

        for index, value in enumerate(polygon_row):
            if index == 0:
                convex_ratio = value
            elif index == 1:
                number_sides = value
            else:
                if index % 2 == 0:
                    x_data.append(value)
                else:
                    y_data.append(value)

        x_data.append(x_data[0])
        y_data.append(y_data[0])

        perimeter = calculate_perimeter(x_data, y_data)
        pg = create_polygon(x_data, y_data)

        # pointcloud polygon
        pcp_list = generate_points_along_sides(x_data, y_data, BUFFER, points_per_polygon)
        # target points

        tp_list, labels = make_target_point_list(pg, WHOLE_RANGE, ONE_POLYGON_TESTNUM, ONE_POLYGON_TESTNUM / 2)
        pcp_flatten_list = [element for tupl in pcp_list for element in tupl]

        basis_image = grid(pcp_list, PIXEL, PIXEL, WHOLE_RANGE)
        for target, label in zip(tp_list, labels):
            temp_image = copy.deepcopy(basis_image)
            x_index = find_index(target.x, WHOLE_RANGE[1], WHOLE_RANGE[0], PIXEL)
            y_index = find_index(target.y, WHOLE_RANGE[3], WHOLE_RANGE[2], PIXEL)
            temp_image[x_index][y_index] = 2

            '''
            raster data
            data_index, perimeter, [pixel_data], label
            '''
            flat_list = [data_index, perimeter]
            flat_list.extend([item for sublist in temp_image for item in sublist])
            flat_list.append(label)
            raster_writer.writerow(flat_list)

            # header.draw_raster_row(flat_list)

            '''
            vector data
            data_index, perimeter, [target_point], [polygon_coords], label
            '''
            vector_data = [str(data_index), perimeter, target.x, target.y]
            vector_data.extend(pcp_flatten_list)
            vector_data.append(label)
            vector_writer.writerow(vector_data)


def calculate_perimeter(x_data, y_data):
    perimeter = 0
    for i in range(len(x_data) - 1):
        x_d = x_data[i] - x_data[i+1]
        y_d = y_data[i] - y_data[i+1]
        perimeter += math.sqrt(x_d * x_d + y_d * y_d)
    return perimeter


for points_num in [60, 70, 80, 90]:
    print(points_num)
    raster_test_file = open(RASTER_DIR + "points/p" + str(points_num) + "_test_" + '{0:03d}'.format(BUFFER) + ".csv", 'w', encoding='utf-8', newline='')
    raster_test_writer = csv.writer(raster_test_file)

    vector_test_file = open(VECTOR_DIR + "points/p" + str(points_num) + "_test_" + '{0:03d}'.format(BUFFER) + ".csv", 'w', encoding='utf-8', newline='')
    vector_test_writer = csv.writer(vector_test_file)

    generate(raster_test_writer, vector_test_writer, points_num)
    raster_test_file.close()
    vector_test_file.close()

