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

TEST_NUMBER = 30000
ONE_POLYGON_TESTNUM = 4
POINTS_PER_POLYGON = 200

BUFFER = 1
CONVEX_OPT = 'non_convex'

RASTER_DIR = "../data/problem2/" + CONVEX_OPT + "/raster_pc/"
VECTOR_DIR = "../data/problem2/" + CONVEX_OPT + "/vector_pc/"

polygons_csv = open("../data/problem2/" + CONVEX_OPT + "/polygon.csv", newline='')
polygons_reader = csv.reader(polygons_csv, quoting=csv.QUOTE_NONNUMERIC)


def generate(raster_writer, vector_writer, reader, number):
    i = 0
    while i < number:
        polygons_csv.seek(0)
        data_index = random.randrange(0, POLYGON_NUMBER)

        x_data = []
        y_data = []

        for rid, row in enumerate(reader):
            if rid == data_index:
                for index, value in enumerate(row):
                    if index == 0:
                        convex_ratio = value
                    elif index == 1:
                        number_sides = value
                    else:
                        if index % 2 == 0:
                            x_data.append(value)
                        else:
                            y_data.append(value)
                break

        x_data.append(x_data[0])
        y_data.append(y_data[0])

        perimeter = int(calculate_perimeter(x_data, y_data))
        pg = create_polygon(x_data, y_data)

        # pointcloud polygon
        pcp_list = generate_points_along_sides(x_data, y_data, BUFFER, POINTS_PER_POLYGON)
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
            data_index, density, [pixel_data], label
            '''
            flat_list = [data_index, perimeter]
            flat_list.extend([item for sublist in temp_image for item in sublist])
            flat_list.append(label)
            raster_writer.writerow(flat_list)

            # header.draw_raster_row(flat_list)

            '''
            vector data
            data_index, density, [target_point], [polygon_coords], label
            '''
            vector_data = [str(data_index), perimeter, target.x, target.y]
            vector_data.extend(pcp_flatten_list)
            vector_data.append(label)
            vector_writer.writerow(vector_data)

            i += 1

            if i % 100 == 0:
                print(i)
            if i >= number:
                break


def calculate_perimeter(x_data, y_data):
    dist = 0
    for i in range(len(x_data) - 1):
        x_d = x_data[i] - x_data[i+1]
        y_d = y_data[i] - y_data[i+1]
        dist += math.sqrt(x_d * x_d + y_d * y_d)
    return dist


raster_test_file = open(RASTER_DIR + "perimeter/" + "test_" + '{0:03d}'.format(BUFFER) + ".csv", 'w', encoding='utf-8', newline='')
raster_test_writer = csv.writer(raster_test_file)

vector_test_file = open(VECTOR_DIR + "perimeter/" + "test_" + '{0:03d}'.format(BUFFER) + ".csv", 'w', encoding='utf-8', newline='')
vector_test_writer = csv.writer(vector_test_file)

generate(raster_test_writer, vector_test_writer, polygons_reader, TEST_NUMBER)
raster_test_file.close()
vector_test_file.close()

