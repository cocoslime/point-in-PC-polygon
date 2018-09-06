import random
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import csv
import collections
from numpy.linalg import lstsq
from numpy import ones, vstack

BUFFER = 1
POINTS_PER_POLYGON = 1000

MAX_DATA_NUM = 100
CONVEX_OPT = 'convex'
SET_NUM = 5
DATA_DIR = "../data/problem3/extruded/"

VOXEL_PIXEL = 10
ONE_AXIS_PIXEL = int(100 / VOXEL_PIXEL)
VOXEL_SHAPE = (ONE_AXIS_PIXEL, ONE_AXIS_PIXEL, ONE_AXIS_PIXEL)
Equation = collections.namedtuple('Equation', 'a b c')  # ax + by + c = 0


def make_equation_list(x_data,  y_data):
    equations = []
    for i in range(len(x_data) - 1):
        x_coords = [x_data[i], x_data[i + 1]]
        y_coords = [y_data[i], y_data[i+1]]
        equ = make_equation(x_coords, y_coords)
        equations.append(equ)
    return equations


def make_equation(x_coords, y_coords):
    assert ((x_coords[0] != x_coords[1]) | (y_coords[0] != y_coords[1]))
    # x = c
    if x_coords[0] == x_coords[1]:
        equ = Equation(1, 0, -x_coords[0])
        return equ

    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0] # y = mx + c
    equ = Equation(m, -1, c)
    return equ


def generate_random_point_in_rectangle(center, buffer):
    x_value = random.randrange(center[0]-buffer, center[0]+buffer+1)
    y_value = random.randrange(center[1]-buffer, center[1]+buffer+1)
    return x_value, y_value


def generate_points_along_sides(x_data, y_data, point_num):
    equations = make_equation_list(x_data, y_data)

    pc = []
    for step in range(point_num):
        r_index = random.randrange(0, len(equations))
        x_coords = [x_data[r_index], x_data[r_index+1]]
        y_coords = [y_data[r_index], y_data[r_index+1]]
        equ = equations[r_index]

        min_x = min(x_coords)
        max_x = max(x_coords)
        x_random = random.randrange(min_x, max_x + 1)
        b = equ.b
        if b == 0:
            low = min(y_coords) - BUFFER
            upp = max(y_coords) + BUFFER
            y_random = random.randrange(low, upp + 1)
        elif b == -1:
            y_random = int(equ.a * x_random + equ.c)
        else:
            return
        result_x, result_y = generate_random_point_in_rectangle([x_random, y_random], BUFFER)
        pc.append([result_x, result_y, 0])
    return pc


def create_polygon(x_coords, y_coords):
    xy = zip(x_coords, y_coords)
    return Polygon(xy)


def generate_points_along_top_bottom(x_data, y_data, point_num, top_height):
    pc = []
    pg = create_polygon(x_data, y_data)
    min_x = min(x_data)
    max_x = max(x_data)
    min_y = min(y_data)
    max_y = max(y_data)

    while len(pc) < point_num:
        random_x = random.randrange(min_x, max_x+1)
        random_y = random.randrange(min_y, max_y+1)

        if not pg.contains(Point(random_x, random_y)):
            continue

        is_in_top = random.randrange(0, 2)
        if is_in_top:
            random_height = random.randrange(top_height - BUFFER, top_height + BUFFER + 1)
        else:
            random_height = random.randrange(0, BUFFER + 1)
        pc.append([random_x, random_y, random_height])
    return pc


def make_target_point(x_data, y_data, max_height, want_in=False):
    min_x = min(x_data)
    max_x = max(x_data)
    min_y = min(y_data)
    max_y = max(y_data)
    pg = create_polygon(x_data, y_data)

    x_dist = max_x - min_x
    y_dist = max_y - min_y
    if want_in:
        random_x = random.randrange(int(min_x), int(max_x) + 1)
        random_y = random.randrange(int(min_y), int(max_y) + 1)
        random_height = random.randrange(0, int(max_height) + 1)
    else:
        random_x = random.randrange(int(min_x - x_dist * 0.5), int(max_x + x_dist * 0.5) + 1)
        random_y = random.randrange(int(min_y - y_dist * 0.5), int(max_y + y_dist * 0.5) + 1)
        random_height = random.randrange(0, int(max_height * 1.5) + 1)
    if random_height > 100:
        random_height = 100

    label = pg.contains(Point(random_x, random_y)) and random_height < max_height
    return [random_x, random_y, random_height], label


def generate_data(solids_reader, vector_writer, raster_writer):
    num_of_out = 0
    write_num = 0
    solids = []
    for rid, row in enumerate(solids_reader):
        solids.append(row)

    while write_num < MAX_DATA_NUM:
        if write_num % 10 == 0:
            print(write_num, "...")

        convex_ratio = row[0]
        num_sides = row[1]
        height = row[2]
        polygon_coords = row[3:]
        polygon_coords.extend([polygon_coords[0], polygon_coords[1]])
        x_data = polygon_coords[0:][::2]
        y_data = polygon_coords[1:][::2]

        # pointcloud polygon
        pcp_list = generate_points_along_sides(x_data, y_data, POINTS_PER_POLYGON * 4)
        for point in pcp_list:
            point[2] = random.randrange(0, height + 1)

        pcp_list_2 = generate_points_along_top_bottom(x_data, y_data, POINTS_PER_POLYGON * 2, height)
        pcp_list.extend(pcp_list_2)
        pcp_flatten_list = [element for tupl in pcp_list for element in tupl]

        if num_of_out > MAX_DATA_NUM / 2:
            target_point, label = make_target_point(x_data, y_data, height, True)
        else:
            target_point, label = make_target_point(x_data, y_data, height)

        if not label:
            if num_of_out > MAX_DATA_NUM * 0.66:
                continue
            num_of_out += 1
        '''
        vector data
        target point, [boundary_points], label
        '''
        vector_row = target_point
        vector_row.extend(pcp_flatten_list)
        vector_row.append(int(label))
        vector_writer.writerow(vector_row)

        '''
        raster data
        [voxel] label
        '''
        voxel = np.zeros(VOXEL_SHAPE)
        for point in pcp_list:
            x = int(point[0] / VOXEL_PIXEL)
            x = min(x, ONE_AXIS_PIXEL - 1)
            x = max(x, 0)
            y = int(point[1] / VOXEL_PIXEL)
            y = min(y, ONE_AXIS_PIXEL - 1)
            y = max(y, 0)
            z = int(point[2] / VOXEL_PIXEL)
            z = min(z, ONE_AXIS_PIXEL - 1)
            z = max(z, 0)
            voxel[x][y][z] = 1

        x = int(target_point[0] / VOXEL_PIXEL)
        x = min(x, ONE_AXIS_PIXEL - 1)
        x = max(x, 0)
        y = int(target_point[1] / VOXEL_PIXEL)
        y = min(y, ONE_AXIS_PIXEL - 1)
        y = max(y, 0)
        z = int(target_point[2] / VOXEL_PIXEL)
        z = min(z, ONE_AXIS_PIXEL - 1)
        z = max(z, 0)
        voxel[x][y][z] = 2

        raster_row = [int(item2) for sublist in voxel for item in sublist for item2 in item]
        raster_row.append(int(label))
        raster_writer.writerow(raster_row)

        write_num += 1


if __name__ == "__main__":
    solids_csv = open(DATA_DIR + CONVEX_OPT + "_solids.csv", newline='')
    solids_reader = csv.reader(solids_csv, quoting=csv.QUOTE_NONNUMERIC)

    raster_test_csv = open(DATA_DIR + CONVEX_OPT + "/raster/" + "test_" + ".csv", 'w', encoding='utf-8', newline='')
    raster_test_writer = csv.writer(raster_test_csv)

    vector_test_csv = open(DATA_DIR + CONVEX_OPT + "/vector/" + "test_" + ".csv", 'w', encoding='utf-8', newline='')
    vector_test_writer = csv.writer(vector_test_csv)

    for set_num in range(SET_NUM):
        print("Train ", set_num)
        raster_train_csv = open(DATA_DIR + CONVEX_OPT + "/raster/" + "training_" + str(set_num) + ".csv", 'w', encoding='utf-8', newline='')
        raster_train_writer = csv.writer(raster_train_csv)

        vector_train_csv = open(DATA_DIR + CONVEX_OPT + "/vector/" + "training_" + str(set_num) + ".csv", 'w', encoding='utf-8', newline='')
        vector_train_writer = csv.writer(vector_train_csv)

        solids_csv.seek(0)
        generate_data(solids_reader, vector_train_writer, raster_train_writer)

        raster_train_csv.close()
        vector_train_csv.close()
    print("Test ")

    solids_csv.seek(0)
    generate_data(solids_reader, raster_test_writer, vector_test_writer)
    raster_test_csv.close()
    vector_test_csv.close()






