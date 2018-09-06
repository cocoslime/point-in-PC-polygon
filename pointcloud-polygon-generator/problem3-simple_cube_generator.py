'''
Rectangle Model
'''
import matplotlib.pyplot as plt
import random
import numpy as np
import csv
import scipy.stats as stats
import os
# header = __import__("problem3-header")

XMAX = 100
YMAX = 100
ZMAX = 100
XMIN = 0
YMIN = 0
ZMIN = 0

TRAINING_NUM = 1000
TEST_NUM = 100
DATA_DIR = "../data/problem3/cube/"
training_file = open(DATA_DIR + "train/cube.txt", 'w', encoding='utf-8', newline='')

training_file.write(str(TRAINING_NUM) + "\n")

index = 0

while index < TRAINING_NUM:
    xmax = random.randrange(XMIN, XMAX)
    ymax = random.randrange(YMIN, YMAX)
    zmax = random.randrange(ZMIN, ZMAX)

    xmin = random.randrange(XMIN, XMAX)
    ymin = random.randrange(YMIN, YMAX)
    zmin = random.randrange(ZMIN, ZMAX)

    if xmax < xmin:
        xmax, xmin = xmin, xmax
    if ymax < ymin:
        ymax, ymin = ymin, ymax
    if zmax < zmin:
        zmax, zmin = zmin, zmax

    if xmax == xmin or ymax == ymin or zmax == zmin:
        continue

    x_dist = xmax - xmin
    y_dist = ymax - ymin
    z_dist = zmax - zmin

    if x_dist * y_dist * z_dist < 300:
        continue

    cube_row = [xmin, ymin, zmin, xmax, ymax, zmax]
    # cube_file.write(" ".join(str(x) for x in cube_row))
    # cube_file.write("\n")

    coords = [
        [xmin, ymin, zmin],
        [xmin, ymax, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymin, zmax],
        [xmin, ymax, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax]
    ]

    coords_str = []
    for xyz in coords:
        coords_str.append(str(xyz[0]) + " " + str(xyz[1]) + " " + str(xyz[2]))

    rectangles_index =[
        [0, 4, 6, 2],
        [6, 7, 3, 2],
        [7, 5, 1, 3],
        [5, 4, 0, 1],
        [6, 4, 5, 7],
        [0, 2, 3, 1]
    ]

    '''
    faces_num face1(vertices_num [vertices]) face2 face3 ... 
    '''
    cube_rectangle_row = [str(len(rectangles_index))]

    for rectangle in rectangles_index:
        cube_rectangle_row.append(str(len(rectangle)))
        for a in rectangle:
            cube_rectangle_row.append(coords_str[a])

    training_file.write(" ".join(str(x) for x in cube_rectangle_row))
    training_file.write("\n")
    index += 1

training_file.close()

test_file = open(DATA_DIR + "test/cube.txt", 'w', encoding='utf-8', newline='')
test_file.write(str(TEST_NUM) + "\n")
index = 0

while index < TEST_NUM:
    xmax = random.randrange(XMIN, XMAX)
    ymax = random.randrange(YMIN, YMAX)
    zmax = random.randrange(ZMIN, ZMAX)

    xmin = random.randrange(XMIN, XMAX)
    ymin = random.randrange(YMIN, YMAX)
    zmin = random.randrange(ZMIN, ZMAX)

    if xmax < xmin:
        xmax, xmin = xmin, xmax
    if ymax < ymin:
        ymax, ymin = ymin, ymax
    if zmax < zmin:
        zmax, zmin = zmin, zmax

    if xmax == xmin or ymax == ymin or zmax == zmin:
        continue

    x_dist = xmax - xmin
    y_dist = ymax - ymin
    z_dist = zmax - zmin

    if x_dist * y_dist * z_dist < 300:
        continue

    cube_row = [xmin, ymin, zmin, xmax, ymax, zmax]
    # cube_file.write(" ".join(str(x) for x in cube_row))
    # cube_file.write("\n")

    coords = [
        [xmin, ymin, zmin],
        [xmin, ymax, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymin, zmax],
        [xmin, ymax, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax]
    ]

    coords_str = []
    for xyz in coords:
        coords_str.append(str(xyz[0]) + " " + str(xyz[1]) + " " + str(xyz[2]))

    rectangles_index =[
        [0, 4, 6, 2],
        [6, 7, 3, 2],
        [7, 5, 1, 3],
        [5, 4, 0, 1],
        [6, 4, 5, 7],
        [0, 2, 3, 1]
    ]

    '''
    faces_num face1(vertices_num [vertices]) face2 face3 ... 
    '''
    cube_rectangle_row = [str(len(rectangles_index))]

    for rectangle in rectangles_index:
        cube_rectangle_row.append(str(len(rectangle)))
        for a in rectangle:
            cube_rectangle_row.append(coords_str[a])

    test_file.write(" ".join(str(x) for x in cube_rectangle_row))
    test_file.write("\n")
    index += 1

test_file.close()