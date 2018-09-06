# unused... C++

import matplotlib.pyplot as plt
import random
import numpy as np
import csv

DATA_DIR = "../data/problem3/cube/"
POINTS_NUMBER = 1000

cube_rectangle_csv = open(DATA_DIR + "cube_rectangle.csv", newline='')
cube_rectangle_reader = csv.reader(cube_rectangle_csv)

cube_csv = open(DATA_DIR + "cube.csv", newline='')
cube_reader = csv.reader(cube_csv, quoting=csv.QUOTE_NONNUMERIC)

raster_training_file = open(DATA_DIR + "raster/" + "training.csv", 'w', encoding='utf-8', newline='')
raster_training_writer = csv.writer(raster_training_file)

vector_training_file = open(DATA_DIR + "vector/" + "training.csv", 'w', encoding='utf-8', newline='')
vector_training_writer = csv.writer(vector_training_file)


def enumerate_rectangle(iterable):
    a= iter(iterable)
    return zip(a, a, a, a)


def str_to_coords(rectangle_str):
    rectangle_coords=[]
    for coord_str in rectangle_str:
        coord_arr = coord_str.split()
        rectangle_coords.append(list(map(int, coord_arr)))
    return rectangle_coords


def make_random_point(mbb):
    target_x = random.randrange(mbb[0], mbb[3])
    target_y = random.randrange(mbb[1], mbb[4])
    target_z = random.randrange(mbb[2], mbb[5])
    return [target_x, target_y, target_z]

def sample_point(rect):



if __name__ == "__main__":
    cube = []
    for mbb in enumerate(cube_reader):
        cube.append(mbb)

    for index, row in enumerate(cube_rectangle_reader):
        rectangles = []
        for rect_str in enumerate_rectangle(row):
            rect_coords = str_to_coords(rect_str)
            rectangles.append(rect_coords)

        target_point = make_random_point(cube[index])
        points = []
        while len(points) < POINTS_NUMBER:
            rid = random.randrange(0, len(rectangles))
            random_point = sample_point(rectangles[rid])
            points.extend(random_point)
