import numpy as np
import csv
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

POINT_NUMBER = 200

# polygons = np.loadtxt(, delimiter=',', dtype=np.float32);
polygons_csv = open("../data/problem1/polygons.csv", newline='')
polygons_reader = csv.reader(polygons_csv, quoting=csv.QUOTE_NONNUMERIC);

index = 0
for line in polygons_reader:
    polygon_coords = []
    for i in range(0, len(line), 2):
        polygon_coords.append([line[i], line[i+1]])
    polygon = Polygon(polygon_coords)
    bounds = polygon.bounds
    half_x_dist = abs(bounds[0] - bounds[2]) / 2.0
    half_y_dist = abs(bounds[1] - bounds[3]) / 2.0
    x_min = bounds[0] - half_x_dist
    y_min = bounds[1] - half_y_dist
    x_max = bounds[2] + half_x_dist
    y_max = bounds[3] + half_y_dist

    pc = []
    for step in range(POINT_NUMBER):
        x_random = float("{0:.3f}".format(np.random.uniform(x_min, x_max)))
        y_random = float("{0:.3f}".format(np.random.uniform(y_min, y_max)))
        point = Point(x_random,y_random)
        isIn = int(polygon.contains(point))
        pc.append([x_random, y_random, isIn])

    points_csv = open("../data/problem1/points_" + str(index) + ".csv", 'w', encoding='utf-8', newline='')
    points_writer = csv.writer(points_csv)
    points_writer.writerow(POINT_NUMBER)
    for row in pc:
        points_writer.writerow(row)
    points_csv.close()
    index += 1

polygons_csv.close()




