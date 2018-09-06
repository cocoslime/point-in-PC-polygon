import random
import numpy as np
import csv

MAX_HEIGHT = 100
MIN_HEIGHT = 10

CONVEX_OPT = 'convex'

polygons_csv = open("../data/problem2/" + CONVEX_OPT + "/polygon.csv", newline='')
polygons_reader = csv.reader(polygons_csv, quoting=csv.QUOTE_NONNUMERIC)

solids_csv = open("../data/problem3/extruded/" + CONVEX_OPT + "_solids.csv", 'w', encoding='utf-8', newline='')
solids_writer = csv.writer(solids_csv)

for rid, row in enumerate(polygons_reader):
    '''
    convex_ratio, numOfSides, height, [polygon_coords]
    '''
    num_sides = row[1]
    coords = row[2:]
    new_row = row[0:2]
    new_row.append(random.randrange(MIN_HEIGHT, MAX_HEIGHT))
    new_row.extend(coords)

    solids_writer.writerow(new_row)
solids_csv.close()



    
