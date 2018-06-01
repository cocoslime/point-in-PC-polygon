import numpy as np
import csv


def load_data(pre_path, data_num):
    x_data = []
    y_data = []
    for i in range(data_num):
        file_path = pre_path + str(i) + ".csv"
        csvfile = open(file_path, newline='')
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)

        i_data_num = 0
        i_x_data = []
        i_boundary = []
        for index, row in enumerate(reader):
            if index == 0:
                for label in row:
                    y_data.append([label])
                i_data_num = len(row)
            elif index <= i_data_num:
                i_x_data.append([row[0], row[1]])
            else:
                i_boundary.extend([row[0], row[1]])

        for i_data in i_x_data:
            i_data.extend(i_boundary)

        x_data.extend(i_x_data)

    return x_data, y_data
