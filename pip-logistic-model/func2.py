import numpy as np
import csv
import tensorflow as tf


def pairwise(iterable):
    """s -> (s0, s1), (s2, s3), (s4, s5), ..."""
    a = iter(iterable)
    return zip(a, a)


def grid(data, x_num, y_num, data_range):
    result = []
    one_cell_x = (data_range[1] - data_range[0]) / x_num
    one_cell_y = (data_range[3] - data_range[2]) / y_num
    for i_data in data:
        index = 0
        grid_data = np.zeros((x_num, y_num))
        for x, y in i_data:
            x_index = (x - data_range[0]) / one_cell_x
            x_index = int(x_index)
            y_index = int((y - data_range[2]) / one_cell_y)
            if x_index == x_num:
                x_index -= 1
            if y_index == y_num:
                y_index -= 1
            if index == 0:
                grid_data[x_index][y_index] = 2
            elif grid_data[x_index][y_index] == 0:
                grid_data[x_index][y_index] = 1
            index += 1
        grid_data = grid_data.reshape([x_num, y_num, 1])

        result.append(grid_data.tolist())
    return result


# load_raster
def load_raster_data(file_path_list):
    x_data = []
    y_data = []
    pl_id = []
    for file_path in file_path_list:
        csvfile = open(file_path, newline='')
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for index, row in enumerate(reader):
            i_x_data = row[0:-1]
            i_y_data = [row[-1]]

            x_data.append(i_x_data)
            y_data.append(i_y_data)

    return x_data, y_data


def load_vector_data(file_path_list, data_num=50000):
    x_data = []
    y_data = []
    pl_id = []
    for file_path in file_path_list:
        csvfile = open(file_path, newline='')
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)

        for index, row in enumerate(reader):
            i_x_data = []
            if len(x_data) == data_num:
                break
            label = row[-1]
            for x, y in pairwise(row[1:-1]):
                i_x_data.append([x, y])

            x_data.append(i_x_data)
            y_data.append([label])
            pl_id.append(row[0])

    return x_data, y_data, pl_id


def make_decode_CSV_list(file_name_list, record_defaults):
    filename_queue = tf.train.string_input_producer(file_name_list, shuffle=False, name='filename_queue')

    reader = tf.TextLineReader()
    #reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    return tf.decode_csv(value, record_defaults=record_defaults)
