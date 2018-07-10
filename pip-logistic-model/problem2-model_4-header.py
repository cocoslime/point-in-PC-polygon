import csv

header = __import__("problem2-model_4-header")

# polygon data will grid by WIDTH_NUM X HEIGHT_NUM
WIDTH_NUM = 32
HEIGHT_NUM = 32

LEARNING_RATE = 0.001
TRAINING_EPOCHS = 501
CAPACITY = 50000
MIN_AFTER_DEQUEUE = 10000
BATCH_SIZE = 100

TEST_SIZE = 10000

DATA_DIR = "../data/problem2/simple/"


# load_raster
def load_raster_data_in_array(file_path):
    x_data = []
    y_data = []
    index_data = []
    csvfile = open(file_path, newline='')
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for index, row in enumerate(reader):
        i_x_data = row[1:-1]
        i_y_data = [row[-1]]
        _index = row[0]

        x_data.append(i_x_data)
        y_data.append(i_y_data)
        index_data.append(_index)

    return x_data, y_data, index_data

