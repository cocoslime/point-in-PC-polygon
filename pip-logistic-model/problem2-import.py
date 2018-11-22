def load_points_dir(DATA_DIR, BUFFER_OPT):
    TRAINING_FILE_ARR = []
    TEST_FILE_ARR = []
    PNUM_ARR = [50, 75, 100, 150, 200, 250, 300, 350, 700]
    for POINTS_NUM in PNUM_ARR:
        POINTS_NUM = "p" + str(POINTS_NUM)
        TRAINING_FILEPATH = DATA_DIR + POINTS_NUM + "_training_" + BUFFER_OPT + ".csv"
        TEST_FILEPATH = DATA_DIR + POINTS_NUM + "_test_" + BUFFER_OPT + ".csv"
        TRAINING_FILE_ARR.append(TRAINING_FILEPATH)
        TEST_FILE_ARR.append(TEST_FILEPATH)
    return TRAINING_FILE_ARR, TEST_FILE_ARR


def load_density_dir(path, dirname, buffer_opt):
    TRAINING_FILE_ARR = [path + dirname + "/training_" + buffer_opt + ".csv"]
    TEST_FILE_ARR = [path + dirname + "/test_" + buffer_opt + ".csv"]
    return TRAINING_FILE_ARR, TEST_FILE_ARR
