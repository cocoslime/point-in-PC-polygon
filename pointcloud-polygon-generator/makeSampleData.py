import matplotlib.pyplot as plt
import random
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
import scipy.stats as stats
import csv

N = 200000

sampleData = open("../SampleData.csv", 'w', encoding='utf-8', newline='')
data_writer = csv.writer(sampleData)

data_writer.writerow([N])


sampleData.close()