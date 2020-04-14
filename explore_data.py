"""
This script loads all data into a dataframe and explores:

	graphs of acceleration

"""

# import modules
from feature_combination import TRAIN_PATH, TEST_PATH
from feature_combination import get_training_samples_dict, get_training_samples_frame
from feature_combination import get_testing_samples_dict, get_testing_samples_frame
import os
import pandas as pd

# get train dataframe
sample_dirs = os.listdir(TRAIN_PATH)
sample_dirs.remove('.DS_Store')
print(sample_dirs)
train_frame = get_training_samples_frame(TRAIN_PATH)
train_frame.to_csv('train_frame.csv')

# get test dataframe
test_frame = get_testing_samples_frame(TEST_PATH)
test_frame.to_csv('test_frame.csv')