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
import numpy as np
import matplotlib.pyplot as plt

# get train dataframe
sample_dirs = os.listdir(TRAIN_PATH)
sample_dirs.remove('.DS_Store')
print(sample_dirs)
train_frame = get_training_samples_frame(TRAIN_PATH)
train_frame.to_csv('train_frame.csv')

# get test dataframe
test_frame = get_testing_samples_frame(TEST_PATH)
test_frame.to_csv('test_frame.csv')

# read csv files
train_frame = pd.read_csv('train_frame.csv')
test_frame = pd.read_csv('test_frame.csv')

# acceleration
acceleration_x = train_frame['acceleration_x']
acceleration_y = train_frame['acceleration_y']
acceleration_z = train_frame['acceleration_z']

# create posture df and label each posture
postures = ['a_ascend','a_descend','a_jump','a_loadwalk','a_walk','p_bent','p_kneel',
			'p_lie','p_sit','p_squat','p_stand','t_bend','t_kneel_stand','t_lie_sit',
			't_sit_lie','t_sit_stand','t_stand_kneel','t_stand_sit','t_straighten','t_turn']
posture_df = train_frame[postures]
posture_label = posture_df.idxmax(axis=1)
start = 0
posture_count = [0] * len(posture_label)
for i in range(len(posture_label)):
	if i == 0:
		posture_count[i] = start
	elif posture_label[i] == posture_label[i-1]:
		posture_count[i] = posture_count[i-1]
	else:
		posture_count[i] = posture_count[i-1] + 1


# create acceleration df for plotting
acceleration_df = train_frame[["acceleration_x","acceleration_y",'acceleration_z']]
acceleration_df['posture_count'] = posture_count
acceleration_df['posture_label'] = posture_label
print(acceleration_df)


npostures = max(posture_count)

for posture in postures:
	posture_accel_df = acceleration_df[posture_label==posture]
	posture_count_subset = posture_accel_df.posture_count.unique()
	posture_time = []

	# loop through posture counts and plot
	for i in posture_count_subset:

		posture_i = posture_accel_df.loc[posture_accel_df["posture_count"] == i]
		print(posture_i)
		# record how long each posture takes
		t = len(posture_i)
		posture_time.append(t)


		
		plt.plot(range(t),posture_i['acceleration_x'])

	plt.savefig('figures/posture_acceleration_%s.png' % posture)
	plt.clf()

	mean_posture_time = np.mean(posture_time)
	print("mean time for %s:" % posture,mean_posture_time)
	