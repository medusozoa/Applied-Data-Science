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
npostures = max(posture_count)
posture_durations = []
mins = []
maxs = []
standard_devs = []
posture_times = []

for posture in postures:
    posture_accel_df = acceleration_df[posture_label==posture]
    posture_count_subset = posture_accel_df.posture_count.unique()
    posture_time = []

	# loop through posture counts and plot
    for i in posture_count_subset:
        posture_i = posture_accel_df.loc[posture_accel_df["posture_count"] == i]
        # record how long each posture takes
        t = len(posture_i)
        posture_time.append(t)

    mean_posture_time = np.mean(posture_time)
    print("mean time for %s:" % posture,mean_posture_time)
    posture_durations.append(round(mean_posture_time,2))
    standard_dev = round(np.std(posture_time),2)
    standard_devs.append(standard_dev)
    mins.append(round(min(posture_time)))
    maxs.append(round(max(posture_time)))
    posture_times.append(posture_time)




plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(postures))


# define posture categories and colour code them
a_ = ['a_ascend', 'a_descend', 'a_jump', 'a_loadwalk', 'a_walk']
p_ = ['p_bent', 'p_kneel', 'p_lie', 'p_sit', 'p_squat', 'p_stand']
t_ = ['t_bend', 't_kneel_stand', 't_lie_sit', 't_sit_lie', 't_sit_stand', 't_stand_kneel', 't_stand_sit','t_straighten', 't_turn']
category = []
colours = []
for posture in postures:
	if posture in a_:
		colours.append('#f4a261')
		category.append('2a_')
	elif posture in p_:
		colours.append('#e76f51')
		category.append('3p_')
	elif posture in t_:
		colours.append('#e9c46a')
		category.append('1t_')

postures = ['Ascend','Descend', 'Jump', 'Loadwalk', 'Walk','Bent', 'Kneel', 'Lie', 'Sit', 'Squat', 'Stand','Bend', 'Kneel to Stand', 'Lie to Sit', 'Sit to Lie', 'Sit to Stand', 'Stand to Kneel', 'Stand to Sit','Straighten', 'Turn']
df = pd.DataFrame({'postures':postures, 'duration':posture_durations,'colour':colours,'category':category,'error':standard_devs,'mins':mins,'maxs':maxs})
df = df.sort_values(by=['category','duration'])
postures = df['postures']
posture_durations = df['duration']
colours = df['colour']
error = df['error']
mins = df['mins']
maxs = df['maxs']

# ax.barh(y_pos, posture_durations, align='center',color=colours) #xerr = error

posture_times = np.array([np.array(xi) for xi in posture_times])

for i,category in enumerate(['2a_',"3p_","1t_"]):
    colours = ['#f4a261','#e76f51','#e9c46a']
    darkercolours = ['#c58654','#b65840','#c1a359']
    colour = colours[i]
    dcolour = darkercolours[i]
    print(i)
    print(category)
    indices = list(df.index[df['category']==category])
    print(list(df.index[df['category']==category]))
    plt.boxplot(posture_times[indices],0,'rh',0, # posture times indexed by original index plt. or ax.?
                    # positions=list(df.index[df['category']==category]),
                    positions=y_pos[df['category']==category], # y position indexed by new indices
                    showfliers=False,
                    showmeans=True,
                    patch_artist=True,
 					boxprops=dict(facecolor=colour, color=colour),
             		capprops=dict(color=colour),
             		whiskerprops=dict(color=colour),
                    
             		medianprops=dict(color=dcolour),
                    meanprops = dict(marker='o', markeredgecolor=dcolour,markerfacecolor=dcolour))


# bplots = ax.boxplot(posture_times,0,'rh',0,positions=df.index,showfliers=False,patch_artist=True,
# 					boxprops=dict(facecolor=colours, color=colours),
#             		capprops=dict(color=colours),
#             		whiskerprops=dict(color=colours),
#             		medianprops=dict(color=colours))


# for bplot in bplots:
#     for patch, color in zip(bplot['boxes'], colours):
#         patch.set_facecolor(color)

ax.set_yticks(y_pos)
ax.set_yticklabels(postures)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Duration (seconds)')
ax.set_title('Average posture duration of training set')
plt.savefig('figures/posture_duration.png')
plt.clf()

# plt.rcdefaults()
# fig, ax = plt.subplots()
# y_pos = np.arange(len(postures))
# ax.boxplot(y_pos,posture_times)
# plt.savefig('figures/posture_duration_boxplot.png')
