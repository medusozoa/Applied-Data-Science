# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl # in python
from feature_combination import TRAIN_PATH, TEST_PATH
from feature_combination import get_training_samples_dict, get_training_samples_frame
from feature_combination import get_testing_samples_dict, get_testing_samples_frame

# read csv files
train_frame = pd.read_csv('train_frame.csv')

features = ['centre_2d_x','centre_2d_y','bb_2d_br_x','bb_2d_br_y','bb_2d_tl_x','bb_2d_tl_y','centre_3d_x','centre_3d_y',
			'centre_3d_z','bb_3d_brb_x','bb_3d_brb_y','bb_3d_brb_z','bb_3d_flt_x','bb_3d_flt_y','bb_3d_flt_z','acceleration_x',
			'acceleration_y','acceleration_z','acceleration_Kitchen_AP','acceleration_Lounge_AP','acceleration_Upstairs_AP',
			'acceleration_Study_AP','pir_bath','pir_bed1','pir_bed2','pir_hall','pir_kitchen','pir_living','pir_stairs','pir_study','pir_toilet','a_ascend','a_descend','a_jump','a_loadwalk','a_walk','p_bent','p_kneel','p_lie','p_sit','p_squat','p_stand','t_bend','t_kneel_stand','t_lie_sit','t_sit_lie','t_sit_stand','t_stand_kneel','t_stand_sit','t_straighten','t_turn']

df = train_frame[features]

# colourmap
# C=[1,2,3:4,5,6]

# C = [(0.4078,0.20784,0.85490),(1,0.91372,0.929411),(1,0.5176,0.25098)]
C = [(81, 201, 231),(255,255,255),(231, 111, 81)]
# C = [(63, 158, 181),(255,255,255),(231, 111, 81)]
# C = [(97, 178, 244),(255,255,255),(244, 163, 97)]
colours= []
for c in C:
	c = tuple([x/255 for x in c])
	colours.append(c)

n_bin = 100

	
cm = mpl.colors.LinearSegmentedColormap.from_list(
        'my_cmap', colours, N=n_bin)
f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number,cmap=cm,vmax=1,vmin=-1)
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Feature Correlation Matrix', fontsize=26,y=1.25)
plt.savefig('figures/correlation_matrix2.png',bbox_inches='tight')