"""
Should train a linear model on the training data, then test accuracy against test data
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from keras.losses import CategoricalCrossentropy
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse
from scipy.special import softmax

TRAINING_DATA = "data/train_training.csv"
TESTING_DATA = "data/train_testing.csv"

def cross_entropy(p, q):
	return -sum([p[i]*np.log2(q[i]) for i in range(len(p))])

FEATURE_COLUMNS = [
    'sample',
    'sample_index',
    'start',
    'end',
    'centre_2d_x',
    'centre_2d_y',
    'bb_2d_br_x',
    'bb_2d_br_y',
    'bb_2d_tl_x',
    'bb_2d_tl_y',
    'centre_3d_x',
    'centre_3d_y',
    'centre_3d_z',
    'bb_3d_brb_x',
    'bb_3d_brb_y',
    'bb_3d_brb_z',
    'bb_3d_flt_x',
    'bb_3d_flt_y',
    'bb_3d_flt_z',
    'acceleration_x',
    'acceleration_y',
    'acceleration_z',
    'acceleration_Kitchen_AP',
    'acceleration_Lounge_AP',
    'acceleration_Upstairs_AP',
    'acceleration_Study_AP',
    'pir_bath',
    'pir_bed1',
    'pir_bed2',
    'pir_hall',
    'pir_kitchen',
    'pir_living',
    'pir_stairs',
    'pir_study',
    'pir_toilet'
]

TARGET_COLUMNS = [
    'a_ascend',
    'a_descend',
    'a_jump',
    'a_loadwalk',
    'a_walk',
    'p_bent',
    'p_kneel',
    'p_lie',
    'p_sit',
    'p_squat',
    'p_stand',
    't_bend',
    't_kneel_stand',
    't_lie_sit',
    't_sit_lie',
    't_sit_stand',
    't_stand_kneel',
    't_stand_sit',
    't_straighten',
    't_turn'
]


### SIMPLE LINEAR REGRESSION

csv = pd.read_csv(TRAINING_DATA)

x = csv[FEATURE_COLUMNS]
targets = csv[TARGET_COLUMNS]

lin_reg = LinearRegression().fit(x, targets)
lin_score = lin_reg.score(x, targets)

### LINEAR REGRESSION WITH KERNEL

csv = pd.read_csv(TRAINING_DATA)

x = csv[FEATURE_COLUMNS]
targets = csv[TARGET_COLUMNS]

ker_lin_reg = KernelRidge().fit(x, targets)
ker_score = ker_lin_reg.score(x, targets)

### LINEAR SVR

csv = pd.read_csv(TRAINING_DATA)

x = csv[FEATURE_COLUMNS]
targets = csv[TARGET_COLUMNS]

#-----------------------------------------------------------------------

test_csv = pd.read_csv(TESTING_DATA)
loss = CategoricalCrossentropy(from_logits=True)

x = test_csv[FEATURE_COLUMNS]
targets = test_csv[TARGET_COLUMNS].to_numpy()

preds = lin_reg.predict(x)
lin_reg_loss = float(loss(targets, preds))
print("loss for simple linear regression is: " + str(lin_reg_loss))

preds = ker_lin_reg.predict(x)
kernal_ridge_loss = float(loss(targets, preds))
print("loss for kernel ridge linear regression is: " + str(kernal_ridge_loss))

print(0)