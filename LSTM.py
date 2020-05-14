import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tqdm import tqdm_notebook
import json
import pickle as pkl
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
import seaborn as sn

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
mpl.rc('font', **font)
# 1
##mpl.rc('figure', **{'figsize': (14,28)})
# 2,3,4
mpl.rc('figure', **{'figsize': (14,7)})
mpl.rc('lines', **{'linewidth': 5,
                   'color': 'y'})
bottom_cut = 0.15
left_cut = 0.3

data = pd.read_csv('data/train_training.csv')
data_test = pd.read_csv('data/train_testing.csv')

TIME_STEPS = 60
BATCH_SIZE = 10

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

def build_timeseries(mat):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))

    for i in tqdm_notebook(range(dim_0)):
        x[i] = mat[i:TIME_STEPS + i]
        # y[i] = mat[TIME_STEPS+i, y_col_index]
    print("length of time-series i/o", x.shape, y.shape)
    return x  # , y

def build_timeseries_y(mat):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))

    for i in tqdm_notebook(range(dim_0)):
        x[i] = mat[i + TIME_STEPS - 1:TIME_STEPS + i]
        # if i == 0:
        # y[i] = mat[TIME_STEPS+i, y_col_index]

    y = np.zeros((x.shape[0], 20))
    for i in range(x.shape[0]):
        y[i] = x[i][0]
        if i == 0:
            print(y[i])

    print("length of time-series i/o", y.shape)

    return y  # , y

def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat

def get_features_loss(features):
    X = data[features].values.astype(float)
    y = data[TARGET_COLUMNS].values.astype(float)
    X_test = data_test[features].values.astype(float)
    y_test = data_test[TARGET_COLUMNS].values.astype(float)

    x_t = build_timeseries(X)
    y_t = build_timeseries_y(y)
    x_t = trim_dataset(x_t, BATCH_SIZE)
    y_t = trim_dataset(y_t, BATCH_SIZE)

    x_val = build_timeseries(X_test)
    y_val = build_timeseries_y(y_test)
    x_val = trim_dataset(x_val, BATCH_SIZE)
    y_val = trim_dataset(y_val, BATCH_SIZE)

    lstm_model = Sequential()
    lstm_model.add(LSTM(20, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0,
                        recurrent_dropout=0.0, stateful=True, kernel_initializer='random_uniform'))
    lstm_model.add(Dense(20, activation='relu'))
    lstm_model.add(Dense(20, activation='softmax'))

    optimizer = optimizers.RMSprop(lr=0.0001)
    lstm_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    early_stopping = EarlyStopping(patience=150, verbose=1)
    model_checkpoint = ModelCheckpoint("./keras.model", save_best_only=True, monitor='val_loss', mode='auto',
                                       verbose=1, )
    reduce_lr = ReduceLROnPlateau(factor=0.25, patience=40, min_lr=0.000003, verbose=1)

    # ---------------------------------------------------------------------------------

    epochs = 50
    batch_size = 10

    history = lstm_model.fit(x_t, y_t,
                             validation_data=[x_val, y_val],
                             epochs=epochs,
                             batch_size=batch_size,
                             callbacks=[early_stopping, model_checkpoint, reduce_lr], shuffle=True)

    return min(history.history['val_loss'])

def get_lstm(features):
    X = data[features].values.astype(float)
    y = data[TARGET_COLUMNS].values.astype(float)
    X_test = data_test[features].values.astype(float)
    y_test = data_test[TARGET_COLUMNS].values.astype(float)

    x_t = build_timeseries(X)
    y_t = build_timeseries_y(y)
    x_t = trim_dataset(x_t, BATCH_SIZE)
    y_t = trim_dataset(y_t, BATCH_SIZE)

    x_valo = build_timeseries(X_test)
    y_valo = build_timeseries_y(y_test)
    x_val = trim_dataset(x_valo, BATCH_SIZE)
    y_val = trim_dataset(y_valo, BATCH_SIZE)

    lstm_model = Sequential()
    lstm_model.add(LSTM(20, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]), dropout=0.0,
                        recurrent_dropout=0.0, stateful=True, kernel_initializer='random_uniform'))
    lstm_model.add(Dense(20, activation='relu'))
    lstm_model.add(Dense(20, activation='softmax'))

    optimizer = optimizers.RMSprop(lr=0.0001)
    lstm_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    early_stopping = EarlyStopping(patience=150, verbose=1)
    model_checkpoint = ModelCheckpoint("./keras.model", save_best_only=True, monitor='val_loss', mode='auto',
                                       verbose=1, )
    reduce_lr = ReduceLROnPlateau(factor=0.25, patience=40, min_lr=0.000003, verbose=1)

    # ---------------------------------------------------------------------------------

    epochs = 50
    batch_size = 10

    history = lstm_model.fit(x_t, y_t,
                             validation_data=[x_val, y_val],
                             epochs=epochs,
                             batch_size=batch_size,
                             callbacks=[early_stopping, model_checkpoint, reduce_lr], shuffle=True)

    return lstm_model, x_valo, y_valo

#losses = {}

# Get individual feature losses
#for feature in FEATURE_COLUMNS:
#    losses[feature] = get_features_loss([feature])
#    with open('feature_losses.json', 'w') as fp:
#        json.dump(losses, fp)

# Sort individual feature losses (smallest losses will be first)
with open('feature_losses.json', 'rb') as fp:
    losses = json.load(fp)
losses = {k: v for k, v in sorted(losses.items(), key=lambda item: -item[1])}
plt.barh(np.arange(len(losses.keys())), losses.values(), color="y")
plt.title("Loss for LSTM Trained on Single Feature")
plt.xlabel("Categorical Cross-entropy Loss")
plt.ylabel("Feature Name")
plt.xlim((1.2, 2.2))
plt.yticks(np.arange(len(losses.keys())), losses.keys(), fontsize=18)
#plt.tight_layout()
plt.show()
plt.gcf().subplots_adjust(bottom=bottom_cut, left=left_cut)
#plt.savefig("1.png")

# Get loss for feature set selections based on feature scores.
# losses = {k: v for k, v in sorted(losses.items(), key=lambda item: item[1])}
# feature_vec = []
# maximum_losses = []
# feature_set_losses = []
# feature_sets = []
# results = {}
# for i, feature in enumerate(losses.keys()):
#     print("testing feature set " + str(i) + "/" + str(len(losses.keys())))
#     feature_vec.append(feature)
#     maximum_losses.append(losses[feature])
#     feature_set_losses.append(get_features_loss(feature_vec))
#     results["Maximum Loss for Features In Feature Set"] = maximum_losses
#     results["Feature Set Losses"] = feature_set_losses
#     results["Feature Sets"] = feature_vec
#     with open('feature_set_results.json', 'w') as fp:
#         json.dump(results, fp)

with open('feature_set_results.json', 'rb') as fp:
    results = json.load(fp)

#results = {k: v for k, v in sorted(results.items(), key=lambda item: item['Maximum Loss for Features In Feature Set'])}
plt.clf()
plt.title('Maximum Feature Loss of Features used')
plt.plot(results['Maximum Loss for Features In Feature Set'], results['Feature Set Losses'], color="y")
plt.xlabel("Maximum Validation Loss for LSTM Trained on Individual Feature")
plt.ylabel("Validation Loss")
plt.ylim((1.4,2.4))
plt.xlim((1.2,2.2))
plt.show()
#plt.tight_layout()
plt.gcf().subplots_adjust(bottom=bottom_cut, left=left_cut)
#plt.savefig("2.png")

### Finding goodness of feature catagories.

CENTER_2D = [
    'centre_2d_x',
    'centre_2d_y'
    ]
BB_2D = [
    'bb_2d_br_x',
    'bb_2d_br_y',
    'bb_2d_tl_x',
    'bb_2d_tl_y'
    ]
CENTER_3D = [
    'centre_3d_x',
    'centre_3d_y',
    'centre_3d_z'
    ]
BB_3D = [
    'bb_3d_brb_x',
    'bb_3d_brb_y',
    'bb_3d_brb_z',
    'bb_3d_flt_x',
    'bb_3d_flt_y',
    'bb_3d_flt_z'
    ]
ACCELERATION = [
    'acceleration_x',
    'acceleration_y',
    'acceleration_z'
    ]
SIGNAL_STRENGTHS = [
    'acceleration_Kitchen_AP',
    'acceleration_Lounge_AP',
    'acceleration_Upstairs_AP',
    'acceleration_Study_AP'
    ]
PIR = [
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
feature_sets = [CENTER_2D, BB_2D, CENTER_3D, BB_3D, ACCELERATION, SIGNAL_STRENGTHS, PIR]
# feature_type_results = {}
# for feature_set, f_s_n in zip(feature_sets,["2D Centers", "Bounding Boxes 2D",
#                                             "3D Centers", "Bounding Boxes 3D",
#                                             "Acceleration", "Room Specific Acceleration",
#                                             "PIR"]):
#     feature_type_results[f_s_n] = get_features_loss(feature_set)
#     with open('feature_set_losses.json', 'w') as fp:
#         json.dump(feature_type_results, fp)

with open('feature_set_losses.json', 'rb') as fp:
    fsl = json.load(fp)
fsl["Signal Strengths"] = fsl.pop("Room Specific Acceleration")
fsl = {k: v for k, v in sorted(fsl.items(), key=lambda item: -item[1])}
plt.clf()
plt.barh(np.arange(len(fsl.keys())), fsl.values(), color="y")
plt.title("Loss for LSTM Trained on Feature Groups")
plt.xlabel("Categorical Cross-entropy Loss")
plt.ylabel("Feature Group Name")
plt.xlim((1.2,2.2))
plt.yticks(np.arange(len(fsl.keys())), fsl.keys(), fontsize=18)
plt.show()
#plt.tight_layout()
plt.gcf().subplots_adjust(bottom=bottom_cut, left=left_cut)
#plt.savefig("3.png")

### ---------------------------------------------


# names_to_lists = {"2D Centers": CENTER_2D,
#                   "Bounding Boxes 2D": BB_2D,
#                   "3D Centers": CENTER_3D,
#                   "Bounding Boxes 3D": BB_3D,
#                   "Room Specific Acceleration": ROOM_SPECIFIC_ACCELERATION,
#                   "PIR": PIR,
#                   "Acceleration": ACCELERATION
#                   }
# feature_group_group_results = {}
# feature_vec = []
# maximum_losses = []
# feature_set_losses = []
# fsl = {k: v for k, v in sorted(fsl.items(), key=lambda item: item[1])}
# for i, feature_group_key in enumerate(fsl.keys()):
#     feature_vec += names_to_lists[feature_group_key]
#     maximum_losses.append(fsl[feature_group_key])
#     feature_set_losses.append(get_features_loss(feature_vec))
#     results["Maximum Loss for Features In Feature Set"] = maximum_losses
#     results["Feature Set Losses"] = feature_set_losses
#     results["Feature Sets"] = feature_vec
#     with open('feature_group_group_results.json', 'w') as fp:
#         json.dump(results, fp)

with open('feature_group_group_results.json', 'rb') as fp:
    fggr = json.load(fp)

plt.clf()
plt.title('Maximum Feature Loss of Feature Groups used')
plt.plot(fggr['Maximum Loss for Features In Feature Set'], fggr['Feature Set Losses'], color="y")
plt.xlabel("Maximum Validation Loss for LSTM Trained on Feature Groups Used")
plt.ylabel("Validation Loss")
plt.ylim((1.4,2.4))
plt.xlim((1.2,2.2))
plt.show()
#plt.tight_layout()
plt.gcf().subplots_adjust(bottom=bottom_cut, left=left_cut)
#plt.savefig("4.png")


feature_set = ACCELERATION+PIR
lstm, xval, yval = get_lstm(feature_set)
lstm_xval_yval = (lstm, xval, yval)
with open('best_lstm_stuff.pkl', 'wb') as handle:
    pkl.dump(lstm_xval_yval, handle)
# make results dictionary

with open('best_lstm_stuff.pkl', 'rb') as fp:
    lstm_stuff = pkl.load(fp)
lstm = lstm_stuff[0]
xval = lstm_stuff[1]
yval = lstm_stuff[2]

acc_res = {}
for key in TARGET_COLUMNS:
    acc_res[key] = {"number": 0,
                    "preds": {k: 0 for k in TARGET_COLUMNS}}
for x, y in zip(xval, yval):
    # see if data is unanimous
    unanimous = len([ay for ay in y if ay == 0]) == len(y) - 1
    if unanimous:
        #find out the correct label
        correct_label = TARGET_COLUMNS[np.where(y==1)[0][0]]
        acc_res[correct_label]["number"] += 1
        # get predicted label
        pred = lstm.predict(np.array([x]*BATCH_SIZE))[0]
        max_likelihood = TARGET_COLUMNS[np.where(pred==max(pred))[0][0]]
        acc_res[correct_label]["preds"][max_likelihood] += 1

with open('accuracy_data.pkl', 'wb') as handle:
    pkl.dump(acc_res, handle)
# confusion_matrix = []
# for k in TARGET_COLUMNS:
#     confusion_matrix.append( [acc_res[k]["preds"][pred] for pred in TARGET_COLUMNS ]  )
df_cm = pd.DataFrame(confusion_matrix, index=TARGET_COLUMNS, columns=TARGET_COLUMNS)
#plt.figure(figsize = (10,7))

actual = []
preds = []
for true in acc_res.keys():
    for pred in acc_res[true]["preds"].keys():
        for i in range(acc_res[true]["preds"][pred]):
            actual.append(true)
            preds.append(pred)
c_f = confusion_matrix(actual, preds, normalize='true')
fig, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(c_f, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(yticks=[l for l in list(range(len(TARGET_COLUMNS)))],
       xticks=[l for l in list(range(len(TARGET_COLUMNS)))],
       yticklabels=TARGET_COLUMNS,
       xticklabels=TARGET_COLUMNS)
plt.xticks(rotation=90)
#sn.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True)
plt.show()

accuracy = len([i for i in range(len(actual)) if actual[i]==preds[i]]) / len(actual)
maj_class_acc = len([i for i in range(len(actual)) if actual[i]=="p_stand"]) / len(actual)

print("accuracy is " + accuracy)
print("accuracy of majority class classifier is " + maj_class_acc)

exit(0)
