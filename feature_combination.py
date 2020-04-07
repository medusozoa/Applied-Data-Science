import os
import pandas as pd

ANNOTATION_NAMES = ['a_ascend', 'a_descend', 'a_jump', 'a_loadwalk', 'a_walk', 'p_bent', 'p_kneel', 'p_lie', 'p_sit',
                    'p_squat', 'p_stand', 't_bend', 't_kneel_stand', 't_lie_sit', 't_sit_lie', 't_sit_stand',
                    't_stand_kneel', 't_stand_sit', 't_straighten', 't_turn']

VIDEO_VALUE_COLS = ['centre_2d_x', 'centre_2d_y', 'bb_2d_br_x', 'bb_2d_br_y', 'bb_2d_tl_x', 'bb_2d_tl_y', 'centre_3d_x',
                    'centre_3d_y', 'centre_3d_z', 'bb_3d_brb_x', 'bb_3d_brb_y', 'bb_3d_brb_z', 'bb_3d_flt_x',
                    'bb_3d_flt_y', 'bb_3d_flt_z']

VIDEO_2D_X_COLS = ['centre_2d_x', 'bb_2d_br_x', 'bb_2d_tl_x']
VIDEO_2D_Y_COLS = ['centre_2d_y', 'bb_2d_br_y', 'bb_2d_tl_y']

VIDEO_3D_COLS = ['centre_3d_x', 'centre_3d_y', 'centre_3d_z', 'bb_3d_brb_x', 'bb_3d_brb_y', 'bb_3d_brb_z',
                   'bb_3d_flt_x','bb_3d_flt_y', 'bb_3d_flt_z']

TRAIN_PATH = r'data/train'
TEST_PATH = r'data/test'

HALLWAY = 'hallway'
LIVING_ROOM = 'living_room'
KITCHEN = 'kitchen'
LOCATION_QUADRANTS = {HALLWAY: (1, 1),
                      LIVING_ROOM: (1, -1),
                      KITCHEN: (-1, 1)
                      # none : (-1, -1)
                      }


# reshapes data into 1 second intervals, averages into 1 second bins, fills NaN for missing seconds
# has a start and end that are a second apart as is data in targets.csv
def video_align_time(series):
    series.insert(2, 'timestamp', pd.to_datetime(series['t'], unit='s'))
    del series['t']

    series = series.set_index('timestamp').resample('1S').mean().reset_index()
    series.insert(1, 'start', series['timestamp'].astype(int) / 10 ** 9)
    series.insert(2, 'end', series['start'] + 1.0)
    del series['timestamp']
    return series


# flips video into its own quadrant of the x,y space
def video_transform_2d_grid(series, location):
    x_mult, y_mult = LOCATION_QUADRANTS[location]
    series[VIDEO_2D_X_COLS] *= x_mult
    series[VIDEO_2D_Y_COLS] *= y_mult
    return series


# absent video is put in the corner of the negative quadrant of the x,y space
def video_fillna_2d(series):
    row_indexer = series[ANNOTATION_NAMES].notna().all(axis=1)
    series[VIDEO_2D_X_COLS] = series.loc[row_indexer, VIDEO_2D_X_COLS].fillna(-640)
    series[VIDEO_2D_Y_COLS] = series.loc[row_indexer, VIDEO_2D_Y_COLS].fillna(-480)

    return series


def video_fillna_3d(series):
    series[VIDEO_3D_COLS] = series[VIDEO_3D_COLS].fillna(0)
    return series


# reads the different video CSVs and preprocesses them, returns them as a dict
def prepare_video_files(base_path, sample_name):
    videos = {}
    for location in LOCATION_QUADRANTS:
        video = pd.read_csv(f"{base_path}/{sample_name}/video_{location}.csv")
        video = video_align_time(video)
        video = video_transform_2d_grid(video, location)
        videos[location] = video

    return videos


# combines the already aligned features into a single data frame
def combine_features(targets, videos):
    data = pd.merge(targets, videos[HALLWAY], how='left',
                    on=['start', 'end'],
                    left_index=True, right_index=False,
                    copy=True, indicator=False,
                    validate='one_to_one')
    data.reset_index(drop=True, inplace=True)

    data.set_index(['start', 'end'], inplace=True)

    living_room_vid = videos[LIVING_ROOM].set_index(['start', 'end'])
    data.update(living_room_vid)

    kitchen_vid = videos[KITCHEN].set_index(['start', 'end'])
    data.update(kitchen_vid)

    data.reset_index(inplace=True)

    data = video_fillna_2d(data)
    data = video_fillna_3d(data)
    return data


# returns a prerprocessed dataframe of all features for a specific training sample
def prepare_training_sample(base_path, sample_name):
    targets = pd.read_csv(f"{base_path}/{sample_name}/targets.csv")
    targets.dropna(how='all', subset=ANNOTATION_NAMES, inplace=True)
    videos = prepare_video_files(base_path, sample_name)

    sample_data = combine_features(targets, videos)

    return sample_data


def get_file_for_all_samples(path, file, preprocess_func):
    sample_dirs = os.listdir(path)
    sample_dirs.sort()

    dfs = []
    for sample in sample_dirs:
        df = pd.read_csv(f"{path}/{sample}/{file}")
        df = preprocess_func(df)
        df.insert(0, 'sample', sample)
        df.insert(1, 'sample_index', df.index)
        dfs.append(df)

    # Concatenate all data into one DataFrame
    return pd.concat(dfs, ignore_index=True)
