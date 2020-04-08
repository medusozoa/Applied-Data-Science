import os
import pandas as pd
import math

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

PIR_LOCATIONS = ['bath', 'bed1', 'bed2', 'hall', 'kitchen', 'living', 'stairs', 'study', 'toilet']


# reshapes data into 1 second intervals, averages into 1 second bins, fills NaN for missing seconds
# has a start and end that are a second apart as is data in targets.csv
def series_align_time(series):
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
    series[VIDEO_2D_X_COLS] = series[VIDEO_2D_X_COLS].fillna(-640)
    series[VIDEO_2D_Y_COLS] = series[VIDEO_2D_Y_COLS].fillna(-480)

    return series


def video_fillna_3d(series):
    series[VIDEO_3D_COLS] = series[VIDEO_3D_COLS].fillna(0)
    return series


# reads the different video CSVs and preprocesses them, returns them as a dict
def prepare_video_files(base_path, sample_name):
    videos = {}
    for location in LOCATION_QUADRANTS:
        video = pd.read_csv(f"{base_path}/{sample_name}/video_{location}.csv")
        video = series_align_time(video)
        video = video_transform_2d_grid(video, location)
        videos[location] = video

    return videos


# combines the already aligned features into a single data frame
def combine_with_video_features(targets, videos):
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


def prepare_acceleration(base_path, sample_name):
    acceleration = pd.read_csv(f"{base_path}/{sample_name}/acceleration.csv")
    acceleration = series_align_time(acceleration)
    acceleration.columns = [f"acceleration_{str(col)}" if col not in ['start', 'end'] else str(col)
                            for col in acceleration.columns]
    return acceleration


def combine_with_acceleration_features(targets, acceleration):
    data = pd.merge(targets, acceleration, how='left',
                    on=['start', 'end'],
                    left_index=True, right_index=False,
                    copy=True, indicator=False,
                    validate='one_to_one')
    data.reset_index(drop=True, inplace=True)
    data.fillna(0, inplace=True)
    return data


def prepare_pir(base_path, sample_name):
    pir = pd.read_csv(f"{base_path}/{sample_name}/pir.csv")

    min_t = math.floor(pir['start'].min()) if len(pir) > 0 else 0
    max_t = math.ceil(pir['end'].max()) if len(pir) > 0 else 0

    intervals = pd.interval_range(start=min_t, end=max_t)
    data = pd.DataFrame(False, index=range(len(intervals)), columns=[f"pir_{x}" for x in PIR_LOCATIONS])
    for _, row in pir.iterrows():
        data_row = f"pir_{row['name']}"
        data[data_row] |= intervals.overlaps(pd.Interval(left=row['start'], right=row['end']))

    data.insert(0, "start", intervals.left)
    data.insert(1, "end", intervals.right)

    return data


def combine_with_pir_features(targets, pir):
    data = pd.merge(targets, pir, how='left',
                    on=['start', 'end'],
                    left_index=True, right_index=False,
                    copy=True, indicator=False,
                    validate='one_to_one')
    data.reset_index(drop=True, inplace=True)
    data.fillna(False, inplace=True)
    return data


def prepare_targets(base_path, sample_name):
    targets = pd.read_csv(f"{base_path}/{sample_name}/targets.csv")
    # targets.dropna(how='all', subset=ANNOTATION_NAMES, inplace=True)

    first_idx = targets[ANNOTATION_NAMES].first_valid_index()
    last_idx = targets[ANNOTATION_NAMES].last_valid_index()
    targets = targets.loc[first_idx:last_idx]
    targets.fillna(1.0 / len(ANNOTATION_NAMES), inplace=True)

    return targets


# returns a prerprocessed dataframe of all features for a specific training sample
def prepare_training_sample(base_path, sample_name):
    targets = prepare_targets(base_path, sample_name)

    videos = prepare_video_files(base_path, sample_name)
    sample_data = combine_with_video_features(targets, videos)

    acceleration = prepare_acceleration(base_path, sample_name)
    sample_data = combine_with_acceleration_features(sample_data, acceleration)

    pir = prepare_pir(base_path, sample_name)
    sample_data = combine_with_pir_features(sample_data, pir)

    return sample_data


def min_time_in_features(videos, acceleration, pir):
    min_t = float('inf')

    for video in videos.values():
        min_t = min(min_t, video['start'].min())

    min_t = min(min_t, acceleration['start'].min())

    min_t = min(min_t, pir['start'].min())

    return min_t


def max_time_in_features(videos, acceleration, pir):
    max_t = 0

    for video in videos.values():
        max_t = max(max_t, video['end'].max())

    max_t = max(max_t, acceleration['end'].max())

    max_t = max(max_t, pir['end'].max())

    return max_t


def prepare_test_sample(base_path, sample_name):
    videos = prepare_video_files(base_path, sample_name)
    acceleration = prepare_acceleration(base_path, sample_name)
    pir = prepare_pir(base_path, sample_name)

    min_t = math.floor(min_time_in_features(videos, acceleration, pir))
    max_t = math.ceil(max_time_in_features(videos, acceleration, pir))

    sample_data = pd.DataFrame(data={'start': range(min_t, max_t),
                                     'end': range(min_t + 1, max_t + 1)})

    sample_data = combine_with_video_features(sample_data, videos)
    sample_data = combine_with_acceleration_features(sample_data, acceleration)
    sample_data = combine_with_pir_features(sample_data, pir)

    return sample_data


def get_samples_dict(base_path, preprocessing_func):
    sample_dirs = os.listdir(base_path)
    sample_dirs.sort()

    dfs = {}
    for sample in sample_dirs:
        df = preprocessing_func(base_path, sample)
        dfs[sample] = df

    return dfs


def get_samples_frame(base_path, preprocessing_func):
    sample_dirs = os.listdir(base_path)
    sample_dirs.sort()

    dfs = []
    for sample in sample_dirs:
        df = preprocessing_func(base_path, sample)
        df.insert(0, 'sample', sample)
        df.insert(1, 'sample_index', df.index)
        dfs.append(df)

    # Concatenate all data into one DataFrame
    return pd.concat(dfs, ignore_index=True)


def get_training_samples_dict(base_path):
    return get_samples_dict(base_path, prepare_training_sample)


def get_training_samples_frame(base_path):
    return get_samples_frame(base_path, prepare_training_sample)


def get_testing_samples_dict(base_path):
    return get_samples_dict(base_path, prepare_test_sample)


def get_testing_samples_frame(base_path):
    return get_samples_frame(base_path, prepare_test_sample)
