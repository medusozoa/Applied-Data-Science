'''
Functions for checking the data.
'''

#------------------------------------------------------------------

import util.utilities as ut

import pandas as pd
import chart_studio.plotly as py
import plotly.figure_factory as ff
import plotly as ply

#------------------------------------------------------------------

TRAIN_PATH = "data/train/"
TRAIN_DATA_DIRECTORIES = [TRAIN_PATH + subdir + "/" for subdir
                          in ut.get_sub_directories(TRAIN_PATH)]
TEST_PATH = "data/test/"
TEST_DATA_DIRECTORIES = [TEST_PATH + subdir + "/" for subdir
                         in ut.get_sub_directories(TEST_PATH)]

LOCATIONS = ['bath', 'bed1', 'bed2', 'hall', 'kitchen', 'living', 'stairs', 'study', 'toilet']
ANNOTATIONS = ['a_ascend', 'a_descend', 'a_jump', 'a_loadwalk', 'a_walk', 'p_bent', 'p_kneel',
               'p_lie', 'p_sit', 'p_squat', 'p_stand', 't_bend', 't_kneel_stand', 't_lie_sit',
               't_sit_lie', 't_sit_stand', 't_stand_kneel', 't_stand_sit', 't_straighten', 't_turn']

#------------------------------------------------------------------


def get_annotation_paths():
    all_train_paths = [ut.get_file_names(t_d_d) for t_d_d in TRAIN_DATA_DIRECTORIES]
    annotation_paths = []
    for path_list in all_train_paths:
        annotation_paths += [p for p in path_list if "annotations_" in p]
    return annotation_paths

def get_location_paths():
    all_train_paths = [ut.get_file_names(t_d_d) for t_d_d in TRAIN_DATA_DIRECTORIES]
    location_paths = []
    for path_list in all_train_paths:
        location_paths += [p for p in path_list if "location_" in p]
    return location_paths

def check_annotation_file(path):
    csv = pd.read_csv(path)
    # Check that mappings between `name` and `index` are consistent and correct
    names = list(csv["name"])
    indexes = list(csv["index"])
    for i, index in enumerate(indexes):
        if ANNOTATIONS[index] != names[i]:
            print("ERROR: inconsistent mapping between `name` and `index` in " + path)
            exit(0)

def check_location_file(path):
    csv = pd.read_csv(path)
    # Check that mappings between `name` and `index` are consistent and correct
    names = list(csv["name"])
    indexes = list(csv["index"])
    for i, index in enumerate(indexes):
        if LOCATIONS[index] != names[i]:
            print("ERROR: inconsistent mapping between `name` and `index` in " + path)
            exit(0)

def viz_locations(path):
    csv = pd.read_csv(path)
    # Check that mappings between `name` and `index` are consistent and correct
    names = list(csv["name"])
    starts = list(csv["start"])
    ends = list(csv["end"])
    df = [dict(Task=names[i], Start=starts[i], Finish=ends[i]) for i in range(len(names))]
    fig = ff.create_gantt(df, group_tasks=True)
    fig.write_image("visualisation/location.png")

if __name__ == "__main__":
    annotation_paths = get_annotation_paths()
    for a_p in annotation_paths:
        check_annotation_file(a_p)
    location_paths = get_location_paths()
    for l_p in location_paths:
        check_location_file(l_p)
        #vizualize if you want?
        #viz_locations(l_p)
    exit(1)