'''
Utility functions.
'''

#------------------------------------------------------------------

import os

#------------------------------------------------------------------

#-----------------------------------------------------------------------------------------
# file functions
#-----------------------------------------------------------------------------------------

def get_sub_directories(directory):
    """
    Returns sub directory names.
    """
    return list(os.walk(directory))[0][1]

def get_file_names(directory):
    """
    Returns names of all the files in the directory.
    """
    return [directory + file for file in list(os.walk(directory))[0][2]]