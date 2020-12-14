# Libraries for Path Management and Handling Directories
import os
from os import makedirs
from os import listdir
from shutil import copyfile
from shutil import rmtree
from random import seed
from random import random

# Main Directory, adjust according to your File System Structure
os.chdir("C:/github_simba_ex2/")

# Set Seed for Generation of Random Numbers for splitting files into Training and Validation
seed(5555)
# define ratio of pictures to use for validation
val_ratio = 0.3
test_ratio = 0 # Not used in this implementation. Test data are manually provided in specific subdirectories

# Static Variables for Folder Structure
src_directory = 'images_all/training_small_140/' # From which directory to take images from
# src_directory = 'images_all/training_all/'
dst_directory = 'images_model/'
categories = ['goldens','non_goldens','simba']
sub_directories = ['train','validation','test']

# Delete Folders (to get rid of content) and create Subfolders
rmtree(dst_directory)
makedirs(dst_directory)
for dir in sub_directories:
    makedirs(dst_directory + '/' + dir)
    for cat in categories:
        makedirs(dst_directory + dir + '/' + cat)

# Copy Images for Training, Valiation and Test Folder
for category in categories:
    for file in listdir(src_directory + category):
        src = src_directory  + category + '/' + file
        if random() < val_ratio:
            dst = dst_directory + 'validation/' + category + '/' + file
            copyfile(src, dst)
        elif random() > (1-test_ratio):
            dst = dst_directory + 'test/' + category + '/' + file
            copyfile(src, dst)
        else:
            dst = dst_directory + 'train/' + category + '/' + file
            copyfile(src, dst)


