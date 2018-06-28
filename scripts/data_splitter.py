#----------------------------------------------
#   USAGE: this script will take in all of the
#       numpy array files created by the pipe-
#       line and will create a hold-out data set
#       and also do a test/train split with
#       proper dir structure to use with Keras
#       
#       
#
#   INPUTS:
#       1 --> relative path to holdout dir
#          
# 
#   OUTPUTS:
#       holdout set of np arrays, not_holdout set
#       thats split into train/test folders,
#       test and train folders are split into their
#       labels (bird/no_bird)
#
#   /Users/vineetkuroodi/Desktop/Bird_No_Bird/BirdVox/hold_out_data
#
#----------------------------------------------



#----------------------------------------------
#   imports
#----------------------------------------------
import sys
import os
import numpy as np
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split


CONST_DATA_SIZE = 20000



#----------------------------------------------
#   main
#----------------------------------------------
def main():
    # get dir paths
    all_directories = get_dirs()
    # decode dir paths 
    hold_out_dir = all_directories[0]
    not_holdout_dir = all_directories[1]
    target_dir = all_directories[2]
    # check to see if everything is correct
    print("Checking some directories...")
    print("\thold_out_dir: {}".format(hold_out_dir))
    print("\tnot_holdout_dir: {}".format(not_holdout_dir))
    print("\ttarget_dir: {}".format(target_dir))
    
    user_answer = input("\nare the above directories correct (y/n): ")

    if user_answer == 'y':
        pass
    else:
        exit()

    # get master list of all files
    all_files = get_files(target_dir)
    # remove hold out set of 5% and work with remaining 95%
    train_files = split_holdout_set(hold_out_dir, all_files, target_dir)
    # test/train split remaining data 80/20
    split_not_holdout_set(not_holdout_dir, train_files, all_files, target_dir)




#----------------------------------------------
#   get_dirs
#----------------------------------------------
def get_dirs():
    user_path = input("Please input path to hold out directory: ")
    hold_out_dir = os.fsencode(user_path).decode('UTF-8')
    bird_vox_dir = "/".join(hold_out_dir.split('/')[:-1])
    not_holdout_dir = bird_vox_dir + "/not_holdoutd_data"

    user_choice = input("\nPlease enter target output (arr/spec): ")
    if user_choice == 'arr':
        target_dir = bird_vox_dir + "/arrays"
    else:
        target_dir = bird_vox_dir + "/spectrograms_grey_scale"

    return hold_out_dir, not_holdout_dir, target_dir


#----------------------------------------------
#   split_not_holdout_set
#----------------------------------------------
def split_not_holdout_set(not_holdout_dir, not_holdout_files, all_files, arrays_dir):
    print_header("creating a not_holdout set", "80/20 test train split")
    print("\tsetting up not_hold_out dir...")
    # create directory paths
    test_set_dir = not_holdout_dir + "/test_set"
    training_set_dir = not_holdout_dir + "/training_set"
    test_bird_dir = test_set_dir + "/bird"
    test_no_bird_dir = test_set_dir + "/no_bird"
    training_bird_dir = training_set_dir + "/bird"
    training_no_bird_dir = training_set_dir + "/no_bird"
    #create directories if they don't exist
    init_dirs(not_holdout_dir)
    init_dirs(test_set_dir)
    init_dirs(training_set_dir)
    init_dirs(test_bird_dir)
    init_dirs(test_no_bird_dir)
    init_dirs(training_bird_dir)
    init_dirs(training_no_bird_dir)
    # populate not_holdout dir
    print("\tpopulating not_holdout dir...")
    train_list, test_list = train_test_split(not_holdout_files, test_size=0.20, random_state=0)
    print("\tlength of train: {}".format(len(train_list)))
    print("\tlength of test: {}".format(len(test_list)))
    #populate train and test sub-directories
    train_list = set(train_list)
    test_list = set(test_list)
    df = pd.read_csv('../data/BirdVox_Labels.csv')

    for not_hold_out_file in os.listdir(arrays_dir):
        if not_hold_out_file in train_list:
            if df[df['itemid'] == not_hold_out_file.split('.')[0]]['hasbird'].values[0]:
                source = arrays_dir + '/' + not_hold_out_file
                destination = training_bird_dir + "/" + not_hold_out_file
                shutil.copy(source, destination)
            else:
                source = arrays_dir + '/' + not_hold_out_file
                destination = training_no_bird_dir + "/" + not_hold_out_file
                shutil.copy(source, destination)

        elif not_hold_out_file in test_list:
            if df[df['itemid'] == not_hold_out_file.split('.')[0]]['hasbird'].values[0]:
                source = arrays_dir + '/' + not_hold_out_file
                destination = test_bird_dir + "/" + not_hold_out_file
                shutil.copy(source, destination)
            else:
                source = arrays_dir + '/' + not_hold_out_file
                destination = test_no_bird_dir + "/" + not_hold_out_file
                shutil.copy(source, destination)

    print("\tnot_hold_out dir populated!")





#----------------------------------------------
#   split_holdout_set
#----------------------------------------------
def split_holdout_set(hold_out_dir, all_files, arrays_dir):
    num_hold_out_samples = int(0.05 * CONST_DATA_SIZE)
    print_header("creating a hold_out set", num_hold_out_samples/CONST_DATA_SIZE)
    print("\tsetting up hold_out dir...")
    # create hold out dir if it doesn't exist already
    init_dirs(hold_out_dir)
    not_holdout_list, holdout_list = train_test_split(all_files, test_size=0.05, random_state=0)
    print("\tlength of not_holdout_list: {}".format(len(not_holdout_list)))
    print("\tlength of holdout_list: {}".format(len(holdout_list)))
    # populate holdout dir with holdout data
    print("\tpopulating hold_out dir...")
    holdout_list = set(holdout_list)
    for hold_out_file in os.listdir(arrays_dir):
        if hold_out_file in holdout_list:
            source = arrays_dir + '/' + hold_out_file
            destination = hold_out_dir + '/' + hold_out_file
            shutil.copy(source, destination)

    print("\thold_out dir populated!")
    return not_holdout_list





#----------------------------------------------
#   get_files
#----------------------------------------------
def get_files(specgram_dir):
    output = []
    for file in os.listdir(specgram_dir):
        output.append(file)

    return output




#----------------------------------------------
#   init_dirs
#----------------------------------------------
def init_dirs(dir_path):
    # check to see if dir exists
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)




#----------------------------------------------
#   print_header
#----------------------------------------------
def print_header(message1, message2):
    print("\n")
    print("------------------")
    print("{} ---> {}".format(message1, message2))
    print("------------------")  



#----------------------------------------------
#   main sentinel
#----------------------------------------------
if __name__ == "__main__": 
    main()