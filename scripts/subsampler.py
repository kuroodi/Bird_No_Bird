#----------------------------------------------
#   imports
#----------------------------------------------
import sys
import os
import numpy as np
import pandas as pd
import shutil
import random
from sklearn.model_selection import train_test_split


CONST_DATA_SIZE = 20000




#----------------------------------------------
#   main
#----------------------------------------------
def main():
    # get info from users
    print_header("Gathering info...", "")
    use_defaults = input("\tUse defaults? (y/n): ")

    if use_defaults == 'n':
        user_answers = get_user_info()
        source_dir =  user_answers[0]
        target_dir = user_answers[1]
    else:
        source_dir =  "/Users/vineetkuroodi/Desktop/Bird_No_Bird_Test/dataset/"
        target_dir =  "/Users/vineetkuroodi/Desktop/Bird_No_Bird_Test/mini_dataset/"

    if os.path.isdir(target_dir):
        print("\n\tLooks like the source_dir exists...removing it now...\n")
        shutil.rmtree(target_dir)

    total_files = int(input("\tHow many files do you want to sample? : "))
    num_per_class = int(total_files/2)


    print("\tcopying from: {}".format(source_dir))
    print("\tcopying to: {}".format(target_dir))

    # copy over current 'dataset' dir
    print_header("Inititaing copy...", "")
    shutil.copytree(source_dir, target_dir)
    print("\tfull copy complete!")
    
    # reduce data
    print_header("Reducing data...", "train size: {}".format(num_per_class))
    reduce_data(directory = target_dir + "training_set/", num_per_class = num_per_class)
    print("\ttraining data reduction complete!")
    


#----------------------------------------------
#   reduce_data
#----------------------------------------------
def reduce_data(directory, num_per_class):
    # check if temporary directories exist and clean them up
    if not os.path.exists(directory + "/bird_temp"):
        os.makedirs(directory + "/bird_temp")
    else:
        shutil.rmtree(directory + "/bird_temp")
        os.makedirs(directory + "/bird_temp")

    if not os.path.exists(directory + "/no_bird_temp"):
        os.makedirs(directory + "/no_bird_temp")
    else:
        shutil.rmtree(directory + "/no_bird_temp")
        os.makedirs(directory + "/no_bird_temp")
    
    # downsample bird data
    random_bird_chooser = set(random.sample(range(0, 7500), num_per_class))
    counter = 0
    for index, file in enumerate(os.listdir(directory + "/bird")):
        if index in random_bird_chooser: 
            shutil.copy(directory + "/bird/" + file, directory + "/bird_temp/" + file)
            counter += 1
        
        if counter == num_per_class:
            break


    # downsample no_bird data
    random_no_bird_chooser = set(random.sample(range(0, 7500), num_per_class))
    counter = 0
    for index, file in enumerate(os.listdir(directory + "/no_bird")):
        if index in random_no_bird_chooser: 
            shutil.copy(directory + "/no_bird/" + file, directory + "/no_bird_temp/" + file)
            counter += 1
        
        if counter == num_per_class:
            break



    # remove older bird and no_bird directories 
    shutil.rmtree(directory + "bird")
    shutil.rmtree(directory + "no_bird")

    # rename temp bird/no_bird directories
    os.rename(directory + "/bird_temp/", directory + "/bird/")
    os.rename(directory + "/no_bird_temp/", directory + "/no_bird/")





#----------------------------------------------
#   get_user_info
#----------------------------------------------
def get_user_info():
    source_dir = input("\tPlease enter source directory path: ")
    target_dir = input("\tPlease enter target directory path: ")

    return source_dir, target_dir





#----------------------------------------------
#   get_files
#----------------------------------------------
def get_files(directory):
    output = []
    for file in os.listdir(directory):
        output.append(file)

    return output



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