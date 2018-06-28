#----------------------------------------------
#   USAGE: 
#       This script will read in all audio 
#       files and pass them through the data 
#       pipeline and can also move newly 
#       generated files up to S3 
#       
#
#   INPUTS:
#       1 --> local path to audio files
#       2 --> user input prompts for S3 info
# 
#   OUTPUTS:
#       Will store spectorgarms of all audio
#       files as PNG files and will store numpy
#       arrays of feature extractions as .npy
#       files on S3 bucket specified
#
#   /Users/vineetkuroodi/Desktop/Bird_No_Bird/BirdVox/wav/
#----------------------------------------------




#----------------------------------------------
#   imports
#----------------------------------------------
import sys
import os
import time
import pylab
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import splitext
from PIL import Image


#----------------------------------------------
#   main
#----------------------------------------------
def main():
    # get bucket name from argument list
    transfer_to_s3 = input("Do you want to transfer files to S3 (y/n): ")
    if transfer_to_s3 == 'y':
        s3_bucket_name = input("Enter S3 bucket path info: ")
    # print quick message
    print_header("Pipeline Initiated", "Transform Audio")
    # get local dir from argument list and convert to string
    local_dir = os.fsencode(sys.argv[1]).decode('UTF-8')
    # initialize output dirs (spectrograms/, arrays/)
    init_temp_dirs(local_dir)
    # push each audio file through pipe and fill output dirs
    audio_pipe(local_dir)
    # move spectorgrams and arrays to S3 bucket
    if transfer_to_s3 == 'y':
        print_header("S3 Transfer Initiated", s3_bucket_name)
        copy_to_s3(s3_bucket_name, local_dir)



#----------------------------------------------
#   copy_to_s3
#----------------------------------------------
def copy_to_s3(s3_bucket_name, local_dir):
    # build up aws command (use recursive)
    aws_cli_command = "time aws s3 cp --recursive --quiet "
    # build up local paths
    local_path_spec = local_dir + '../spectrograms/'
    local_path_arry = local_dir + '../arrays/'
    local_path_greg = local_dir + '../spectrograms_grey_scale/'
    # build path to s3 bucket
    aws_s3_bucket_spec = " s3://" + s3_bucket_name + "/spectrograms/" 
    aws_s3_bucket_arry = " s3://" + s3_bucket_name + "/arrays/"
    aws_s3_bucket_grey = " s3://" + s3_bucket_name + "/spectrograms_grey_scale/"

    # kick off transfer command and time it
    os.system(aws_cli_command + local_path_spec + aws_s3_bucket_spec)
    os.system(aws_cli_command + local_path_arry + aws_s3_bucket_arry)
    os.system(aws_cli_command + local_path_greg + aws_s3_bucket_grey)



#----------------------------------------------
#   audio_pipe
#----------------------------------------------
def audio_pipe(local_dir):
    i = 0
    start_time = time.time()
    # iterate through each wav file
    for file_name in os.listdir(local_dir):
        i += 1
        # generate file path 
        path_to_wav = local_dir + str(file_name)
        # generate spectogram (with labels)
        S_power_to_db = gen_save_specgram(path_to_wav, local_dir, file_name, grey_scale = 0)
        # generate spectogram (grey-scale, no labels)
        gen_save_specgram(path_to_wav, local_dir, file_name, grey_scale = 1)
        # save numpy array
        np.save(local_dir + '../arrays/' + splitext(str(file_name))[0], S_power_to_db) 

        if (i % 2000) == 0:
            print("\t {} files complete...".format(i))

    
    print("\n\tElapsed time: {}".format(time.time()-start_time))



#----------------------------------------------
#   gen_save_specgram
#----------------------------------------------
def gen_save_specgram(path_to_wav, local_dir, file_name, grey_scale = 0):
    # load wav file through libROSA
    y, sr = librosa.load(path_to_wav)
    # generate melspectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    # log transform
    S_power_to_db = librosa.power_to_db(S, ref=np.max)

    # make picture
    if grey_scale == 0:
        plt.figure(figsize=(12,6))
        librosa.display.specshow(S_power_to_db, x_axis='time', y_axis='mel')
        plt.title('mel spectrogram (power to db)')
        plt.colorbar(format='%+02.0f dB')
        plt.savefig(local_dir + '../spectrograms/' + splitext(str(file_name))[0]+'.png')
        plt.close()
    else:
        plt.figure(figsize=(2.56,2.56))
        librosa.display.specshow(S_power_to_db, cmap='gray_r')
        file_name = local_dir + '../spectrograms_grey_scale/' + splitext(str(file_name))[0]+ '.png'
        plt.savefig(file_name)
        plt.close()
        img = Image.open(file_name).convert('LA')
        img.save(file_name)
    
    return S_power_to_db




#----------------------------------------------
#   init_temp_dirs
#----------------------------------------------
def init_temp_dirs(local_dir):
    # check to see if spectorgram dir exists
    if not os.path.exists(local_dir + '../spectrograms/'):
        os.makedirs(local_dir + '../spectrograms/')
    # check to see if arrays dir exists
    if not os.path.exists(local_dir + '../arrays/'):
        os.makedirs(local_dir + '../arrays/')
    # check to see if grey_scale spectogram dir exists
    if not os.path.exists(local_dir + '../spectrograms_grey_scale/'):
        os.makedirs(local_dir + '../spectrograms_grey_scale/')



#----------------------------------------------
#   print_header
#----------------------------------------------
def print_header(message1, message2):
    print("\n")
    print("------------------")
    print("{} ---> {}".format(message1, message2))
    print("------------------")  
    print("\n")



#----------------------------------------------
#   main sentinel
#----------------------------------------------
if __name__ == "__main__":   
    main()