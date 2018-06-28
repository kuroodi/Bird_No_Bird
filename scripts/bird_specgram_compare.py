#----------------------------------------------
#   USAGE: 
#       This script will convert
#       10 bird and no_bird audio files into
#       spectorgrams and saves them as image 
#       files for visual comparison
#       
#
#   INPUTS:
#       1 --> relative path to wav dir
#          
# 
#   OUTPUTS:
#       image files in ./spectorgrams/
#----------------------------------------------


#----------------------------------------------
#   imports
#----------------------------------------------
import sys
import os
import matplotlib.pyplot as plt
import pylab
import librosa
import librosa.display
import numpy as np
import pandas as pd



#----------------------------------------------
#   main
#----------------------------------------------
def main():
    #get path_to_wav_dir from argument list and convert to string
    path_to_wav_dir = os.fsencode(sys.argv[1]).decode('UTF-8')
    #get list of bird chirp file names and no chirp file names
    bird, no_bird = read_in_data()
    #print names of bird/no_bird examples 
    print("**BIRDS--->")
    print("\n".join(bird))
    print('\n\n\n**NO_BIRDS--->')
    print("\n".join(no_bird))

    img_or_array = input('\n\nsave image or save array (img/array)? ')
    if img_or_array == 'img':
        #check to see if spectorgram dir exists
        if not os.path.exists('./spectograms/'):
            os.makedirs('./spectograms/')
        #convert 2 audio files to spectograms and save output images
        conversion_stype = input('\nInput your conversion style (standard/log): ')
        if conversion_stype == 'standard':
            convert_wav_to_specgram_standard(path_to_wav_dir, bird, no_bird)
        else:
            convert_wav_to_specgram_log(path_to_wav_dir, bird, no_bird)
    else:
        exit()



#----------------------------------------------
#   convert_wav_to_specgram_standard
#----------------------------------------------
def convert_wav_to_specgram_standard(path_to_wav_dir, bird, no_bird):
    for file in bird:
        path_to_wav = path_to_wav_dir + str(file) + '.wav'
        y, sr = librosa.load(path_to_wav)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        # Make a new figure
        plt.figure(figsize=(12,6))
        S_power_to_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_power_to_db, x_axis='time', y_axis='mel')
        # Put a descriptive title on the plot
        plt.title('mel spectrogram (power to db)')
        # draw a color bar
        plt.colorbar(format='%+02.0f dB')
        plt.savefig('./spectograms/' + 'bird_' + str(file) + '.png')

    for file in no_bird:
        path_to_wav = path_to_wav_dir + str(file) + '.wav'
        y, sr = librosa.load(path_to_wav)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_power_to_db = librosa.power_to_db(S, ref=np.max)
        # Make a new figure
        plt.figure(figsize=(12,6))
        librosa.display.specshow(S_power_to_db, x_axis='time', y_axis='mel')
        # Put a descriptive title on the plot
        plt.title('mel spectrogram (power to db)')
        # draw a color bar
        plt.colorbar(format='%+02.0f dB')
        plt.savefig('./spectograms/' + 'nobird_' + str(file) + '.png')



#----------------------------------------------
#   convert_wav_to_specgram_log
#----------------------------------------------
def convert_wav_to_specgram_log(path_to_wav_dir, bird, no_bird):
    for file in bird:
        path_to_wav = path_to_wav_dir + str(file) + '.wav'
        # Load sound file
        y, sr = librosa.load(path_to_wav)
        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S = librosa.amplitude_to_db(S, ref=np.max)
        # Make a new figure
        plt.figure(figsize=(12,8))
        # Display the spectrogram on a mel scale
        # sample rate and hop length parameters are used to render the time axis
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
        # Put a descriptive title on the plot
        plt.title('mel spectrogram (amp to db)')
        # draw a color bar
        plt.colorbar(format='%+02.0f dB')
        plt.savefig('./spectograms/' + 'bird_' + str(file) + '.png')


    for file in no_bird:
        path_to_wav = path_to_wav_dir + str(file) + '.wav'
        # Load sound file
        y, sr = librosa.load(path_to_wav)
        # Let's make and display a mel-scaled power (energy-squared) spectrogram
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S = librosa.amplitude_to_db(S, ref=np.max)
        # Make a new figure
        plt.figure(figsize=(12,8))
        # Display the spectrogram on a mel scale
        # sample rate and hop length parameters are used to render the time axis
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
        # Put a descriptive title on the plot
        plt.title('mel spectrogram (amp to db)')
        # draw a color bar
        plt.colorbar(format='%+02.0f dB')
        plt.savefig('./spectograms/' + 'nobird_' + str(file) + '.png')





#----------------------------------------------
#   read_in_data
#----------------------------------------------
def read_in_data():
    choice = input('\nWhich set of labels do you want to load (BirdVox/FreeField)? ')
    if choice == 'BirdVox':
        df = pd.read_csv('../data/BirdVox_Labels.csv')
    else:
        df = pd.read_csv('../data/FreeField_Labels.csv')

    no_bird = list(df[df['hasbird'] == 0].head(10)['itemid'])
    bird = list(df[df['hasbird'] == 1].head(10)['itemid'])
    return bird, no_bird



#----------------------------------------------
#   main sentinel
#----------------------------------------------
if __name__ == "__main__": 
    main()