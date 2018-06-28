#----------------------------------------------
#   imports
#----------------------------------------------
import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

IMG_DIM = 128


#----------------------------------------------
#   main
#----------------------------------------------
def main():
    print_header("Creating CNN Architecture...")
    # Initialising the CNN
    classifier = Sequential()
    print("\tcreating convolutional layers...")
    print("\tadding max-pooling after every convolutional layer...")
    print("\tadding dropout to prevent overfitting...")

    # FIRST Convolution Layer + Max Pooling + Drop Out
    classifier.add(Conv2D(32, (3, 3), input_shape = (IMG_DIM, IMG_DIM, 1), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.20))

    # SECOND Convolution Layer + Max Pooling
    classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.25))


    # THIRD Convolution Layer + Max Pooling
    classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Dropout(0.25))

    # FOURTH Convolution Layer + Max Pooling
    classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (3, 3)))
    classifier.add(Dropout(0.20))
    
    # Flattening
    print("\tflattening feature maps...")
    classifier.add(Flatten())

    # FIRST Fully Connected Layer
    print("\tadding dense layers for ANN...")
    classifier.add(Dense(units = 256, activation = 'relu'))
    classifier.add(Dropout(0.25))

    # Output Layer With Sigmoid Activation
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    print('\tarchitecture complete!')
    
    # Compile settings
    print('\tcompiler settings complete!')
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Image Pre-Processing
    print_header("Image Pre-Processing...")
    print('\tsetting up train and test data generators...')
    train_datagen = ImageDataGenerator(rescale = 1./255)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_set = train_datagen.flow_from_directory(
                                                        'dataset/training_set',
                                                        target_size = (IMG_DIM, IMG_DIM),
                                                        batch_size = 32,
                                                        color_mode = 'grayscale',
                                                        class_mode = 'binary'
                                                    )

    test_set = test_datagen.flow_from_directory(
                                                    'dataset/test_set',
                                                    target_size = (IMG_DIM, IMG_DIM),
                                                    batch_size = 32,
                                                    color_mode = 'grayscale',
                                                    class_mode = 'binary'
                                                )

    print('\tpre-processing complete!')
    
    # Fit CNN Model
    print_header("Fit and Train CNN...")
    classifier.fit_generator(
                                generator = training_set,
                                validation_data = test_set,
                                epochs = 50,
                                workers = 12,
                                max_q_size = 100
                            )

    print('\tTraining Complete!')
    print_header("SUMMARY!")
    print(classifier.summary())
    classifier.save('my_model.h5')



#----------------------------------------------
#   print_header
#----------------------------------------------
def print_header(message1):
    print("\n")
    print("--------------------------------------------------")
    print("{}".format(message1))
    print("--------------------------------------------------") 


#----------------------------------------------
#   get_files
#----------------------------------------------
def get_files(directory):
    output = []
    for file in os.listdir(directory):
        output.append(file)

    return output


#----------------------------------------------
#   main sentinel
#----------------------------------------------
if __name__ == "__main__": 
    main()