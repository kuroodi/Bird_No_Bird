# Bird Chirp Detector

This project is an audio based anomaly detector that uses deep learning to determine whether or not a bird chriped in a given audio file. The data consists of 20K raw audio files sourced from the BirdVox project(collaboration between Cornell and NYU). Each file is a simple 10 second audio clip recocrded by remote audio monitoring devices placed throughout a forest near Ithaca, NY. 

## Pipeline Overview

At face value, this is an audio recognition project. However, I chose to pose it as an image recognition problem by first converting each raw audio file into a 2-D image called a Spectrogram via a Fast Fourier Transform (FFT). Once converted, the image gets further processed before ultimetly getting passed through a CNN which classifies each audio clip as 'Bird' or 'No Bird'. 

While Spectrograms are 2-D image representations of sound, they actually contain 3 dimensions of information. The first 2 dimensions are the X and Y axies. The X-Axis represents time, the Y-Axis represents descrete frequencies found in the audio source file. The 3rd dimension is encoded as intensity of color and represents the amplitude/power of the signal at a given frequency. 

The audio to image conversion itself is done using the LibROSA python package which, in general, is really good for music and audio analysis.   

Spectrograms are by default generated as color images which means the have 3 channels (RGB). In the interest of keeping things light and fast, I chose to first convert each image to Gre-Scale (only 1 channel) and then to downsize each image. 

