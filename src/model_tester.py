#----------------------------------------------
#   imports
#----------------------------------------------
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


from keras.preprocessing import image
from keras.models import load_model


IMG_DIM = (128, 128)
path_to_png_files = '/Users/vineetkuroodi/Desktop/Bird_No_Bird/BirdVox/hold_out_data/'



#----------------------------------------------
#   main
#----------------------------------------------
def main():
    # load up trained model
    classifier = load_model('my_model.h5')
    print(classifier.summary())
    # load in labels
    df = pd.read_csv('../data/BirdVox_Labels.csv')
    
    #declare output lists
    y_pred_prob = []
    y_predicted = []
    y_actual = []

    os.chdir(path_to_png_files)
    i = 0

    # iterate through all hold_out PNG files
    for not_hold_out_file in os.listdir("."):
        i += 1
        # pipeline to process images
        current_image = image.load_img(not_hold_out_file, target_size = IMG_DIM, grayscale = True)
        current_image = image.img_to_array(current_image)
        current_image = np.expand_dims(current_image, axis = 0)
        # make raw prediction
        prediction = classifier.predict(current_image)
        # append and save probability 
        y_pred_prob.append(prediction[0][0])
        # append the actual value taken from CSV
        bird_no_bird = df[df['itemid'] == not_hold_out_file.split('.')[0]]['hasbird'].values[0]
        y_actual.append(bird_no_bird)   


    # convert to numpy 
    y_pred_prob = np.array(y_pred_prob)
    y_actual = np.array(y_actual)

    # create numpy array of predictions
    mask = np.array([y_pred_prob > 0.5])
    y_predicted = np.invert(mask).astype(int).flatten()

    report = classification_report(y_actual, y_predicted)
    print(report)
    accuracy = accuracy_score(y_actual, y_predicted)
    print(accuracy)
    plot_roc(y_actual, y_pred_prob, "Bird No Bird")





#----------------------------------------------
#   plot_roc
#----------------------------------------------
def plot_roc(y_actual, y_pred_prob, plot_title):
    AUC = roc_auc_score(y_actual, 1 - y_pred_prob)
    fpr, tpr, thresholds = roc_curve(y_actual, 1 - y_pred_prob)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, marker='.', color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic {}'.format(plot_title))
    plt.legend(loc="lower right")
    plt.show()





#----------------------------------------------
#   main sentinel
#----------------------------------------------
if __name__ == "__main__": 
    main()