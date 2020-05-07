import argparse
import os
import pandas as pd
import csv
from PIL import Image

from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

import numpy as np
from numpy import argmax

# # Deep learning code inspired by the Project DeepWeeds
# # Source: https://github.com/AlexOlsen/DeepWeeds/blob/master/deepweeds.py

# # Data obtained from FluSense project
# # Source: https://github.com/Forsad/FluSense-data

# # Global info
LABEL_DIRECTORY = "./labels/"
IMG_DIRECTORY = "./processed-data/"
# Global variables
RAW_IMG_SIZE = (256, 256)
IMG_SIZE = (224, 224)
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
BATCH_SIZE = 32
CLASSES = [0, 1]
CLASS_NAMES = ['cough', 'speech']


def parse_args():
    parser = argparse.ArgumentParser(description='Train and test deep learning models on FluSense data')
    parser.add_argument('--model', default='resnet')
    args = parser.parse_args()
    return args.model


def predict(model_name):
    CLASSES_str = ['0', '1']
    OUTPUT_DIRECTORY = model_name+'/'

    test_dataframe = pd.read_csv('labels/test.csv',dtype=str)
    test_image_count = test_dataframe.shape[0]
    test_dataframe['Filenamefull'] = IMG_DIRECTORY+test_dataframe['Filename']

    # Image augmentation
    test_data_generator = ImageDataGenerator(rescale=1. / 255)

    # Load test images in batches from directory and apply rescaling
    test_data_generator = test_data_generator.flow_from_dataframe(
        test_dataframe,
        '',
        x_col="Filenamefull",
        y_col="Label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        has_ext=True,
        shuffle=False,
        classes=CLASSES_str,
        class_mode='categorical')

    # Load the model
    model = load_model(OUTPUT_DIRECTORY + model_name+'.hdf5')

    # Evaluate model on test data
    predictions = model.predict_generator(test_data_generator, test_image_count // BATCH_SIZE + 1)
    y_true = test_data_generator.classes
    y_pred = np.argmax(predictions, axis=1)
    
    # Generate and print classification metrics and confusion matrix
    print(classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASS_NAMES))
    report = classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASS_NAMES, output_dict=True)
    with open(OUTPUT_DIRECTORY +model_name+ '_classification_report.csv', 'w') as f:
        for key in report.keys():
            f.write("%s,%s\n" % (key, report[key]))
    conf_arr = confusion_matrix(y_true, y_pred, labels=CLASSES)
    print(conf_arr)
    np.savetxt(OUTPUT_DIRECTORY +model_name+ "_confusion_matrix.csv", conf_arr, delimiter=",")


# Run prediction on 1 of 5 models: densenet, resnet, inception-resnet, resnet152 and xception
# Example: python3 prediction_generator.py --model densenet
if __name__ == '__main__':
    # Parse command line arguments
    model = parse_args()
    predict(model)