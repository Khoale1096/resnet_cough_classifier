import argparse
import os
import pandas as pd
import csv
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

import numpy as np
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
MAX_EPOCH = 2
BATCH_SIZE = 32
STOPPING_PATIENCE = 32
LR_PATIENCE = 16
INITIAL_LR = 0.0001
CLASSES = [0, 1]
CLASS_NAMES = ['cough', 'speech']

def parse_args():
    parser = argparse.ArgumentParser(description='Train and test deep learning models on FluSense')
    parser.add_argument('--model', default='resnet')
    args = parser.parse_args()
    return args.model

def crop(img, size):
    (h, w, c) = img.shape
    x = int((w - size[0]) / 2)
    y = int((h - size[1]) / 2)
    return img[y:(y + size[1]), x:(x + size[0]), :]

def crop_generator(batches, size):
    while True:
        batch_x, batch_y = next(batches)
        (b, h, w, c) = batch_x.shape
        batch_crops = np.zeros((b, size[0], size[1], c))
        for i in range(b):
            batch_crops[i] = crop(batch_x[i], (size[0], size[1]))
        yield (batch_crops, batch_y)

def train(model_name):
    CLASSES_str = ['0', '1']
    OUTPUT_DIRECTORY = model_name+'/'

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    train_dataframe = pd.read_csv('labels/train.csv',dtype=str)
    val_dataframe = pd.read_csv('labels/val.csv',dtype=str)
    train_image_count = train_dataframe.shape[0]
    val_image_count = val_dataframe.shape[0]

    train_dataframe['Filenamefull'] = IMG_DIRECTORY+train_dataframe['Filename']
    val_dataframe['Filenamefull'] = IMG_DIRECTORY+val_dataframe['Filename'] 

    # Training image augmentation
    train_data_generator = ImageDataGenerator(
        rescale=1. / 255,
        fill_mode="constant",
        shear_range=0.2,
        zoom_range=(0.5, 1),
        horizontal_flip=True,
        rotation_range=360)

    # Validation image augmentation
    val_data_generator = ImageDataGenerator(
        rescale=1. / 255,
        fill_mode="constant",
        shear_range=0.2,
        zoom_range=(0.5, 1),
        horizontal_flip=True,
        rotation_range=360)

    # Load train images in batches from directory and apply augmentations
    train_data_generator = train_data_generator.flow_from_dataframe(
        train_dataframe,
        '',
        x_col='Filenamefull',
        y_col='Label',
        target_size=RAW_IMG_SIZE,
        batch_size=BATCH_SIZE,
        has_ext=True,
        classes=CLASSES_str,
        class_mode='categorical')

    # Load validation images in batches from directory and apply rescaling
    val_data_generator = val_data_generator.flow_from_dataframe(
        val_dataframe,
        '',
        x_col="Filenamefull",
        y_col="Label",
        target_size=RAW_IMG_SIZE,
        batch_size=BATCH_SIZE,
        has_ext=True,
        classes=CLASSES_str,
        class_mode='categorical')

    # Crop augmented images from 256x256 to 224x224
    train_data_generator = crop_generator(train_data_generator, IMG_SIZE)
    val_data_generator = crop_generator(val_data_generator, IMG_SIZE)

    # Load ImageNet pre-trained model
    if model_name == "resnet":
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    elif model_name == "inception":
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    elif model_name == "inception-resnet":
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    elif model_name == "xception":
        base_model = Xception(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    elif model_name == "densenet":
        base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    elif model_name == "resnet152":
        base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

    x = base_model.output
    # Add a global average pooling layer
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # Add fully connected output layer with sigmoid activation for multi label classification
    outputs = Dense(len(CLASSES), activation='sigmoid', name='fc9')(x)
    # Assemble the modified model
    model = Model(inputs=base_model.input, outputs=outputs)

    # Checkpoints for training
    model_checkpoint = ModelCheckpoint(OUTPUT_DIRECTORY + model_name+".hdf5", verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(patience=STOPPING_PATIENCE, restore_best_weights=True)
    tensorboard = TensorBoard(log_dir=OUTPUT_DIRECTORY, histogram_freq=0, write_graph=True, write_images=False)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=LR_PATIENCE, min_lr=0.000003125)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=INITIAL_LR), metrics=['categorical_accuracy'])
    csv_logger = CSVLogger(OUTPUT_DIRECTORY + "training_metrics.csv")

    # Train model until MAX_EPOCH, restarting after each early stop when learning has plateaued
    global_epoch = 0
    restarts = 0
    last_best_losses = []
    last_best_epochs = []
    while global_epoch < MAX_EPOCH:
        history = model.fit_generator(
            generator=train_data_generator,
            steps_per_epoch=train_image_count // BATCH_SIZE,
            epochs=MAX_EPOCH - global_epoch,
            validation_data=val_data_generator,
            validation_steps=val_image_count // BATCH_SIZE,
            callbacks=[tensorboard, model_checkpoint, early_stopping, reduce_lr, csv_logger],
            shuffle=False)
        last_best_losses.append(min(history.history['val_loss']))
        last_best_local_epoch = history.history['val_loss'].index(min(history.history['val_loss']))
        last_best_epochs.append(global_epoch + last_best_local_epoch)
        if early_stopping.stopped_epoch == 0:
            print("Completed training after {} epochs.".format(MAX_EPOCH))
            break
        else:
            global_epoch = global_epoch + early_stopping.stopped_epoch - STOPPING_PATIENCE + 1
            print("Early stopping triggered after local epoch {} (global epoch {}).".format(
                early_stopping.stopped_epoch, global_epoch))
            print("Restarting from last best val_loss at local epoch {} (global epoch {}).".format(
                early_stopping.stopped_epoch - STOPPING_PATIENCE, global_epoch - STOPPING_PATIENCE))
            restarts = restarts + 1
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=INITIAL_LR / 2 ** restarts),
                            metrics=['categorical_accuracy'])
            model_checkpoint = ModelCheckpoint(OUTPUT_DIRECTORY + "lastbest-{}.hdf5".format(restarts),
                                                monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Save last best model info
    with open(OUTPUT_DIRECTORY + "last_best_models.csv", 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Model file', 'Global epoch', 'Validation loss'])
        for i in range(restarts + 1):
            writer.writerow([model_name+"_lastbest-{}.hdf5".format(i), last_best_epochs[i], last_best_losses[i]])

# Run training on 1 of 5 models: densenet, resnet, inception-resnet, resnet152 and xception
# Example: python3 model_generator.py --model densenet
if __name__ == '__main__':
    # Parse command line arguments
    model = parse_args()
    train(model)