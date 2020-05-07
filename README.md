# resnet_cough_classifier

## System Requirement

* Need Python versions >= 3.7.6
* Install TensorFlow: ```pip3 install tensorflow```
* Install Keras: ```pip3 install keras```
* Install scikit-learn: ```pip3 install scikit-learn```
* Install pandas: ```pip3 install pandas```

## Steps to run the inference

* Download the ResNet model into the [resnet](./resnet/) directory and run ```python3 prediction_generator.py --model resnet```. The script will run this network through the test data and produce two outputs: a classification reports and a confusion matrix, both stored in [resnet](./resnet/)

## Step to run the end-to-end pipeline

* Download the raw audio signals into [raw-data](./raw-data/) directory (since the sizes of the raw data is quite large, so they are not included in this repository). For access to the raw data, please contact the [FluSense](https://github.com/Forsad/FluSense-data) authors.
* To create processed spectrogram images from these raw audio signals, run ```python3 processed_data_generator.py```
* To create the train-val-test datasets, run ```python3 label_generator.py```
* (Optional) to look at the class distribution , run ```python3 stat_generator.py```
* To train the neural network classifier, run ```python3 model_generator.py --model resnet```. IMPORTANT NOTE: the code also provide the ability to train other well-known deep learning architectures such as DenseNet, Xception and InceptionV3, and can be obtain by passing ```densenet```, ```xception```, ```inception``` to the argument ```--model```.
* To test the classifier, run ```python3 prediction_generator.py --model resnet```. Two CSVs are output: a classification reports and a confusion matrix in the directory named after the model used.