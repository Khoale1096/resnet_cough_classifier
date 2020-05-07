import os
import random

imgs = os.listdir('processed-data/')
stats ={}
cough = []
speech = []

# Append the filenames of each label to their own list
for img in imgs:
    label = img.split('.png')[0].split('-')[-1]
    if label == '0':
        cough.append(img)
    elif label == '1':
        speech.append(img)

# Randomize data points
random.shuffle(cough)
random.shuffle(speech)

# Create csv files for train dataset, validation dataset and test dataset
cutoff = min(len(cough), len(speech))
train_cutoff = cutoff*60//100
val_cutoff = train_cutoff + cutoff*20//100
with open('labels/train.csv', 'w') as w1:
    w1.write('Filename,Label\n')
    for i in range (0,train_cutoff):
        w1.write(cough[i]+','+'0'+'\n')
        w1.write(speech[i]+','+'1'+'\n')
    w1.close()

with open('labels/val.csv', 'w') as w1:
    w1.write('Filename,Label\n')
    for i in range (train_cutoff,val_cutoff):
        w1.write(cough[i]+','+'0'+'\n')
        w1.write(speech[i]+','+'1'+'\n')
    w1.close()

with open('labels/test.csv', 'w') as w1:
    w1.write('Filename,Label\n')
    for i in range (val_cutoff,cutoff):
        w1.write(cough[i]+','+'0'+'\n')
        w1.write(speech[i]+','+'1'+'\n')
    w1.close()
