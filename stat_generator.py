import os

stats ={}
imgs = os.listdir('processed-data/')
labels = ['cough', 'speech']

# Counting the number of samples for each label
for img in imgs:
    label = img.split('.png')[0].split('-')[-1]
    if label not in stats:
        stats[label] = 1
    else:
        stats[label] +=1
for key in stats.keys():
    print(labels[int(key)] +': '+str(stats[key]))
