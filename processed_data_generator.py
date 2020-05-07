import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import os
from praatio import tgio

# Obtain spectrogram
# Reference: https://stackoverflow.com/questions/47954034/plotting-spectrogram-in-audio-analysis
def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

labels = ['cough', 'speech']
speech_count = 0
annotations = os.listdir('flusense_data_label/')
for annotation in annotations:
    # Skip if not Label file
    if not annotation.endswith('.TextGrid'):
        continue

    # Loading in raw data
    raw_name = annotation.split('.')[0]+".wav"
    sample_rate, samples = wavfile.read('raw-data/'+raw_name)
    if len(samples.shape) > 1:
        samples = samples[:,0]
    else:
        samples = samples[:,]

    # Create spectrogram from audio file
    frequencies, times, spectrogram, new_spectrogram = 0, 0, 0, 0
    try:
        frequencies, times, spectrogram = log_specgram(samples, sample_rate)
        new_spectrogram = np.flipud(spectrogram.T)
    except:
        print("error happen with "+raw_name)
        continue

    # Capture the first 265 time points in the spectrogram
    window_size = 265
    full_path = os.path.join('flusense_data_label/', annotation)
    tg = tgio.openTextgrid(full_path)
    t_name = tg.tierNameList[0]
    entry_list = tg.tierDict[t_name].entryList
    ids = {}

    # Iterate through each entry and only capture entry with label "cough" or "speech"
    for entry in entry_list:

        # Set up file name for spectrogram
        if entry.label not in labels:
            continue
        if entry.label not in ids:
            ids[entry.label] = 0
        else:
            ids[entry.label] +=1

        # Get the chunk of the spectrogram that correspond to "cough" or "speech"
        range_of_interest = np.argwhere(np.logical_and(times>entry.start,times<entry.end)).reshape(-1)
        if len(range_of_interest) <window_size+25:
            continue
        
        # Clean out parts of the spectrogram that are empty, indicating there are no sound
        outlier_check = np.sum(new_spectrogram[:,range_of_interest[:window_size]],axis = 0)
        noise = np.argwhere(outlier_check<-10000)
        if len(noise) > len(outlier_check)//2:
            continue
            
        # Limit speech sample to only about 800 to have equal amounts of samples with cough
        if entry.label == 'speech' and speech_count >= 800:
            continue

        # Create spectrogram as 256 x 256 image
        fig = plt.figure(figsize=(256/144, 256/144), dpi=144)
        ax = plt.Axes(fig, [0., 0., 1., 1.], )
        ax.set_axis_off()
        fig.add_axes(ax)
        backing = new_spectrogram.shape[0] - window_size
        plt.imshow(new_spectrogram[:,range_of_interest[:window_size]])
        plt.savefig('processed-data/'+annotation.strip('.TextGrid')+'-'+str(ids[entry.label])+'-'+str(labels.index(entry.label)), dpi = 144)
        plt.clf()
        fig.clf()
        plt.close()

        # Keep check of the number of speech sample to have balanced label
        if entry.label == 'speech':
            speech_count+=1