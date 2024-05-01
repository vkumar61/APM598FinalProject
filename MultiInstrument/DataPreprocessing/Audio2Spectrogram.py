import os
import torch
import torchaudio
from torch.utils.data import TensorDataset
import numpy as np
import librosa
import pickle

def audio_to_spectrogram(audio_path):
    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(audio_path)
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=512)
    spec = spectrogram(SPEECH_WAVEFORM)
    return SPEECH_WAVEFORM, SAMPLE_RATE, spec

def label_to_vector(lines, path):
    instrument_words = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]
    vector = np.zeros(11)
    for line in lines:
        clean_line = line.replace("\t", "").replace("\n", "")
        index = instrument_words.index(clean_line)
        vector[index] = 1
    return vector

train_spectrograms = []
train_labels = []
validation_spectrograms = []
validation_labels = []
test_spectrograms = []
test_labels = []

instrument_labels = [0,1,2,3,4,5,6,7,8,9,10]

# Modify path
train_data_path = "train"

for file in os.listdir(train_data_path):
    if file.endswith(".wav"):
        audio_path = os.path.join(train_data_path, file)
        waveform, frequency, spectrogram = audio_to_spectrogram(audio_path)
        train_spectrograms.append(spectrogram)

        label_file = os.path.splitext(file)[0] + ".txt"
        label_path = os.path.join(train_data_path, label_file)
        with open(label_path, 'r') as f:
          lines = f.readlines()
        vector = label_to_vector(lines, train_data_path)
        train_labels.append(vector)

validation_data_path = "validation"

for file in os.listdir(validation_data_path):
    if file.endswith(".wav"):
        audio_path = os.path.join(validation_data_path, file)
        waveform, frequency, spectrogram = audio_to_spectrogram(audio_path)
        validation_spectrograms.append(spectrogram)

        label_file = os.path.splitext(file)[0] + ".txt"
        label_path = os.path.join(validation_data_path, label_file)
        with open(label_path, 'r') as f:
          lines = f.readlines()
        vector = label_to_vector(lines, validation_data_path)
        validation_labels.append(vector)

test_data_path = "test"

for file in os.listdir(test_data_path):
    if file.endswith(".wav"):
        audio_path = os.path.join(test_data_path, file)
        waveform, frequency, spectrogram = audio_to_spectrogram(audio_path)
        test_spectrograms.append(spectrogram)

        label_file = os.path.splitext(file)[0] + ".txt"
        label_path = os.path.join(test_data_path, label_file)
        with open(label_path, 'r') as f:
          lines = f.readlines()
        vector = label_to_vector(lines, test_data_path)
        test_labels.append(vector)

for i in range(len(train_spectrograms)):
    train_spectrograms[i] = librosa.power_to_db(train_spectrograms[i][0])
    train_spectrograms[i] = np.expand_dims(train_spectrograms[i], axis=0)
    train_spectrograms[i] = torch.from_numpy(train_spectrograms[i])

train_set_tensor = TensorDataset(torch.stack(train_spectrograms), torch.tensor(train_labels, dtype=torch.float32))

for i in range(len(validation_spectrograms)):
    validation_spectrograms[i] = librosa.power_to_db(validation_spectrograms[i][0])
    validation_spectrograms[i] = np.expand_dims(validation_spectrograms[i], axis=0)
    validation_spectrograms[i] = torch.from_numpy(validation_spectrograms[i])

validation_set_tensor = TensorDataset(torch.stack(validation_spectrograms), torch.tensor(validation_labels, dtype=torch.float32))

for i in range(len(test_spectrograms)):
    test_spectrograms[i] = librosa.power_to_db(test_spectrograms[i][0])
    test_spectrograms[i] = np.expand_dims(test_spectrograms[i], axis=0)
    test_spectrograms[i] = torch.from_numpy(test_spectrograms[i])

test_set_tensor = TensorDataset(torch.stack(test_spectrograms), torch.tensor(test_labels, dtype=torch.float32))


# Save the datasets to a file
with open('data/multiDatasets.pkl', 'wb') as f:
    pickle.dump((train_set_tensor, test_set_tensor, test_set_tensor), f)