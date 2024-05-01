import torchaudio
import os
import torch
import numpy as np
import librosa
from torch.utils.data import TensorDataset
import pickle


def audio_to_spectrogram(audio_path):
    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(audio_path)
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=512)
    spec = spectrogram(SPEECH_WAVEFORM)
    return SPEECH_WAVEFORM, SAMPLE_RATE, spec

train_spectrograms = []
train_labels = []
validation_spectrograms = []
validation_labels = []
test_spectrograms = []
test_labels = []


instrument_directories = [
    "cel", "cla", "flu", "gac", "gel",
    "org", "pia", "sax", "tru", "vio", "voi"
]

instrument_labels = [0,1,2,3,4,5,6,7,8,9,10]

# Modify path
train_data_path = "train"

for instrument_label in instrument_directories:
    instrument_dir = os.path.join(train_data_path, instrument_label)
    for file in os.listdir(instrument_dir):
        if file.endswith(".wav"):
            audio_path = os.path.join(instrument_dir, file)
            waveform, frequency, spectrogram = audio_to_spectrogram(audio_path)
            train_spectrograms.append(spectrogram)
            instrument_index =  instrument_directories.index(instrument_label)
            train_labels.append(instrument_labels[instrument_index])

validation_data_path = "validation"

for instrument_label in instrument_directories:
    instrument_dir = os.path.join(validation_data_path, instrument_label)
    for file in os.listdir(instrument_dir):
        if file.endswith(".wav"):
            audio_path = os.path.join(instrument_dir, file)
            waveform, frequency, spectrogram = audio_to_spectrogram(audio_path)
            validation_spectrograms.append(spectrogram)
            instrument_index =  instrument_directories.index(instrument_label)
            validation_labels.append(instrument_labels[instrument_index])

test_data_path = "test"

for instrument_label in instrument_directories:
    instrument_dir = os.path.join(test_data_path, instrument_label)
    for file in os.listdir(instrument_dir):
        if file.endswith(".wav"):
            audio_path = os.path.join(instrument_dir, file)
            waveform, frequency, spectrogram = audio_to_spectrogram(audio_path)
            test_spectrograms.append(spectrogram)
            instrument_index =  instrument_directories.index(instrument_label)
            test_labels.append(instrument_labels[instrument_index])

for i in range(len(train_spectrograms)):
    train_spectrograms[i] = librosa.power_to_db(train_spectrograms[i][0])
    train_spectrograms[i] = np.expand_dims(train_spectrograms[i], axis=0)
    train_spectrograms[i] = torch.from_numpy(train_spectrograms[i])

train_set_tensor = TensorDataset(torch.stack(train_spectrograms), torch.tensor(train_labels, dtype=torch.int64))

for i in range(len(validation_spectrograms)):
    validation_spectrograms[i] = librosa.power_to_db(validation_spectrograms[i][0])
    validation_spectrograms[i] = np.expand_dims(validation_spectrograms[i], axis=0)
    validation_spectrograms[i] = torch.from_numpy(validation_spectrograms[i])

validation_set_tensor = TensorDataset(torch.stack(validation_spectrograms), torch.tensor(validation_labels, dtype=torch.int64))

for i in range(len(test_spectrograms)):
    test_spectrograms[i] = librosa.power_to_db(test_spectrograms[i][0])
    test_spectrograms[i] = np.expand_dims(test_spectrograms[i], axis=0)
    test_spectrograms[i] = torch.from_numpy(test_spectrograms[i])

test_set_tensor = TensorDataset(torch.stack(validation_spectrograms), torch.tensor(validation_labels, dtype=torch.int64))

# Save the datasets to a file
with open('datasets.pkl', 'wb') as f:
    pickle.dump((train_set_tensor, validation_set_tensor, test_set_tensor), f)