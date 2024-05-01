# APM598FinalProject
This is all of our code for our APM598 Final project where we try to predict the instruments from audio files via CNNs.

It is organized into 2 directories: /SingleInstrument and /MultiInstrument. We consider these two cases separately as not only is the data organized differently, but we construct unique architectures and loss functions for each case. Further, each of these directories has a /DataPreprocessing subdirectory which contains all files related to processing the raw data, .wav files, into spectrograms, images, that the CNN architecture excels at analyzing. Also each of the main directories also contains 2 files, one with a linear model and another with the CNN.
