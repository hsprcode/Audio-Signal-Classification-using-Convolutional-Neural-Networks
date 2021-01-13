## Music-Genre-Classification-using-Convolutional-Neural-Networks

# Data Description
GTZAN Dataset (Tzanetakis & Cook, 2002) is composed of 1000 half a minute audio excerpts. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format. It the most used dataset for Music Genre Recognition by machine learning algorithms. The files were collected in 2000-2001 from a variety of sources including personal CDs, radio, microphone recordings, in order to represent a variety of recording conditions. 
It is available to download from: http://marsyas.info/downloads/datasets.html

## MODE 1: Feature Extraction & Artificial Neural Network
Feature extraction is the way of finding numerical representation that can be used to describe an audio signal. The basic intention of this process is to recognize patterns from huge amount of data. Once the features are pulled out, Neural Networks can be used directly using the numerical data.
LibROSA is a python package for music and audio analysis. It offers many functions to extract features directly from audio data. 
The extracted features are labelled and arranged in the csv file.

## MODE 2: 1-D Amplitude values of Audio & Convolutional Neural Network
Another way to input data from audio signal is to get the amplitude values directly for each time step and feed it into a one-dimensional CNN. The main hindrance is that most audio files (including the GTZAN dataset) have high sample rate. There are two ways we can get around this problem. One is to reduce the sample rate, and the other is to cut the length of each audio sample in the dataset. Reducing sampling rate leads to loss of many important features of music. Therefore, the audio samples are cut into 2 second pieces with 1 second overlap. Doing this also increases the size of the data set.  Cut audio length and the overlap percentage plays a major role in the quality of the classification process; Features expressed over time longer than the cut audio length, will be lost if overlap is not included.
Many research papers were studied to get a good 1-D CNN architecture and a paper on environmental sound classification (Abdoli, Cardinal, & Koerich, 2019), which discusses about different architectures considering several input sizes. They proposed CNN architectures which only needs a small number of parameters comparatively, and thus the volume of data essential for training is also reduced. 

Abdoli, S., Cardinal, P., & Koerich, A. (2019). End-to-End Environmental Sound Classification using a 1DConvolutional Neural Network. Department of Software and IT Engineering, ́Ecole de Technologie Sup ́erieure, Universit ́e du Qu ́ebec. Link: https://arxiv.org/pdf/1904.08990.pdf
