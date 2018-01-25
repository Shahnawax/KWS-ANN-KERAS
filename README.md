# KWS-ANN-KERAS
This repository contains the pretrained artificial neural network ([ANN](https://en.wikipedia.org/wiki/Artificial_neural_network)) model (.h5) files developed and trained in Keras, for Key Word Spotting(KWS). All these models are the equellents of the [Tensorflow](https://www.tensorflow.org/) based varients present in [ML-KWS-for-MCU](https://github.com/ARM-software/ML-KWS-for-MCU) and proposed in the paper: [Hello Edge: Keyword spotting on Microcontrollers](https://arxiv.org/pdf/1711.07128.pdf). For details have a look on the paper or  `ML-KWS-for-MCU` repository.

## Pipeline of the KWS systems 
The pipeline of a Key word spotting system is given in the figure below.

<img src="https://raw.githubusercontent.com/Shahnawax/KWS-ANN-KERAS/master/kws_pipeline.png">

## Dataset used

For experiements we used the [Speech Commands dataset](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html) created by [TensorFlow](https://www.tensorflow.org/) and [AIY labs](https://aiyprojects.withgoogle.com/).  The dataset has `65,000` one-second long utterances saved as `.wav` files of `30` single word commands, by pronounced by thousands of different people. The sampling freuqncy of the sound files is `16000 samples/sec`.
## Preprocessing systems

The raw wave files are processed to generate the feature matrix containing the Mel-frequency cepstral coefficients (mfcc) for overlapping windowed signal. For the current results we used a window size of `40 msec` with a stride size of `20 msec`. Influenced from the networks in [ML-KWS-for-MCU](https://github.com/ARM-software/ML-KWS-for-MCU), we used the `10 mfccs` only.

## Training of the models

The training commands with all the hyperparameters to reproduce the models shown in the 
[paper](https://arxiv.org/pdf/1711.07128.pdf) are given [here](https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/train_commands.txt). For these experiment we trained the models on `12` classese, e.g. with all numbers from `zero` to `nine` and two classes named as `silence` and `unknown`. The `unknown` class contains instances from all other classes. The ratio for `unknown` and `silence` is kept as `10%`. For details see the `input_data.py` file in this [repository](https://github.com/ARM-software/ML-KWS-for-MCU).

## Pretrained models

Pretrained models (.h5 files) for different neural network architectures such as Deep Neural Networks (DNN),
Convolutional Neural Networks (CNN), Basic Long Short-Term Memory (LSTM), Gated Recurrent Unis (GRU) networks, Convolutional and Recurrent Neural Networks (CRNN) and Depth-wise Seperable Convolutional Neural Networks (DS-CNN) shown in [arXiv paper](https://arxiv.org/pdf/1711.07128.pdf) are added in [Pretrained_models](Pretrained_models). Accuracy of the models on testing set and their memory requirements per inference are also summarized in the following table. 

<img src="https://raw.githubusercontent.com/Shahnawax/KWS-ANN-KERAS/master/accuracy_table.png">

PS. Please note that all the weights and activations are considered to be 32 bits long.

