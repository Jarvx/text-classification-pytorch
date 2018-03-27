# kim_cnn

Implementation for Some pupular machine learning algorithms for text classification.
This project is partly based castorini's work on https://github.com/castorini/Castor. 

## Model Type

- rand: All words are randomly initialized and then modified during training.
- static: A model with pre-trained vectors from [word2vec](https://code.google.com/archive/p/word2vec/). All words -- including the unknown ones that are initialized with zero -- are kept static and only the other parameters of the model are learned.
- non-static: Same as above but the pretrained vectors are fine-tuned for each task.
- multichannel: A model with two sets of word vectors. Each set of vectors is treated as a 'channel' and each filter is applied to both channels, but gradients are back-propagated only through one of the channels. Hence the model is able to fine-tune one set of vectors while keeping the other static. Both channels are initialized with word2vec.# text-classification-cnn
Implementation for Convolutional Neural Networks for Sentence Classification of [Kim (2014)](https://arxiv.org/abs/1408.5882) with PyTorch.

## Requirement

PyTorch and torch text

## Quick Start



## Results

For CNN.  

| Dev Accuracy on SST-1 |     rand      |    static    |   non-static  |  multichannel | 
|:--------------------------:|:-----------:|:-----------:|:-------------:|:---------------:| 
| My-Implementation      | 42.597639| 48.773842| 48.864668   | 49.046322  |  

| Test Accuracy on SST-1|      rand      |    static    |    non-static |  multichannel | 
|:--------------------------:|:-----------:|:-----------:|:-------------:|:---------------:| 
| Kim-Implementation    | 45.0            | 45.5        | 48.0             | 47.4                 | 
| My- Implementation    | 39.683258  | 45.972851| 48.914027|  47.330317       |
For LSTM

## TODO

- More datasets
- Bio-LSTM
- RL's application in text classification

