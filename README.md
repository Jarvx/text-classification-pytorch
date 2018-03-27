# Text classification

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

Dataset | `rand` | `static` | `nonstatic` | `multichannel`
--- | --- | --- | --- | ---
MR | 74.5 | 81.93  | 81.74 | 82.87
SST2| - |83.20   |84.56  |85.72 
TREC | 92.6 | 93.00 | 93.6 | 93.4?
CR |  |  |  | 
MPQA |  |  |  | 

For LSTM


Dataset | `rand` | `static` | `nonstatic` 
--- | --- | --- | --- 
MR |75.74  | 81.64 |82.69  | 
SST1 |43.48  |47.82  | 49.00 | 
SST2 | 80.50 |  82.92|85.77  | 
TREC | 87.8 | 92.80 | 93.8 
CR |  |  |  | 
MPQA |  |  |  | 




## TODO

- More datasets
- Bio-LSTM
- RL's application in text classification

