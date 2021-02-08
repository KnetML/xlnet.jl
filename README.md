# xlnet.jl  

This repository contains the [Knet](https://github.com/denizyuret/Knet.jl) implementation of XLNet Pretraining model. (Z. Yang Et al.) Original implementation can be found [here](https://github.com/zihangdai/xlnet).

## Usage

Two main models are exported from ```XLNetModel``` and ```XLNetClassifier```. ```XLNetClassifier``` model can be constructed for arbitary number of classes, and also easy to use interface is provided for saving and loading model weights. 
Only classification downstream task is implemented for know but ```XLNetModel``` can be used for text modeling, but you need to specify special tokens carefully.

Detailed usage is demonstrated in the [notebook](xlnet_sentiment_classification.ipynb) for IMDB sentiment classification dataset.

## Todo

Models for other downstream tasks (Question-Answering etc.) will be be implemented.

