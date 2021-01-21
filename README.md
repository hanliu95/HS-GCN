# HamGNN
This is our experiment codes for the paper:

Graph Neural Networks in Hamming Space for Efficient Recommendation.

## Environment settings
* Python 3.7
* Pytorch 1.4.0
* PyTorch Geometric 1.6.1
* Numpy 1.19.5
* Pandas 1.1.3

## File specification
* data_load.py : loads the raw data in path `./raw_data`, and the results are saved in path `./para`.
* data_triple.py : obtains the triplets for model training, and the results are saved in path `./para`.
* HamGNN_model.py : implements the model framework of HamGNN.
* model_train.py : the training process of model.
* model_test.py : the testing process of model.
