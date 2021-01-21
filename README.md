# HamGNN
This is our experiment codes for the paper :

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

## Usage
* Execution sequence

  The execution sequence of codes is as follows: data_load.py--->data_triple.py--->model_train.py--->model_test.py
  
* Execution results

  During the execution of file model_train.py, the epoch, iteration, and training loss will be printed as the training process:
  
  ```
  [1,   600] loss: 1.21214
  [1,  1200] loss: 1.19586
  [1,  1800] loss: 1.18090
  [2,   600] loss: 1.13528
  [2,  1200] loss: 1.12297
  [2,  1800] loss: 1.11104
  [3,   600] loss: 1.07233
  [3,  1200] loss: 1.06290
  [3,  1800] loss: 1.05153
  ...
  ```
  
  File model_test.py should be executed after the training process, and the performance of HamGNN will be printed:
  
  ```
  HR@50: 0.2052; NDCG@50: 0.3081; P@50: 0.2020
  ```
