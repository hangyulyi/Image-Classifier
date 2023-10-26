# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

In order to use the application:
1. train.py is used to train a new network on a dataset and save the model as a checkpoint.
2. predict.py uses a trained network to predict the class for an input image

## How to use train.py

- Basic usage:
  >
  > python train.py data_directory
  >
  will print out training loss, validation loss, and validation accuray during network training
- Options:
    - To set directory for checkpoints:
        >
        > python train.py data_directory --save_dir save_directory
        >
    - To choose an architecture(will be defaulted to VGG16 if not indicated):
        >
        > python train.py data_directory --arch "vgg13"
        >
    - To set specific hyper parameters (learning rate, hidden units, epoch number):
        >
        > python train.py data_directory --learning_rate 0.01 --hidden_units 512 --epochs 20
        >
    - To set GPU usage for training:
        >
        > python train.py data_dir --gpu
        >
