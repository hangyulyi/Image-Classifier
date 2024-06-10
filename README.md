# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

In order to use the application:

NOTE: Data set of flower files is not in repo (it's like 10,000 pictures of flowers)
1. train.py is used to train a new network on a dataset and save the model as a checkpoint.
2. predict.py uses a trained network to predict the class for an input image

## How to use train.py

- Basic usage:
  `python train.py data_directory`
  
  will print out training loss, validation loss, and validation accuray during network training
- Options:
    - To set directory for checkpoints:
        `python train.py data_directory --save_dir save_directory`
        
    - To choose an architecture(will be defaulted to VGG16 if not indicated):
        `python train.py data_directory --arch "vgg13"`
    - To set specific hyper parameters (learning rate, hidden units, epoch number):
        `python train.py data_directory --learning_rate 0.01 --hidden_units 512 --epochs 20`
    - To set GPU usage for training:
        `python train.py data_dir --gpu`

## How to use predict.py

- Basic usage:
  `python predict.py /path/to/image /path/to/checkpoint_file`
  will return flower class and probablity (will not display name unless indicated with option stated below)
- Options:
   - To set number of top K most likely classes (defaulted to 5 if not indicated):
      `python predict.py /path/to/image /path/to/checkpoint --top_k 3`
   - To get names of flowers instead of class number:
      `python predict.py /path/to/image /path/to/checkpoint --category_names cat_to_name.json`
   - To set GPU for inference:
      `python predict.py /path/to/image /path/to/checkpoint --gpu`
