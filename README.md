[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
# Mobile LinkNet
This repository contains an implementation of Mobile LinkNet model which can be used for efficient image segmentation.

More information can be found here: [janoboril.pythonanywhere.com/cell-segmentation](https://janoboril.pythonanywhere.com/cell-segmentation)

## Requirements:
Python >= 3.9  
tensorflow >= 2.9  
keras  
tensorflow-addons  
scikit-image  
opencv-python  
flask  
matplotlib  

## Setup and training
For GPU acceleration, you also need the correct version of CUDA and CUDNN.

I would recommend training the model on desktop with GPU (at least 8 GB vRAM), or on a similar setup on cloud (like Paperspace Gradient Notebooks).

## Contents - development and training
The relevant files for model development and training are in `src/`.

Files:
```
train.py - the basic training pipeline
inference.py - the basic pipeline for using the trained model to segment cells
test_augmentation.py - a script to test the augmentation pipeline
images - contains the training dataset that I used
mobile_linknet - my custom library, it contains:
    - __init__.py - basically a header file
    - models.py - definition of the model and gradient handling
    - metrics.py - the definitions of loss functions and metrics
    - postprocessing.py - functions which help to process the CNN output, this was somewhat optimized for speed
    - utils.py - collection of various helper functions
    - augmentation.py - the augmentation pipeline, this can be tweaked to make the most from the training dataset
    - random_background.png - this image is used to create pseudo-random noise during the augmentation (mostly blobs in HSV color-space)
```

I don't think is much need to modify the mobile_linknet library, except for:
 * tweaking the augmentation pipeline (say reducing the extent of the augmentation if larger dataset is used)
 * tweaking the model, namely changing the number of output channels (for example adding a channel to detect dead cells)

If you are changing the output layers of the model, you need to carefully choose the activation function and make sure that the loss function is appropriate.

## GUI
The folder `GUI/` contains standalone web-based GUI for inference. To open the GUI, run `python app.py` and open the `localhost` link that will appear in the terminal.