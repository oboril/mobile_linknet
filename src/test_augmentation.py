"""
I used this script to tune parameters of the augmentation pipeline.

It is always a good idea to check that the augmented images look as expected before training.

The smaller image size (96*4,96*3) compared to (224*4,224*3) is used to make this faster.
"""

import mobile_linknet as ml
from matplotlib import pyplot as plt

# Create a dataset from a single image
dataset = ml.load_dataset("images/train/",["images/cytoplasm/","images/nuclei/"], ["msc_1.jpg"],(96*4,96*3))

# Augment the image (this creates 5 different augmented images from the same training image)
augmented = dataset.repeat(5).map(lambda i,l: ml.augment(i,l,preprocess_input=False))

# Display the augmented image and labels
for image,labels in augmented.take(5):
    labels = ml.postprocessing.prediction_to_rgb(labels)
    ml.show_image(labels,block=False)
    ml.show_image(image/255.)

