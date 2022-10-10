import numpy as np
import tifffile
import cv2
from matplotlib import pyplot as plt
import time
import tensorflow as tf

def open_image(path, size=None, grayscale=False, nearest=False):
    if path.endswith(".tif") or path.endswith(".tiff"):
        img = tifffile.imread(path)
    else:
        img = cv2.imread(path)[...,::-1]
    
    if size is not None:
        img = cv2.resize(img,size,interpolation=cv2.INTER_LINEAR if not nearest else cv2.INTER_NEAREST)
    img = np.array(img)
    if grayscale:
        img = to_grayscale(img)
    return img

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def show_image(image, cmap="hot", block=True):
    plt.figure()
    plt.imshow(image,cmap)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show(block=block)

def load_labelled_image(train_folder, label_folders, name, size):
    """
    Loads training data. Training images can be .jpg or .tif, labels must be .tif
    """
    # Load source image 
    image = open_image(train_folder+name, size=size).astype(np.float32)

    # Load labels
    labels=[]
    for label_folder in label_folders:
        label = open_image(label_folder+name.replace(".jpg",".tif"), size=size, nearest=True)
        labels.append(label)

    labels = tf.stack(labels, axis=-1)

    return image, labels

def load_dataset(train_folder, label_folders, image_names, size):
    """
    Loads training data into dataset into RAM. Training images can be .jpg or .tif, labels must be .tif.
    Returns dataset containing the images and labels
    """
    images = []
    labels = []
    for image_name in image_names:
        img, label = load_labelled_image(train_folder, label_folders, image_name, size)
        images.append(img)
        labels.append(label)

    images = np.array(images).astype(np.float32)
    labels = np.array(labels).astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    return dataset