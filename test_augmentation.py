import mobile_linknet as ml
from matplotlib import pyplot as plt

dataset = ml.load_dataset("images/train/",["images/cells/","images/nuclei/"], ["msc_1.jpg"],(96*4,96*3))

augmented = dataset.repeat(5).map(lambda i,l: ml.augment(i,l,preprocess_input=False))

for image,labels in augmented.take(5):
    labels = ml.postprocessing.prediction_to_rgb(labels)
    ml.show_image(labels,block=False)
    ml.show_image(image/255.)

