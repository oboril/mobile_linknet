import mobile_linknet as ml
from matplotlib import pyplot as plt

image = "images/train/msc_1.jpg"
image = ml.utils.open_image(image, size=[128*4,128*3])

model = ml.Mobile_LinkNet_SAM(load_saved="trained_model.h5")

predicted = ml.postprocessing.get_prediction(model, image)

ml.show_image(ml.postprocessing.prediction_to_rgb(predicted))

cells, masks = ml.postprocessing.segment_cells(predicted)

plt.imshow(image)
plt.plot(cells[:,1], cells[:,0], 'rx')
plt.show()