import mobile_linknet as ml
from matplotlib import pyplot as plt

# open image
image = "images/train/simone_1.tif"
image = ml.utils.open_image(image, size=[96*4,96*3])

# load model
model = ml.Mobile_LinkNet_SAM(load_saved="checkpoint.h5")

# run prediction
predicted = ml.postprocessing.get_prediction(model, image)

cells, masks = ml.postprocessing.segment_cells(predicted, cells_smoothing_sigma=0.01)

overlay = ml.postprocessing.overlay_segmentation_masks(image/255, masks)

# show results
ml.show_image(ml.postprocessing.prediction_to_rgb(predicted), block=False)
ml.show_image(overlay, block=False)

plt.figure()
plt.imshow(image)
plt.plot(cells[:,1], cells[:,0], 'rx')
plt.show()