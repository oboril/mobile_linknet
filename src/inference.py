"""
This script can be used to segment cells.

It nicely shows how to use the 'postprocessing' module,
but I would recommend using the GUI for seeing how a model performs.

You don't need GPU if you segment only one or few images (personal laptip is just fine).
For batch processing, I would recommend using desktop with GPU acceleration.
"""

import mobile_linknet as ml
from matplotlib import pyplot as plt

# Open image
image = "images/train/msc_2.jpg"
image = ml.utils.open_image(image, size=[224*4,224*3])

# Load model
model = ml.Mobile_LinkNet_SAM(load_saved="model.h5")

# Run prediction
predicted = ml.postprocessing.get_prediction(model, image) # Runs the CNN

cells, masks = ml.postprocessing.segment_cells(predicted, cells_smoothing_sigma=0.01) # Uses watershed etc to obtain invidivual cells

overlay = ml.postprocessing.overlay_segmentation_masks(image/255, masks) # Creates the overlay of segmented cells

print(f"Found {len(cells)} cells")

# Show results
ml.show_image(ml.postprocessing.prediction_to_rgb(predicted), block=False) # Shows the raw output of the CNN
ml.show_image(overlay, block=False) # Shows the segmented overlay

plt.figure()
plt.imshow(image)
plt.plot(cells[:,1], cells[:,0], 'rx') # Shows the cell centers as red crosses
plt.show()