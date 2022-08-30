import numpy as np
from scipy import ndimage
from mobile_linknet.augmentation import mobilenetv2_preprocess_input as preprocess_input

from skimage.segmentation import watershed
from skimage.morphology import h_maxima
from skimage.measure import regionprops
from skimage.color import hsv2rgb

def get_prediction(model, image):
    """Preprocesses the image and runs inference. Returns the predicted masks"""
    prediction = model.predict(preprocess_input(image.astype(np.float32).reshape((1,)+image.shape)))[0,:,:]

    return prediction

def prediction_to_rgb(predicted):
    if len(predicted.shape) == 3:
        cells = predicted[:,:,0]
        nuclei = predicted[:,:,1]
        return np.array([cells-nuclei, nuclei, nuclei*0]).transpose(1,2,0)
    elif len(predicted.shape) == 4:
        cells = predicted[:,:,:,0]
        nuclei = predicted[:,:,:,1]
        return np.array([cells-nuclei, nuclei, nuclei*0]).transpose(1,2,3,0)
    else:
        raise Exception("Invalid input shape")

def get_cell_centers(predicted, distance_threshold=1., prob_threshold=0.5):
    """
    Uses thresholding and distance transform to find centres of the cells.
    Returns list of cell coordinates and image with markers
    """
    # get predicted nuclei
    nuclei = predicted[:,:,1]

    # threshold nuclei
    nuclei_thresholded = nuclei > prob_threshold

    # calculate distance transform
    distance_nuclei = ndimage.distance_transform_edt(nuclei_thresholded)

    # add small arbitrary value to ensure unique maxima
    distance_nuclei += np.linspace(-1e-2, 1e-2,distance_nuclei.shape[0]*distance_nuclei.shape[1]).reshape(distance_nuclei.shape) 

    # find maxima
    peaks_mask = h_maxima(distance_nuclei, distance_threshold)

    # label the maxima
    markers, peaks_count = ndimage.label(peaks_mask)

    # get maxima centers
    peaks = np.array(list(map(lambda p: p["centroid"], regionprops(markers))))

    return peaks, markers

def segment_cells(predicted, distance_threshold=1., prob_threshold=0.5, cells_smoothing_sigma=1.):
    """
    Returns list of cell centres and watershed labeles of segmented cells
    """
    # Get cell centres
    peaks, markers = get_cell_centers(predicted, distance_threshold, prob_threshold)

    # Smooth and threshold cell prediction
    cells = predicted[:,:,0]
    cells = ndimage.gaussian_filter(cells, cells_smoothing_sigma)
    cells_thresholded = cells > prob_threshold

    labels = watershed(-cells, markers, mask=cells_thresholded, watershed_line=True).astype(int)

    return peaks, labels

def overlay_segmentation_masks(image, labels, opacity=0.5):
    colors = [hsv2rgb([hue, 1., 1.]) for hue in np.linspace(0.,1.,10,endpoint=False)]
    labels = labels.astype(int)

    hue = (labels%10)/10
    ones = np.ones(hue.shape)
    hsv = np.stack([hue, ones, ones], axis=2)

    overlay = hsv2rgb(hsv)

    overlay_mask = np.where(labels == 0, 0, opacity)
    image_mask = np.where(labels == 0, 1., 1.-opacity)

    return (overlay.transpose(2,0,1)*overlay_mask + image.transpose(2,0,1)*image_mask).transpose(1,2,0)
