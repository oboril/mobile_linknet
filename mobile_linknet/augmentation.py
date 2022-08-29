import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input
from mobile_linknet.utils import open_image
from math import pi

AUGMENTATION_CONFIG= {
    "random_corners_shift":50,
    "shear": 0.2,
    "zoom": 0.2,
    "hue_max_delta": 0.1,
    "brightness_max_delta": 0.7,
    "contrast_max_delta": 0.7,
    "sharpness_factor": 2.,
    "noise": 5,
    "random_background_sigma": 5.,
    "random_background_coeff": 0.3
}

random_background_image = tf.constant(open_image("mobile_linknet/random_background.png"), dtype=tf.float32)/127-1
@tf.function
def random_background(image, blur_max, aug_range, seed):
    background = random_background_image

    kernel = get_affine_transform_kernel(image.shape[0],image.shape[1], 0.3,0.3,50, 250, input_width=background.shape[0]//2, input_height=background.shape[1]//2)
    background = tfa.image.transform(background, kernel, output_shape=image.shape[:2])

    sigma = tf.random.uniform([], minval=0, maxval=blur_max)
    background = gaussian_blur(background, int(blur_max*2), sigma)

    coeff = tf.random.uniform([],0,aug_range, dtype=tf.float32)

    hsv = tf.image.rgb_to_hsv(image/255)
    hsv = hsv + background*coeff
    image = tf.image.hsv_to_rgb(hsv)*255
    
    return image

rng = tf.random.Generator.from_seed(123, alg='philox')
def seed():
    return rng.make_seeds(2)[0]

@tf.function
def get_affine_transform_kernel(height,width,shear,zoom,random_corners_shift,random_shift=0, input_width=None, input_height=None):
    if input_width is None: input_width = width
    if input_height is None: input_height = height

    corners = tf.constant([[-input_width/2,-input_height/2],[input_width/2,-input_height/2],[input_width/2,input_height/2],[-input_width/2,input_height/2]])
    corners_transormed = tf.constant([[-width/2,-height/2],[width/2,-height/2],[width/2,height/2],[-width/2,height/2]])

    # random shift corners
    corners += tf.random.stateless_normal([4,2], seed(), stddev=random_corners_shift)

    # random_shift
    corners += tf.random.stateless_uniform([2], seed(), -random_shift, random_shift)

    # random flip
    flip = tf.random.stateless_uniform([], seed(), 0, 1)
    flip = tf.math.round(flip)
    flip_matrix = tf.constant([[-1.,0.],[0.,1.]],dtype=tf.float32)
    flip_matrix = tf.math.pow(flip_matrix, flip+1.)

    corners_transormed = tf.transpose(corners_transormed)
    corners_transormed = tf.matmul(flip_matrix,corners_transormed)

    #random zoom
    corners_transormed *= tf.random.stateless_normal([], seed(), mean=1., stddev=zoom)

    #random shear
    shear_factor = tf.random.stateless_normal([], seed(), stddev=shear)
    shear_matrix = tf.constant([[0.,1.],[0.,0.]])*shear_factor + tf.constant([[1.,0.],[0.,1.]])
    corners_transormed = tf.matmul(shear_matrix,corners_transormed)

    #random rotate
    s = seed()
    angle = tf.random.stateless_uniform([], s, 0, pi*2)
    rotation_matrix = tf.stack([
        tf.stack([tf.math.cos(angle),tf.math.sin(angle)]),
        tf.stack([-tf.math.sin(angle),tf.math.cos(angle)])
        ])
    
    corners_transormed = tf.matmul(rotation_matrix,corners_transormed)
    corners_transormed = tf.transpose(corners_transormed)

    corners_transormed += tf.constant([width/2,height/2])
    corners += tf.constant([width/2,height/2])

    #solve kernel
    ones = tf.ones([4])
    X1 = tf.stack([corners_transormed[:,0],corners_transormed[:,1],ones,ones*0,ones*0,ones*0,-corners_transormed[:,0]*corners[:,0],-corners_transormed[:,1]*corners[:,0]],axis=1)
    X2 = tf.stack([ones*0,ones*0,ones*0,corners_transormed[:,0],corners_transormed[:,1],ones,-corners_transormed[:,0]*corners[:,1],-corners_transormed[:,1]*corners[:,1]], axis=1)
    Y1 = corners[:,0]
    Y2 = corners[:,1]

    X = tf.concat([X1,X2],axis=0)
    Y = tf.concat([Y1,Y2],axis=0)

    Xinv = tf.linalg.inv(X)

    kernel = tf.linalg.matvec(Xinv,Y)

    return kernel
    
@tf.function
def gaussian_blur(image, shape, sigma):
    sigma = tf.clip_by_value(sigma, 0.1, 9999)
    kernel = tf.linspace(-shape//2, shape//2, shape)
    kernel = tf.cast(kernel, tf.float32)
    kernel = tf.exp(-kernel**2/2/(sigma**2))
    kernel = kernel/tf.reduce_sum(kernel)
    kernel = tf.convert_to_tensor([[kernel, kernel*0, kernel*0],[kernel*0, kernel, kernel*0],[kernel*0, kernel*0, kernel]])
    kernel = tf.transpose(kernel, (2,0,1))

    image = tf.nn.conv1d(image, kernel, stride=1, padding="SAME")
    image_T = tf.transpose(image, (1,0,2))
    image_T = tf.nn.conv1d(image_T, kernel, stride=1, padding="SAME")
    image = tf.transpose(image_T, (1,0,2))
    return image

@tf.function
def augment(image, labels, config=AUGMENTATION_CONFIG, preprocess_input=True):
    """
    Automatically applies the following random augmentations (can be tweaked by config):
     - Rotation, flip, shear, skew, zoom
     - Brightness, hue, contrast, sharpness
     - Random noise, random background
    
    Also preprocesses the input by default (preprocess_input=True).
    """
    # pad labels to have 3 channels
    labels = tf.pad(labels, ((0,0),(0,0),(0,1)))

    # Affine transformation
    affine_transform_kernel = get_affine_transform_kernel(image.shape[0],image.shape[1],config["shear"],config["zoom"],config["random_corners_shift"],0)
    means = tf.reduce_mean(image, axis=[-2,-3])
    image = tfa.image.transform(image-means, affine_transform_kernel,fill_value=0)+means
    labels = tfa.image.transform(labels, affine_transform_kernel,fill_value=0)

    # hue
    s = seed()
    image = tf.image.stateless_random_hue(image, config["hue_max_delta"], seed=s)
    # brightness
    s = seed()
    image = tf.image.stateless_random_brightness(image, config["brightness_max_delta"], seed=s)
    # contrast
    s = seed()
    image = tf.image.stateless_random_contrast(image, 1-config["contrast_max_delta"], 1+config["contrast_max_delta"], seed=s)
    # circles
    image = random_background(image, config["random_background_sigma"], config["random_background_coeff"], seed)
    # sharpen
    sharpness = tf.random.normal([])*config["sharpness_factor"]
    sharpness_sign = tf.sign(sharpness)
    image_sharp = tfa.image.sharpness(image, sharpness_sign*sharpness)
    image_blur = gaussian_blur(image, 10, sharpness_sign*sharpness/2)
    sharpness_positive = sharpness_sign/2+0.5
    image = sharpness_positive*image_sharp + (1-sharpness_positive)*image_blur
    # noise
    noise = tf.random.uniform([],0,config["noise"])
    image += tf.random.normal(image.shape, stddev=noise)

    # preprocess for CNN
    if preprocess_input:
        image = mobilenetv2_preprocess_input(image)

    return image, labels[:,:,:2]
