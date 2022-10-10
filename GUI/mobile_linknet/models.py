import tensorflow as tf
from keras.layers import Dense, Conv2D, Add, DepthwiseConv2D, BatchNormalization, UpSampling2D, ReLU
from keras.applications.mobilenet_v2 import MobileNetV2


class Mobile_LinkNet_SAM(tf.keras.Model):
    """
    This class is adapted from https://colab.research.google.com/github/sayakpaul/Sharpness-Aware-Minimization-TensorFlow/blob/main/SAM.ipynb#scrollTo=ENUFObDrbqlC
    """
    def __init__(self, rho=0.05, load_saved=None, input_shape=None):
        super(Mobile_LinkNet_SAM, self).__init__()
        if load_saved is None:
            self.linknet_model = raw_mobile_linknet(input_shape)
            self.linknet_model.compile()
        else:
            self.linknet_model = tf.keras.models.load_model(load_saved)
        self.rho = rho
    
        self.accumulated_gradients = [tf.Variable(var*0., trainable=False) for var in self.linknet_model.trainable_variables]
        self.accumulated_gradients_count = tf.Variable(0., trainable=False)

    @tf.function
    def call(self, image):
        return self.linknet_model.call(image)