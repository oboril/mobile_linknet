import tensorflow as tf
from keras.layers import Dense, Conv2D, Add, DepthwiseConv2D, BatchNormalization, UpSampling2D, ReLU
from keras.applications.mobilenet_v2 import MobileNetV2

def raw_mobile_linknet(input_shape=None):
    """
    The architecture of this CNN is described at janoboril.pythonanywhere.com/cell-segmentation.
    There are 17 encoder and 10 decoder layers based on mobilenet inverse residual blocks.
    """

    def inverted_residual(input_tensor, name, projected_channels = None, expanded_channels=None, residual_conv=None):
        input_channels = input_tensor.shape[-1]
        if projected_channels is None:
            projected_channels = input_channels
        if expanded_channels is None:
            expanded_channels = 6*input_channels
        if residual_conv is None:
            if input_channels == projected_channels:
                residual_conv = False
            else:
                residual_conv = True

        expand = Conv2D(expanded_channels, (1,1), name=name+"_expand_pointwise")(input_tensor)
        expand_bn = BatchNormalization(name=name+"_expand_BN")(expand)
        expand_relu = ReLU(max_value=6, name=name+"_expand_ReLU")(expand_bn)

        depthwise_conv = DepthwiseConv2D((3,3), padding="same", name=name+"_depthwise_conv")(expand_relu)
        depthwise_bn = BatchNormalization(name=name+"_depthwise_BN")(depthwise_conv)
        depthwise_relu = ReLU(max_value=6, name=name+"_depthwise_ReLU")(depthwise_bn)

        project = Conv2D(projected_channels, (1,1), name=name+"_project_pointwise")(depthwise_relu)
        project_BN = BatchNormalization(name=name+"_project_BN")(project)

        if residual_conv:
            residual_conv = Conv2D(projected_channels, (1,1), name=name+"_residual_conv")(input_tensor)
            residual_add = Add(name=name+"_residual_add")([project_BN,residual_conv])
        else:
            residual_add = Add(name=name+"_residual_add")([project_BN,input_tensor])
        
        return residual_add
    
    encoder = MobileNetV2(include_top=False,input_shape=input_shape)

    # 7x7
    decoder1 = inverted_residual(encoder.get_layer("block_16_project_BN").output, "decoder1", projected_channels=96, expanded_channels=96*6)
    decoder1_upsample = UpSampling2D(interpolation="bilinear", name="decoder1_upsample")(decoder1)

    # 14x14
    decoder2_link = Add(name="decoder2_link")([decoder1_upsample,encoder.get_layer("block_12_add").output])
    decoder2 = inverted_residual(decoder2_link, "decoder2")

    decoder3 = inverted_residual(decoder2, "decoder3")

    decoder4 = inverted_residual(decoder3, "decoder4", projected_channels = 32)
    decoder4_upsample = UpSampling2D(interpolation="bilinear", name="decoder4_upsample")(decoder4)

    # 28x28
    decoder5_link = Add(name="decoder5_link")([decoder4_upsample,encoder.get_layer("block_5_add").output])
    decoder5 = inverted_residual(decoder5_link, "decoder5")

    decoder6 = inverted_residual(decoder5, "decoder6")

    decoder7 = inverted_residual(decoder6, "decoder7", projected_channels = 24)
    decoder7_upsample = UpSampling2D(interpolation="bilinear", name="decoder7_upsample")(decoder7)

    # 56x56
    decoder8_link = Add(name="decoder8_link")([decoder7_upsample,encoder.get_layer("block_2_add").output])
    decoder8 = inverted_residual(decoder8_link, "decoder8")

    decoder9 = inverted_residual(decoder8, "decoder9", projected_channels = 16)
    decoder9_upsample = UpSampling2D(interpolation="bilinear", name="decoder9_upsample")(decoder9)

    # 112x112
    decoder10_link = Add(name="decoder10_link")([decoder9_upsample,encoder.get_layer("expanded_conv_project_BN").output])
    decoder10 = inverted_residual(decoder10_link, "decoder10")

    dense_conv1 = Conv2D(16,(1,1),padding="same",name="dense_conv1")(decoder10)
    dense_BN1 = BatchNormalization(name="dense_BN1")(dense_conv1)
    dense_relu1 = ReLU(max_value=6,name="dense_ReLU1")(dense_BN1)
    dense_conv2 = Conv2D(16,(1,1),padding="same",name="dense_conv2")(dense_relu1)
    dense_BN2 = BatchNormalization(name="dense_BN2")(dense_conv2)
    dense_relu2 = ReLU(max_value=6,name="dense_ReLU2")(dense_BN2)
    dense_upsample = UpSampling2D(interpolation="bilinear", name="dense_upsample")(dense_relu2)

    # 224x224
    output_conv = Conv2D(2, (3,3), padding="same", activation="sigmoid", name="output_conv")(dense_upsample)

    model = tf.keras.Model(
        inputs = encoder.input,
        outputs = output_conv
    )

    for layer in model.layers:
        if isinstance(layer, BatchNormalization):
            layer.momentum=0.95

    return model

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

    def save(self,filename,*args,**kwargs):
        try:
            self.linknet_model.save(filename)
        except Exception as ex:
            print("Saving failed,",ex)

    def compile(self,optimizer,loss,metrics):
        self.my_optimizer = optimizer
        self.my_loss = loss
        self.my_metrics = metrics
        self.my_metrics["loss"] = self.my_loss

    @tf.function
    def add_gradient(self, data):
        (images, labels) = data

        with tf.GradientTape() as tape:
            predictions = self.linknet_model(images)
            loss = self.my_loss(labels, predictions)

        gradients = tape.gradient(loss, self.linknet_model.trainable_variables)

        for g, accum in zip(gradients, self.accumulated_gradients):
            accum.assign_add(g)
        
        self.accumulated_gradients_count.assign_add(1.)

        return loss

    @tf.function
    def apply_gradients(self):      
        for grad in self.accumulated_gradients:
            grad.assign(grad/self.accumulated_gradients_count)
        
        self.my_optimizer.apply_gradients(
            zip(self.accumulated_gradients, self.linknet_model.trainable_variables)
        )

        for grad in self.accumulated_gradients:
            grad.assign(grad*0.)

        self.accumulated_gradients_count.assign(0)
        
    @tf.function
    def add_gradient_SAM(self, data):
        (images, labels) = data

        # Calculate e_w
        with tf.GradientTape() as tape:
            predictions = self.linknet_model(images)
            loss = self.my_loss(labels, predictions)

        gradients = tape.gradient(loss, self.linknet_model.trainable_variables)

        grad_norm = self._grad_norm(gradients)
        scale = self.rho / (grad_norm + 1e-12)

        # Adjust model weights against gradient
        e_ws = []
        for (grad, param) in zip(gradients, self.linknet_model.trainable_variables):
            e_w = grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)

        # Get new gradient
        with tf.GradientTape() as tape:
            predictions = self.linknet_model(images)
            loss = self.my_loss(labels, predictions)

        gradients = tape.gradient(loss, self.linknet_model.trainable_variables)

        # Restore weights
        for (param, e_w) in zip(self.linknet_model.trainable_variables, e_ws):
            param.assign_sub(e_w)
        
        # Accumulate gradient
        for g, accum in zip(gradients, self.accumulated_gradients):
            accum.assign_add(g)
        
        self.accumulated_gradients_count.assign_add(1.)

        return loss

    @tf.function
    def test_step(self, data):
        (images, labels) = data
        predictions = self.linknet_model(images, training=False)
        metrics = {"val_"+key: m(labels, predictions) for key,m in self.my_metrics.items()}
        return metrics

    def test_all(self, test_data):
        metrics = {}
        test_count = 0.
        for images, labels in test_data:
            batch_size = float(len(images))
            test_count += batch_size
            m = self.test_step([images,labels])
            metrics = {key:val*batch_size + (metrics[key] if key in metrics else 0) for key,val in m.items()}
        metrics = {key: val/test_count for key,val in metrics.items()}
        return metrics

    @tf.function
    def _grad_norm(self, gradients):
        norm = tf.norm(
            tf.stack([
                tf.norm(grad) for grad in gradients if grad is not None
            ])
        )
        return norm

    @tf.function
    def call(self, image):
        return self.linknet_model.call(image)
