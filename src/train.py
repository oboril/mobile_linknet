"""
This is the script I use for training the model.

It is important to always check that all settings (paths, batch sizes, model definition, training loop, etc)
are correct before starting the training.

If GPU is available (I would not train without GPU), you need to specify that TensorFlow should use GPU acceleration.

You should monitor the training progress using TensorBoard (the logs will be in the 'logs' folder)
"""

import tensorflow as tf
import mobile_linknet as ml

# I use list of all annotated images (this could be automated by just loading all files in directory)
images = "fib1_1.jpg fib1_2.jpg fib1_3.jpg img1.jpg msc_1.jpg pH_1.jpg pH_2.jpg pH_3.jpg simone_3.tif".split()

# Load the dataset (the three folders contain: raw images, cytoplasm masks, nuclei masks). The images must have the same name in all the folders.
# The extension for the raw image can be anything (specified by the file name), the extension of the masks is always .tif
# I think (224*4,224*3) is a good input size
dataset = ml.load_dataset("images/train/",["images/cytoplasm/","images/nuclei/"], images, (224*4,224*3))

# Create the datasets
# Currently, I have 9 training images, repeat(8) gives 8*9=72 training images
# For training, 2 images are processed in parallel (depending on GPU RAM this can be more) and 32 calls are averaged in a batch.
# Thus, the effective batch size is 2*32 = 64
# Larger batch sizes converge faster, but take longer to compute and converge to worse minima. It is important to choose a good batch size
train_data = dataset.repeat(8).map(ml.augment, num_parallel_calls=16).shuffle(8*9).batch(2,drop_remainder=True).batch(32,drop_remainder=True).prefetch(1)

# The testing dataset are just unaugmented training data. This is not good and should be changed for actual testing data.
test_data = dataset.map(lambda i,l: (ml.preprocess_input(i),l), num_parallel_calls=16).batch(9).prefetch(1)

# This can be used to log some training images and masks into TensorBoard. It is always a good idea to check the data before training
if False:
    w = tf.summary.create_file_writer('test/logs')
    with tf.summary.create_file_writer('logs/SAM_rho0p1').as_default():
        for images,labels in augmented.take(1):
            tf.summary.image("train_image", images[0]+0.5, step=0)
            tf.summary.image("train_labels", ml.postprocessing.prediction_to_rgb(labels[0]), step=0)
        for images,labels in dataset.take(1):
            tf.summary.image("test_image", images+0.5, step=0)
            tf.summary.image("test_labels", ml.postprocessing.prediction_to_rgb(labels), step=0)

# Create a new model. rho specifies the perimiter for SAM (Sharpness Aware Minimization)
model = ml.Mobile_LinkNet_SAM(rho=0.01)
# It is also possible to load pretrained model
#model = ml.Mobile_LinkNet_SAM(rho=0.01, load_saved="saved_model.h5")

# Note that the 'Mobile_LinkNet_SAM' is a custom class and differs significantly from TensorFlow class 'Model'

# Callbacks are the prefered way of logging/changing anything during the training
train_callbacks = [
    #tf.keras.callbacks.EarlyStopping(
    #    monitor="val_loss", patience=20,
    #    restore_best_weights=True
    #),
    #tf.keras.callbacks.ReduceLROnPlateau(
    #    monitor="val_loss", factor=0.5,
    #    patience=8, verbose=1
    #),
    tf.keras.callbacks.TensorBoard(
        log_dir="logs/SAM_rho0p1"
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "checkpoint.h5",
        monitor='val_loss',
        save_best_only=True
    )
]
train_callbacks[0].best=1.
callbacks = tf.keras.callbacks.CallbackList(train_callbacks, model=model)

# Initialize the optimizer. Adam and RMSprop work good, AdaBound might converge to better minima
optimizer = tf.keras.optimizers.Adam(0.001)

# Compile the model, similar to TensorFlow model
model.compile(optimizer=optimizer,loss=ml.metrics.IoU_focal,metrics={"accuracy":ml.metrics.accuracy, "precision":ml.metrics.precision, "recall":ml.metrics.recall})

# Specify the maximum number of epochs. In practice, I stop earlier based on the validation loss or time constrains
EPOCHS = 5000

# This is the training loop
for epoch in range(1,EPOCHS+1):
    print(f"Epoch {epoch} ", end="", flush=True)
    callbacks.on_epoch_begin(epoch)

    # These variables keep track of the loss and will be used to calculate the average loss
    loss_count = 0.
    loss = 0.

    # Each 'images_all' is a tensor containing 32*2 training images
    for batch, (images_all, labels_all) in enumerate(train_data):
        print("|", end="", flush=True)
        callbacks.on_batch_begin(batch)

        # This loops runs the individual minibatches (2 images at a time)
        for images, labels in zip(images_all, labels_all):
            print("#", end="", flush=True)
            loss_count += 1.
            # The gradient is accumulated using this function
            loss += model.add_gradient_SAM([images,labels])
            # It is possible not to use SAM:
            # loss += model.add_gradient([images,labels])
            # If the CNN is the bottleneck, SAM can slow down the training cca 2x
            # However, for most setups the bottleneck is the data augmentation pipeline

        # After all 64 images are processed, the averaged gradient is applied
        model.apply_gradients()
        callbacks.on_batch_end(batch, {"loss":loss/loss_count})

    print("| ", end="", flush=True)

    # Getting metrics
    metrics = {"loss": loss/loss_count} | model.test_all(test_data)

    print(" - ".join([f"{key} {val:0.4f}" for key,val in metrics.items()]), flush=True)

    callbacks.on_epoch_end(epoch, metrics)
        
# If the training loop finishes, save the final model
model.save("saved_model.h5")

