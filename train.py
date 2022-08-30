import tensorflow as tf
import mobile_linknet as ml

images = "fib1_1.jpg fib1_2.jpg fib1_3.jpg img1.jpg msc_1.jpg pH_1.jpg pH_2.jpg pH_3.jpg simone_3.tif".split()
dataset = ml.load_dataset("images/train/",["images/cells/","images/nuclei/"], images,(96*4,96*3))

augmented = dataset.repeat(8).map(ml.augment, num_parallel_calls=16).shuffle(8*9).batch(4,drop_remainder=True).batch(16,drop_remainder=True).prefetch(1)
dataset = dataset.map(lambda i,l: (ml.preprocess_input(i),l), num_parallel_calls=16).batch(9).prefetch(1)
dataset2 = dataset.map(lambda i,l: (ml.preprocess_input(i),l), num_parallel_calls=16).batch(4).prefetch(1)

w = tf.summary.create_file_writer('test/logs')
with tf.summary.create_file_writer('logs/SAM_rho0p1').as_default():
    for images,labels in augmented.take(1):
        tf.summary.image("train_image", images[0]+0.5, step=0)
        tf.summary.image("train_labels", ml.postprocessing.prediction_to_rgb(labels[0]), step=0)
    for images,labels in dataset.take(1):
        tf.summary.image("test_image", images+0.5, step=0)
        tf.summary.image("test_labels", ml.postprocessing.prediction_to_rgb(labels), step=0)


model = ml.Mobile_LinkNet_SAM(rho=0.001, load_saved="saved_model.h5")

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
        #profile_batch=(3,10)
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "checkpoint.h5",
        monitor='val_loss',
        save_best_only=True
    )
]
train_callbacks[0].best=1.
callbacks = tf.keras.callbacks.CallbackList(train_callbacks, model=model)

optimizer = tf.keras.optimizers.Adam(0.001)
model.compile(optimizer=optimizer,loss=ml.metrics.IoU_focal,metrics={"accuracy":ml.metrics.accuracy, "precision":ml.metrics.precision, "recall":ml.metrics.recall})

EPOCHS = 5000

for epoch in range(1001,EPOCHS+1):
    print(f"Epoch {epoch} ", end="", flush=True)
    callbacks.on_epoch_begin(epoch)

    loss_count = 0.
    loss = 0.

    for batch, (images_all, labels_all) in enumerate(augmented):
        print("|", end="", flush=True)
        callbacks.on_batch_begin(batch)
        for images, labels in zip(images_all, labels_all):
            print("#", end="", flush=True)
            loss_count += 1.
            loss += model.add_gradient_SAM([images,labels])
        model.apply_gradients()
        callbacks.on_batch_end(batch, {"loss":loss/loss_count})

    print("| ", end="", flush=True)

    metrics = {}
    test_count = 0.
    for images, labels in dataset:
        test_count += len(images)
        m = model.test_step([images,labels])
        metrics = {key:val*len(images) + (metrics[key] if key in metrics else 0) for key,val in m.items()}
    metrics = {key: val/test_count for key,val in metrics.items()}

    metrics2 = {}
    test_count = 0.
    for images, labels in dataset2:
        test_count += len(images)
        m = model.test_step([images,labels])
        metrics2 = {key:val*len(images) + (metrics2[key] if key in metrics2 else 0) for key,val in m.items()}
    metrics2 = {key: val/test_count for key,val in metrics2.items()}

    print(metrics)
    print(metrics2)
    
    metrics = {"loss": loss/loss_count} | metrics

    print(" - ".join([f"{key} {val:0.4f}" for key,val in metrics.items()]), flush=True)

    callbacks.on_epoch_end(epoch, metrics)
        

model.save("saved_model.h5")

