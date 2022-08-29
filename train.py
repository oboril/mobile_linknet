import tensorflow as tf
import mobile_linknet as ml

images = "fib1_1.jpg fib1_2.jpg fib1_3.jpg img1.jpg msc_1.jpg pH_1.jpg pH_2.jpg pH_3.jpg simone_3.tif".split()
dataset = ml.load_dataset("images/train/",["images/cells/","images/nuclei/"], ["msc_1.jpg","fib1_1.jpg"],(96*4,96*3))

augmented = dataset.repeat(8).map(ml.augment).shuffle(64).batch(16, drop_remainder=True)
dataset = dataset.batch(1)

model = ml.Mobile_LinkNet_SAM()

train_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.75,
        patience=3, verbose=1
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir="logs"
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "checkpoint.h5",
        monitor='val_loss',
        save_best_only=True
    )
]

model.compile(optimizer="adam",loss=ml.metrics.IoU_focal,metrics=[ml.metrics.accuracy, ml.metrics.precision, ml.metrics.recall])
model.fit(augmented, validation_data=dataset, epochs=100, callbacks=train_callbacks)

model.save("saved_model.h5")

