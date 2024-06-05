import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

SEED = 10
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

train_path = 'Img/train'
validation_path = 'Img/validation'

batch_size = 1024
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    image_size=(100, 40),
    batch_size=batch_size,
    seed=SEED
).cache().prefetch(buffer_size=AUTOTUNE)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    validation_path,
    image_size=(100, 40),
    batch_size=batch_size,
    seed=SEED
).cache().prefetch(buffer_size=AUTOTUNE)

train_image_count = 168000
validation_image_count = 28000
print(f"Number of training samples: {train_image_count}")
print(f"Number of validation samples: {validation_image_count}")

train_steps_per_epoch = np.ceil(train_image_count / batch_size)
validation_steps_per_epoch = np.ceil(validation_image_count / batch_size)

tf.keras.mixed_precision.set_global_policy('mixed_float16')

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    vgg16 = VGG16(weights=None, include_top=False, input_shape=(100, 40, 3))
    model = tf.keras.Sequential([
    vgg16,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(8, activation='softmax')
    ])

    model_checkpoint_callback = ModelCheckpoint(
    filepath='Results/model/Mel_CNN(Generalization-Dropout,ReduceLR)_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.h5',
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_accuracy',
        mode='max',
        factor=0.1,
        patience=5,
        min_lr=1e-6,
        verbose=1)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

model.summary()

epochs = 50

history = model.fit(
    train_dataset,
    steps_per_epoch=train_steps_per_epoch,
    epochs=epochs,
    validation_data=validation_dataset,
    validation_steps=validation_steps_per_epoch,
    callbacks=[model_checkpoint_callback, reduce_lr_callback]
)
