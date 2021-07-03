import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K

# config
batch_size = 2
img_height = 1376
img_width = 1038
epochs = 2


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def fit_model(train_ds, val_ds):
    with tf.device('/gpu:0'):
        # create model
        model = Sequential([
            layers.experimental.preprocessing.Rescaling(
                1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(64, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            layers.Dropout(0.1),
            layers.Conv2D(64, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            layers.Dropout(0.1),
            layers.Conv2D(128, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(256, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            layers.Dropout(0.1),
            layers.Conv2D(256, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(512, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            layers.Dropout(0.1),
            layers.Conv2D(512, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(1024, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            layers.Dropout(0.1),
            layers.Conv2D(1024, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            layers.Conv2DTranspose(
                512, (2, 2), strides=(2, 2), padding='same'),
            layers.Conv2D(512, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            layers.Dropout(0.1),
            layers.Conv2D(512, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            layers.Conv2DTranspose(
                256, (2, 2), strides=(2, 2), padding='same'),
            layers.Conv2D(256, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            layers.Dropout(0.1),
            layers.Conv2D(256, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            layers.Conv2DTranspose(
                128, (2, 2), strides=(2, 2), padding='same'),
            layers.Conv2D(128, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            layers.Dropout(0.1),
            layers.Conv2D(128, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same'),
            layers.Conv2D(64, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            layers.Dropout(0.1),
            layers.Conv2D(64, (3, 3), activation='elu',
                          kernel_initializer='he_normal', padding='same'),
            # layers.Conv2D(1, (1, 1), activation='sigmoid'),
        ])

        # compile model
        model.compile(optimizer='adam',
                      loss='binary_crossentropy', metrics=["accuracy"])

        # model summary
        model.summary()

        # train model
        results = model.fit(train_ds, validation_data=val_ds,
                            batch_size=batch_size, epochs=epochs)

    return results


if __name__ == '__main__':
    # disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

    # prepare data_dir
    data_dir = "data_dir"
    data_dir = pathlib.Path(data_dir)

    # get images and images_count
    images = list(data_dir.glob("*/*.*"))
    images_count = len(images)

    # prepare training data set
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # prepare validation data set
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # prepare class names
    class_names = train_ds.class_names
    print(class_names)

    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # normalize data from [0, 255] to [0, 1]
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

    # prepare normalized data set and use it
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    print(np.min(first_image), np.max(first_image))

    results = fit_model(train_ds, val_ds)

    # # create model
    # num_classes = 5
    # model = Sequential([
    #     layers.experimental.preprocessing.Rescaling(
    #         1./255, input_shape=(img_height, img_width, 3)),
    #     layers.Conv2D(16, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(32, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(64, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(num_classes)
    # ])

    # # compile model
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(
    #                   from_logits=True),
    #               metrics=['accuracy'])

    # # model summary
    # model.summary()

    # # train model
    # epochs = 1
    # history = model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=epochs
    # )

    # # visualize taining results
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']

    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

    # epochs_range = range(epochs)

    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')

    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()
