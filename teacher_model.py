# !/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function
import tensorflow.keras

from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Convolution2D
from tensorflow.keras.callbacks import ModelCheckpoint
import os

import tensorflow as tf

from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

batch_size = 128
num_classes = 10
epochs = 300

os.makedirs('./teacher_models/', exist_ok=True)

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


input_layer = Input(x_train.shape[1:])
x = Convolution2D(64, (3, 3), padding='same')(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = Convolution2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)

x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = Dropout(0.3)(x)

x = Convolution2D(128, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = Convolution2D(128, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)

x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = Dropout(0.3)(x)

x = Convolution2D(256, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = Convolution2D(256, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = Convolution2D(256, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)

x = MaxPooling2D((2, 2), strides=(2, 2))(x)
x = Dropout(0.3)(x)

x = Flatten()(x)
x = Dense(512, activation=None)(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)

logits = Dense(num_classes, activation=None)(x)
output = Activation('softmax')(logits)

opt = tensorflow.keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model = Model(input_layer, output)
model.summary()
# plot_model(model, show_shapes=True, to_file='teacher_model.png')
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['acc'])


print('Using real-time data augmentation.')

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.1,  # set range for random shear
    zoom_range=0.2,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None)

datagen.fit(x_train)

callbacks = [ModelCheckpoint(filepath="./teacher_models/teacher_model_epoch_{epoch:02d}-val_acc_{val_acc}.hdf5")]
model.fit_generator(datagen.flow(x_train, y_train,
                                 batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    workers=4, callbacks=callbacks)
