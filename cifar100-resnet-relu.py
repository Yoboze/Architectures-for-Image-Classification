#!/usr/bin/env python3

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
#import matplotlib.pyplot as plt

strategy = tf.distribute.OneDeviceStrategy('gpu:1')

# load the data cifar-100
with strategy.scope():
     (x_train, y_train),(x_test, y_test) = keras.datasets.cifar100.load_data()
     x_train, x_test = x_train / 255.0, x_test / 255.0

# untrained ResNet
with strategy.scope():
     x = keras.layers.Input(shape = x_train.shape[1:])
     y = x # Placeholder
     
     # Linear projection - learned upsampling
     y = keras.layers.Conv2DTranspose(3,(4,4),4)(y)
     
     # Functionalized ResNet with fresh weights
     resnet = keras.applications.ResNet50V2(weights = None,
                                            classes = 100,
                                            input_shape = y.shape[1:])

     y = resnet(y)

     model = keras.Model(x,y)
     model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001),
                   loss = keras.losses.SparseCategoricalCrossentropy(),
                   metrics = [keras.metrics.SparseCategoricalAccuracy()])

model.summary()

print(model.summary())


# a data augmentation pipeline to the training phase

with strategy.scope():
     data_generator = keras.preprocessing.image.ImageDataGenerator(
         width_shift_range = 0.1,
         height_shift_range = 0.1,
         rotation_range = 10,
         zoom_range = 0.1,
         horizontal_flip = True)
     dg_trainer = data_generator.flow(x_train, y_train, batch_size=256)
     history = model.fit(
         dg_trainer,
         validation_data = (x_test, y_test),
         epochs = 50,
         verbose = 1)

 
print("Validation accuracy:",*["%.8f"%(x) for x in
                              history.history['val_sparse_categorical_accuracy']])


     
