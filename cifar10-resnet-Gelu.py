#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


#devices = tf.config.get_visible_devices()
#tf.config.set_visible_devices(devices[0:1]+devices[1:2])
#strategy = tf.distribute.MirroredStrategy()

strategy = tf.distribute.OneDeviceStrategy('gpu:1')

# load the cifar-10 data
with strategy.scope():
    (x_train, y_train),(x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    x = keras.layers.Input(shape=x_train.shape[1:])
    y = x # Placeholder
    # Linear projection - learned upsampling
    y = keras.layers.Conv2DTranspose(3,(4,4),4)(y)
    
    #y = keras.layers.Input(y.shape[1:])
    #y = x
    y = keras.layers.ZeroPadding2D(padding= ((3,3),(3,3)))(y)
    y = keras.layers.Conv2D(64,kernel_size = (7,7),strides = (2,2))(y)
    y = keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(y)
    y = keras.layers.MaxPooling2D(3,2)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("gelu")(y)
    
    y1 = keras.layers.Conv2D(64,kernel_size=(1,1),use_bias=False)(y)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(y1)
    y1 = keras.layers.Conv2D(64,3,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(256,kernel_size=(1,1))(y1)
    yy = keras.layers.Conv2D(256,kernel_size=(1,1))(y)
    y = keras.layers.Add(name = "conv2_block1_out")([y1,yy])

    y1 = keras.layers.BatchNormalization()(y)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(64,1,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(y1)
    y1 = keras.layers.Conv2D(64,3,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(256,kernel_size=(1,1))(y1)
    y = keras.layers.Add(name = "conv2_block2_out")([y1,y])


    y1 = keras.layers.BatchNormalization()(y)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(64,1,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(y1)
    y1 = keras.layers.Conv2D(64,3,2,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(256,kernel_size=(1,1))(y1)
    yy = keras.layers.MaxPooling2D(1,1)(y1)
    y = keras.layers.Add(name = "conv2_block3_out")([y1,yy])

    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("gelu")(y)
    y1 = keras.layers.Conv2D(128,1,1,use_bias=False)(y)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(y1)
    y1 = keras.layers.Conv2D(128,3,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(512,kernel_size=(1,1))(y1)
    yy = keras.layers.Conv2D(512,kernel_size=(1,1))(y)
    y = keras.layers.Add(name = "conv3_block1_out")([y1,yy])

    y1 = keras.layers.BatchNormalization()(y)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(128,1,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(y1)
    y1 = keras.layers.Conv2D(128,3,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(512,kernel_size=(1,1))(y1)
    y = keras.layers.Add(name = "conv3_block2_out")([y1,y])

    y1 = keras.layers.BatchNormalization()(y)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(128,1,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(y1)
    y1 = keras.layers.Conv2D(128,3,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(512,kernel_size=(1,1))(y1)
    y = keras.layers.Add(name = "conv3_block3_out")([y1,y])


    y1 = keras.layers.BatchNormalization()(y)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(128,1,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(y1)
    y1 = keras.layers.Conv2D(128,3,2,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(512,kernel_size=(1,1))(y1)
    yy = keras.layers.MaxPooling2D(1,1)(y1)
    y = keras.layers.Add(name = 'conv3_block4_out')([y1,yy])



    y1 = keras.layers.BatchNormalization()(y)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(256,1,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(y1)
    y1 = keras.layers.Conv2D(256,3,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(1024,kernel_size=(1,1))(y1)
    yy = keras.layers.Conv2D(1024,kernel_size=(1,1))(y)
    y = keras.layers.Add(name = 'conv4_block1_out')([y1,yy])

    y1 = keras.layers.BatchNormalization()(y)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(256,1,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(y1)
    y1 = keras.layers.Conv2D(256,3,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(1024,kernel_size=(1,1))(y1)
    y = keras.layers.Add(name = 'conv4_block2_out')([y1,y])


    y1 = keras.layers.BatchNormalization()(y)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(256,1,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(y1)
    y1 = keras.layers.Conv2D(256,3,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(1024,kernel_size=(1,1))(y1)
    y = keras.layers.Add(name = 'conv4_block3_out')([y1,y])


    y1 = keras.layers.BatchNormalization()(y)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(256,1,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(y1)
    y1 = keras.layers.Conv2D(256,3,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(1024,kernel_size=(1,1))(y1)
    y = keras.layers.Add(name = 'conv4_block4_out')([y1,y])


    y1 = keras.layers.BatchNormalization()(y)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(256,1,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(y1)
    y1 = keras.layers.Conv2D(256,3,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(1024,kernel_size=(1,1))(y1)
    y = keras.layers.Add(name = 'conv4_block5_out')([y1,y])


    y1 = keras.layers.BatchNormalization()(y)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(256,1,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(y1)
    y1 = keras.layers.Conv2D(256,3,2,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(1024,kernel_size=(1,1))(y1)
    yy = keras.layers.MaxPooling2D(1,1)(y1)
    y = keras.layers.Add(name = 'conv4_block6_out')([y1,yy])


    y1 = keras.layers.BatchNormalization()(y)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(512,1,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(y1)
    y1 = keras.layers.Conv2D(512,3,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(2048,kernel_size=(1,1))(y1)
    yy = keras.layers.Conv2D(2048,kernel_size=(1,1))(y)
    y = keras.layers.Add(name = 'conv5_block1_out')([y1,yy])


    y1 = keras.layers.BatchNormalization()(y)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(512,1,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(y1)
    y1 = keras.layers.Conv2D(512,3,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(2048,kernel_size=(1,1))(y1)
    y = keras.layers.Add(name = 'conv5_block2_out')([y1,y])


    y1 = keras.layers.BatchNormalization()(y)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(512,1,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.ZeroPadding2D(padding=((1,1),(1,1)))(y1)
    y1 = keras.layers.Conv2D(512,3,1,use_bias=False)(y1)
    y1 = keras.layers.BatchNormalization()(y1)
    y1 = keras.layers.Activation("gelu")(y1)
    y1 = keras.layers.Conv2D(2048,kernel_size=(1,1))(y1)
    y = keras.layers.Add(name = 'conv5_block3_out')([y1,y])


    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("gelu")(y)
    y = keras.layers.GlobalAveragePooling2D()(y)
    y = keras.layers.Dense(10, activation="softmax")(y) # Low-dimensional Projection

    model = keras.Model(x,y)
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001),
                   loss = keras.losses.SparseCategoricalCrossentropy(),
                   metrics = [keras.metrics.SparseCategoricalAccuracy()])


print(model.summary())


with strategy.scope():
    data_generator = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True)
    dg_trainer = data_generator.flow(x_train, y_train, batch_size=128)
    history = model.fit(
        dg_trainer,
        validation_data=(x_test,y_test),
        epochs=50,
        verbose=1)

print("Validation accuracy:",*["%.8f"%(x) for x in
                               history.history['val_sparse_categorical_accuracy']])
 

