from keras import models
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import MaxPooling2D, UpSampling2D,Conv2D
from keras.layers.normalization import BatchNormalization

def segnet(nClasses, img_h=360, img_w=480):


    kernel = 3

    encoding_layers = [
        Conv2D(64, (kernel, kernel), padding='same', input_shape=(img_h, img_w, 3)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),


    ]

    autoencoder = models.Sequential()
    autoencoder.encoding_layers = encoding_layers

    for l in autoencoder.encoding_layers:
        autoencoder.add(l)

    decoding_layers = [

        UpSampling2D(),
        Conv2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(nClasses, (1, 1), padding='valid'),
        BatchNormalization(),
    ]
    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)
    width = autoencoder.output_shape[2]
    height = autoencoder.output_shape[1]
    autoencoder.add(Reshape((nClasses, height * width)))
    autoencoder.add(Permute((2, 1)))
    autoencoder.add(Activation('softmax'))



    return autoencoder, height, width

