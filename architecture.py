

from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Layer, Activation, Reshape, Permute


def set_architecture(n_classes, img_shape):
    model = Sequential()
    
    # Encoding
    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=img_shape, data_format='channels_first'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')) 
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')) 
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')) 
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first'))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first'))
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format='channels_first' )(x)

    # Block 5
    #x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format='channels_first' )(x)
    #x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format='channels_first' )(x)
    #x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format='channels_first' )(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format='channels_first' )(x)
    #f5 = x

    #x = Flatten(name='flatten')(x)
    #x = Dense(4096, activation='relu', name='fc1')(x)
    #x = Dense(4096, activation='relu', name='fc2')(x)
    #x = Dense( 1024 , activation='softmax', name='predictions')(x)

    #vgg  = Model(  img_input , x  )
    #vgg.load_weights(VGG_Weights_path)

    #levels = [f1 , f2 , f3 , f4 , f5 ]

    #o = levels[ vgg_level ]
    # Decoding
    model.add(ZeroPadding2D( (1,1) , data_format='channels_first' ))
    model.add(Conv2D(512, (3, 3), padding='valid', data_format='channels_first'))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(512, (3, 3), padding='same', data_format='channels_first')) # added


    model.add(UpSampling2D( (2,2), data_format='channels_first'))
    model.add(ZeroPadding2D( (1,1), data_format='channels_first'))
    model.add(Conv2D( 256, (3, 3), padding='valid', data_format='channels_first'))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(256, (3, 3), padding='same', data_format='channels_first')) # added


    model.add(UpSampling2D( (2,2), data_format='channels_first'))
    model.add(ZeroPadding2D( (1,1), data_format='channels_first'))
    model.add(Conv2D( 128, (3, 3), padding='valid', data_format='channels_first'))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(128, (3, 3), padding='same', data_format='channels_first')) # added

    model.add(UpSampling2D( (2,2), data_format='channels_first'))
    model.add(ZeroPadding2D( (1,1), data_format='channels_first'))
    model.add(Conv2D( 64, (3, 3), padding='valid', data_format='channels_first'))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')) # added
    
    model.add(Conv2D( n_classes , (3, 3) , padding='same', data_format='channels_first' )) 
    #o_shape = Model(img_input , o ).output_shape
    o_shape = model.output_shape
    outputHeight = o_shape[2]
    outputWidth = o_shape[3]

    model.add(Reshape((-1, outputHeight*outputWidth)))
    model.add(Permute((2, 1)))
    model.add(Activation('softmax'))


    #model = Model( img_input , o )
    #model.outputWidth = outputWidth
    #model.outputHeight = outputHeight

    #model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    return model




if __name__ == '__main__':
    n_classes = 10
    img_channels = 1 
    img_h = 256
    img_w = 256
    
    input_shape = (img_channels, img_h, img_w)  

    model = set_architecture(n_classes, input_shape)

