from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Reshape, BatchNormalization, concatenate
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, Cropping2D
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K


def __conv_block(ip, nb_filter, batch_normalization=False, dropout_rate=None, weight_decay=1e-4, padding='same'):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    if batch_normalization:
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding=padding, use_bias=False)(x)
    else:
        x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding=padding, use_bias=False)(ip)
        x = Activation('relu')(x)
        
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
        
    print(x.shape)

    return x


def __get_cropping(tensor1, tensor2):
    assert tensor1.shape[1] >= tensor2.shape[1]
    assert tensor1.shape[2] >= tensor2.shape[2]
    
    rows = int(tensor1.shape[1]) - int(tensor2.shape[1])
    cols = int(tensor1.shape[2]) - int(tensor2.shape[2])
    
    if (rows%2==1) and (cols%2==1):
        return ((rows // 2 + 1, rows // 2), (cols // 2 + 1, cols // 2))
    else:
        return ((rows // 2, rows // 2), (cols // 2, cols // 2))


def UNET(input_shape = (572, 572, 1), batch_normalization=True, dropout_rate=None, weight_decay=1e-4, padding='same'):
    inputs = Input(input_shape)
    conv1 = __conv_block(inputs, 64, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    conv1 = __conv_block(conv1, 64, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='valid')(conv1)
    
    conv2 = __conv_block(pool1, 128, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    conv2 = __conv_block(conv2, 128, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = __conv_block(pool2, 256, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    conv3 = __conv_block(conv3, 256, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = __conv_block(pool3, 512, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    conv4 = __conv_block(conv4, 512, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = __conv_block(pool4, 1024, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    conv5 = __conv_block(conv5, 1024, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    up6 = UpSampling2D(size=(2, 2))(conv5)
    merge6 = concatenate([Cropping2D(cropping=__get_cropping(conv4, up6))(conv4), up6], axis = 3)
    
    conv6 = __conv_block(merge6, 512, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    conv6 = __conv_block(conv6, 512, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    up7 = UpSampling2D(size=(2, 2))(conv6)
    merge7 = concatenate([Cropping2D(cropping=__get_cropping(conv3, up7))(conv3), up7], axis = 3)
    
    conv7 = __conv_block(merge7, 256, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    conv7 = __conv_block(conv7, 256, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    up8 = UpSampling2D(size=(2, 2))(conv7)
    merge8 = concatenate([Cropping2D(cropping=__get_cropping(conv2, up8))(conv2), up8], axis = 3)
    
    conv8 = __conv_block(merge8, 128, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    conv8 = __conv_block(conv8, 128, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    up9 = UpSampling2D(size=(2, 2))(conv8)
    merge9 = concatenate([Cropping2D(cropping=__get_cropping(conv1, up9))(conv1), up9], axis = 3)
    
    conv9 = __conv_block(merge9, 64, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    conv9 = __conv_block(conv9, 64, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    outputs = Conv2D(1, (1,1), activation = 'sigmoid', kernel_initializer='he_normal', padding='same', use_bias=False)(conv9)

    model = Model(inputs = inputs, outputs = outputs, name='unet')

    return model

