import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Reshape, BatchNormalization, concatenate
from tensorflow.keras.layers import Conv3D, UpSampling3D, Cropping3D
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

from se_densenet_3d import SEDenseNet


def __conv_block(ip, nb_filter, batch_normalization=False, dropout_rate=None, weight_decay=1e-4, padding='same'):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    if batch_normalization:
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
        x = Activation('relu')(x)

        x = Conv3D(nb_filter, (3, 3, 3), kernel_initializer='he_normal', padding=padding, use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    else:
        x = Conv3D(nb_filter, (3, 3, 3), kernel_initializer='he_normal', padding=padding, use_bias=False, kernel_regularizer=l2(weight_decay))(ip)
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
    
    
def __get_backbone(input_shape = (572, 572, 1)):
    return SEDenseNet(input_shape=input_shape,
                        depth=40,
                        nb_dense_block=3,
                        growth_rate=12,
                        nb_filter=-1,
                        nb_layers_per_block=-1,
                        bottleneck=True,
                        reduction=0.5,
                        dropout_rate=0.2,
                        weight_decay=1e-4,
                        subsample_initial_block=True,
                        include_top=True,
                        weights=None,
                        input_tensor=None,
                        classes=5,
                        activation='softmax')


def __freeze_model(model):
    """model all layers non trainable, excluding BatchNormalization layers"""
    for layer in model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    return

def UNET(input_shape = (64, 64, 64, 1), model_path=None, batch_normalization=True, dropout_rate=None, weight_decay=1e-4):
    if model_path is not None and os.path.exists(model_path):
        backbone = load_model(model_path)
        __freeze_model(backbone) # freeze the backbone weights
    else:
        backbone = __get_backbone(input_shape)        
    
    # backbone.summary()
       
    inputs = backbone.input 
    x = backbone.layers[-3].output
    out1 = backbone.layers[-115].output # multiply_2 (Multiply) 
    out2 = backbone.layers[105].output # multiply (Multiply) 
    out3 = backbone.layers[3].output # Activation
       
    print('backbone output:', x.shape)
    
    print('backbone out1:', backbone.layers[-115], out1.shape)
    print('backbone out2:', backbone.layers[105], out2.shape)
    print('backbone out3:', backbone.layers[3], out3.shape)
    
#     conv1 = __conv_block(inputs, 64, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
#     conv1 = __conv_block(conv1, 64, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
#     pool1 = MaxPooling2D(pool_size=(2, 2), padding='valid')(conv1)
    
#     conv2 = __conv_block(pool1, 128, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
#     conv2 = __conv_block(conv2, 128, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
#     conv3 = __conv_block(pool2, 256, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
#     conv3 = __conv_block(conv3, 256, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
#     conv4 = __conv_block(pool3, 512, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
#     conv4 = __conv_block(conv4, 512, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

#     conv5 = __conv_block(x, 1024, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
#     conv5 = __conv_block(conv5, 1024, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay, padding=padding)
    up6 = UpSampling3D(size=(2, 2, 2))(x)
    print('up6:', up6.shape)
    merge6 = concatenate([out1, up6], axis = -1)   
    print('merge6:', merge6.shape)
    
    conv6 = __conv_block(merge6, 168, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay)
    conv6 = __conv_block(conv6, 168, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay)
    up7 = UpSampling3D(size=(2, 2, 2))(conv6)
    print('u7:', up7.shape)
    merge7 = concatenate([out2, up7], axis = -1)
    print('merge7:', merge7.shape)
    
    conv7 = __conv_block(merge7, 32, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay)
    conv7 = __conv_block(conv7, 32, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay)
    up8 = UpSampling3D(size=(2, 2, 2))(conv7)
    print('u8:', up8.shape)
    merge8 = concatenate([out3, up8], axis = -1)
    print('merge8:', merge8.shape)
    
    conv8 = __conv_block(merge8, 16, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay)
    conv8 = __conv_block(conv8, 16, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay)
    up9 = UpSampling3D(size=(2, 2, 2))(conv8)
    print('u9:', up9.shape)
    merge9 = concatenate([inputs, up9], axis = -1)
    print('merge9:', merge9.shape)
    
    conv9 = __conv_block(merge9, 16, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay)
    conv9 = __conv_block(conv9, 16, batch_normalization=batch_normalization, dropout_rate=dropout_rate, weight_decay=weight_decay)
    outputs = Conv3D(1, (1,1,1), activation = 'sigmoid', kernel_initializer='he_normal', padding='same', use_bias=False)(conv9)

    model = Model(inputs = inputs, outputs = outputs, name='unet3d')

    return model

