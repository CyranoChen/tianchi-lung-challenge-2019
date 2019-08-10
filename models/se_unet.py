# coding: utf-8
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Reshape, BatchNormalization, concatenate, Lambda, add, multiply
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, Cropping2D
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K


'''
Hyper-parameters
'''

# network structure
FILTER_NUM = 32 # number of basic filters for the first layer
FILTER_SIZE = 3 # size of the convolutional filter
SE_RATIO = 1  # reduction ratio of SE block



def expend_as(tensor, rep):
     return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)


def double_conv_layer(ip, filter_size, size, dropout, batch_norm=False, weight_decay=1e-4):
    '''
    construction of a double convolutional layer using
    SAME padding
    RELU nonlinear activation function
    :param x: input
    :param filter_size: size of convolutional filter
    :param size: number of filters
    :param dropout: FLAG & RATE of dropout.
            if < 0 dropout cancelled, if > 0 set as the rate
    :param batch_norm: flag of if batch_norm used,
            if True batch normalization
    :return: output of a double convolutional layer
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
        
    if batch_norm:
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
        x = Activation('relu')(x)
        x = Conv2D(size, (filter_size, filter_size), kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    else:
        x = Conv2D(size, (filter_size, filter_size), kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(ip)
        x = Activation('relu')(x)

    if batch_norm:
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = Conv2D(size, (filter_size, filter_size), kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    else:
        x = Conv2D(size, (filter_size, filter_size), kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)

    if dropout > 0:
        x = Dropout(dropout)(x)
        
    shortcut = x

    if batch_norm is True:
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = Conv2D(size, kernel_size=(1, 1), kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    else:
        x = Conv2D(size, kernel_size=(1, 1), kernel_initializer='he_normal', padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)

    res = add([x, shortcut])
    return res


def SE_block(x, out_dim, ratio, name, batch_norm=False):
    """
    self attention squeeze-excitation block, attention mechanism on channel dimension
    :param x: input feature map
    :return: attention weighted on channel dimension feature map
    """
    # Squeeze: global average pooling
    x_s = GlobalAveragePooling2D(data_format=None)(x)
    # Excitation: bottom-up top-down FCs
    if batch_norm:
        x_s = BatchNormalization()(x_s)
    x_e = Dense(units=out_dim//ratio)(x_s)
    x_e = Activation('relu')(x_e)
    if batch_norm:
        x_e = BatchNormalization()(x_e)
    x_e = Dense(units=out_dim)(x_e)
    x_e = Activation('sigmoid')(x_e)
    x_e = Reshape((1, 1, out_dim), name=name+'channel_weight')(x_e)
    result = multiply([x, x_e])
    return result



def gating_signal(ip, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :param input:   down-dim feature map
    :param out_size:output channel number
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    if batch_norm:
        x = BatchNormalization()(ip)
        x = Activation('relu')(x)
        x = Conv2D(out_size, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
    else:
        x = Conv2D(out_size, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False)(ip)
        x = Activation('relu')(x)

    return x

def attention_block(x, gating, inter_shape, name):
    """
    self gated attention, attention mechanism on spatial dimension
    :param x: input feature map
    :param gating: gate signal, feature map from the lower layer
    :param inter_shape: intermedium channle numer
    :param name: name of attention layer, for output
    :return: attention weighted on spatial dimension feature map
    """

    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), kernel_initializer='he_normal', padding='same', use_bias=False)(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False)(gating)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 kernel_initializer='he_normal', padding='same', use_bias=False)(phi_g)  # 16
    # upsample_g = UpSampling2D(size=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
    #                                  data_format="channels_last")(phi_g)

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False)(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]),
                                       name=name+'_weight')(sigmoid_xg)  # 32

    upsample_psi = expend_as(upsample_psi, shape_x[3])


    y = multiply([upsample_psi, x])

    result_bn = BatchNormalization()(y)
    result = Conv2D(shape_x[3], (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False)(result_bn)

    return result


def Attention_ResUNet_PA(input_shape, dropout_rate=0.0, batch_norm=True):
    '''
    Residual UNet construction, with attention gate
    convolution: 3*3 SAME padding
    pooling: 2*2 VALID padding
    upsampling: 3*3 VALID padding
    final convolution: 1*1
    :param dropout_rate: FLAG & RATE of dropout.
            if < 0 dropout cancelled, if > 0 set as the rate
    :param batch_norm: flag of if batch_norm used,
            if True batch normalization
    :return: model
    '''
    # input data
    # dimension of the image depth
    inputs = Input(input_shape)
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Downsampling layers
    # DownRes 1, double residual convolution + pooling
    conv_128 = double_conv_layer(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = double_conv_layer(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = double_conv_layer(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = double_conv_layer(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = double_conv_layer(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers

    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    # channel attention block
    se_conv_16 = SE_block(conv_16, out_dim=8*FILTER_NUM, ratio=SE_RATIO, name='att_16')
    # spatial attention block
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = attention_block(se_conv_16, gating_16, 8*FILTER_NUM, name='att_16')
    # attention re-weight & concatenate
    up_16 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, att_16], axis=concat_axis)
    up_conv_16 = double_conv_layer(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)

    # UpRes 7
    # channel attention block
    se_conv_32 = SE_block(conv_32, out_dim=4*FILTER_NUM, ratio=SE_RATIO, name='att_32')
    # spatial attention block
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = attention_block(se_conv_32, gating_32, 4*FILTER_NUM, name='att_32')
    # attention re-weight & concatenate
    up_32 = UpSampling2D(size=(2, 2), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, att_32], axis=concat_axis)
    up_conv_32 = double_conv_layer(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)

    # UpRes 8
    # channel attention block
    se_conv_64 = SE_block(conv_64, out_dim=2*FILTER_NUM, ratio=SE_RATIO, name='att_64')
    # spatial attention block
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = attention_block(se_conv_64, gating_64, 2*FILTER_NUM, name='att_64')
    # attention re-weight & concatenate
    up_64 = UpSampling2D(size=(2, 2), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, att_64], axis=concat_axis)
    up_conv_64 = double_conv_layer(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)

    # UpRes 9
    # channel attention block
    se_conv_128 = SE_block(conv_128, out_dim=FILTER_NUM, ratio=SE_RATIO, name='att_128')
    # spatial attention block
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    # attention re-weight & concatenate
    att_128 = attention_block(se_conv_128, gating_128, FILTER_NUM, name='att_128')
    up_128 = UpSampling2D(size=(2, 2), data_format="channels_last")(up_conv_64)
    up_128 = concatenate([up_128, att_128], axis=concat_axis)
    up_conv_128 = double_conv_layer(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    # valid padding
    # batch normalization
    # sigmoid nonlinear activation
    conv_final = Conv2D(1, kernel_size=(1,1), kernel_initializer='he_normal', padding='same', use_bias=False)(up_conv_128)
    conv_final = Activation('sigmoid')(conv_final)

    # Model integration
    model = Model(inputs, conv_final, name="AttentionSEResUNet")
    return model



