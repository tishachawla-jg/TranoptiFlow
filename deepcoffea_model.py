# The model is the DF model by Sirinam et al
import pickle
import numpy as np
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Activation
from keras.layers import ELU
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPooling1D, MaxPooling2D, GlobalAveragePooling1D
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.initializers import glorot_uniform
from keras.initializers import RandomNormal

from keras.layers import MultiHeadAttention, LayerNormalization

def transformer_block(x, num_heads, key_dim, ff_dim):
    # Multi-Head Attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)

    # Feed Forward Network
    ff_output = Dense(ff_dim, activation="relu")(x)
    ff_output = Dense(x.shape[-1])(ff_output)
    return LayerNormalization(epsilon=1e-6)(x + ff_output)

def create_model_e(input_shape, emb_size=None, model_name=''):
    # -----------------Entry flow -----------------
    input_data = Input(shape=input_shape)

    filter_num = ['None', 32, 64, 128, 256]
    kernel_size = ['None', 8, 8, 8, 8]
    conv_stride_size = ['None', 1, 1, 1, 1]
    pool_stride_size = ['None', 4, 4, 4, 4]
    pool_size = ['None', 8, 8, 8, 8]
    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv1'+'_'+model_name)(input_data)
    model = ELU(alpha=1.0, name='block1_adv_act1'+'_'+model_name)(model)
    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv2'+'_'+model_name)(model)
    model = ELU(alpha=1.0, name='block1_adv_act2'+'_'+model_name)(model)
    model = MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                         padding='same', name='block1_pool'+'_'+model_name)(model)
    model = Dropout(0.1, name='block1_dropout'+'_'+model_name)(model)

    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv1'+'_'+model_name)(model)
    model = Activation('relu', name='block2_act1'+'_'+model_name)(model)
    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv2'+'_'+model_name)(model)
    model = Activation('relu', name='block2_act2'+'_'+model_name)(model)
    model = MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                         padding='same', name='block2_pool'+'_'+model_name)(model)
    model = Dropout(0.1, name='block2_dropout'+'_'+model_name)(model)

    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same', name='block3_conv1'+'_'+model_name)(model)
    model = Activation('relu', name='block3_act1'+'_'+model_name)(model)
    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same', name='block3_conv2'+'_'+model_name)(model)
    model = Activation('relu', name='block3_act2'+'_'+model_name)(model)
    model = MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                         padding='same', name='block3_pool'+'_'+model_name)(model)
    model = Dropout(0.1, name='block3_dropout'+'_'+model_name)(model)

    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same', name='block4_conv1'+'_'+model_name)(model)
    model = Activation('relu', name='block4_act1'+'_'+model_name)(model)
    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same', name='block4_conv2'+'_'+model_name)(model)
    model = Activation('relu', name='block4_act2'+'_'+model_name)(model)
    model = MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                         padding='same', name='block4_pool'+'_'+model_name)(model)

    
    output = Flatten()(model)

    dense_layer = Dense(emb_size, name='FeaturesVec'+'_'+model_name)(output)

    shared_conv2 = Model(inputs=input_data, outputs=dense_layer, name=model_name)

    print('SHAPE OF MODEL')
    print(shared_conv2.summary())
    return shared_conv2


def create_model(input_shape=None, emb_size=None, model_name=''):
    # -----------------Entry flow -----------------
    input_data = Input(shape=input_shape)

    filter_num = ['None', 32, 64, 128, 256]
    kernel_size = ['None', 8, 8, 8, 8]
    conv_stride_size = ['None', 1, 1, 1, 1]
    pool_stride_size = ['None', 4, 4, 4, 4]
    pool_size = ['None', 8, 8, 8, 8]
    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv1'+'_'+model_name)(input_data)
    model = ELU(alpha=1.0, name='block1_adv_act1'+'_'+model_name)(model)
    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv2'+'_'+model_name)(model)
    model = ELU(alpha=1.0, name='block1_adv_act2'+'_'+model_name)(model)
    model = MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                         padding='same', name='block1_pool'+'_'+model_name)(model)
    model = Dropout(0.1, name='block1_dropout'+'_'+model_name)(model)

    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv1'+'_'+model_name)(model)
    model = Activation('relu', name='block2_act1'+'_'+model_name)(model)
    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv2'+'_'+model_name)(model)
    model = Activation('relu', name='block2_act2'+'_'+model_name)(model)
    model = MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                         padding='same', name='block2_pool'+'_'+model_name)(model)
    model = Dropout(0.1, name='block2_dropout'+'_'+model_name)(model)

    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same', name='block3_conv1'+'_'+model_name)(model)
    model = Activation('relu', name='block3_act1'+'_'+model_name)(model)
    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same', name='block3_conv2'+'_'+model_name)(model)
    model = Activation('relu', name='block3_act2'+'_'+model_name)(model)
    model = MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                         padding='same', name='block3_pool'+'_'+model_name)(model)
    model = Dropout(0.1, name='block3_dropout'+'_'+model_name)(model)

    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same', name='block4_conv1'+'_'+model_name)(model)
    model = Activation('relu', name='block4_act1'+'_'+model_name)(model)
    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same', name='block4_conv2'+'_'+model_name)(model)
    model = Activation('relu', name='block4_act2'+'_'+model_name)(model)
    model = MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                         padding='same', name='block4_pool'+'_'+model_name)(model)

    #output = Flatten()(model)

    #model = GlobalAveragePooling1D(name='cnn_global_avg_pool'+'_'+model_name)(model)
    model = transformer_block(model, num_heads=16, key_dim=64, ff_dim=128)
    #new line
    # Flatten and Dense layer
    
    output = Flatten()(model)

    dense_layer = Dense(emb_size, name='FeaturesVec'+'_'+model_name)(output)

    shared_conv2 = Model(inputs=input_data, outputs=dense_layer, name=model_name)

    print('SHAPE OF MODEL after trans')
    print(shared_conv2.summary())
    return shared_conv2


