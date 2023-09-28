import tensorflow as tf
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.activations import gelu
from keras.layers import MultiHeadAttention
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, LayerNormalization, ELU
from keras.initializers import glorot_uniform, RandomNormal
from keras import layers
from keras.layers import Reshape
from keras.layers import Rescaling


from keras.models import Model
from keras.layers import Dense
from keras.layers import Input

from keras.layers import Activation
from keras.layers import ELU
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPooling1D, MaxPooling2D
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.core import Flatten
from keras.initializers import glorot_uniform
from keras.initializers import RandomNormal
import tensorflow as tf
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, ELU, BatchNormalization


'''
def create_model(input_shape=None, emb_size=None, model_name=''):
    input_data = Input(shape=input_shape)

    # First CNN block
    cnn_output = Conv1D(filters=8, kernel_size=1, padding='same')(input_data)
    cnn_output = ELU(alpha=1.0)(cnn_output)
    cnn_output = MaxPooling1D(pool_size=2)(cnn_output)
    cnn_output = Dropout(0.1)(cnn_output)  # Adding Dropout

    # Additional CNN blocks for decreased complexity
    num_cnn_blocks = 3  # You can adjust this number based on your needs

    for i in range(1, num_cnn_blocks + 1):
        cnn_output = Conv1D(filters=8 * 2**i, kernel_size=3, padding='same')(cnn_output)
        cnn_output = ELU(alpha=1.0)(cnn_output)
        cnn_output = MaxPooling1D(pool_size=2)(cnn_output)
        cnn_output = Dropout(0.1)(cnn_output)  # Adding Dropout

    # Transformer layers
    num_layers = 2
    num_heads = 4
    hidden_size = 256

    transformer_input = Rescaling(1.0 / emb_size)(cnn_output)  # Rescale input for compatibility with the transformer

    encoder_output = transformer_input
    for _ in range(num_layers):
        encoder_output = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size // num_heads)(encoder_output, encoder_output[:, :2, :])
        encoder_output = Dropout(0.1)(encoder_output)  # Adding Dropout
        encoder_output = LayerNormalization(epsilon=1e-6)(encoder_output)
        encoder_output = layers.Activation('relu')(encoder_output)  # Adding ReLU activation

    output = GlobalAveragePooling1D()(encoder_output[:, :2, :])
    output = Dense(emb_size, activation='sigmoid', kernel_initializer=glorot_uniform(seed=0))(output)

    model = Model(inputs=input_data, outputs=output, name=model_name)
    return model
'''


'''def create_model(input_shape=None, emb_size=None, model_name=''):
    input_data = Input(shape=input_shape)

    # CNN layer
    cnn_output = Conv1D(filters=8, kernel_size=1, padding='same')(input_data)
    cnn_output = ELU(alpha=1.0)(cnn_output)
    cnn_output = MaxPooling1D(pool_size=2)(cnn_output)

    # Transformer layers
    num_layers = 2
    num_heads = 4
    hidden_size = 256

    transformer_input = Rescaling(1.0 / emb_size)(cnn_output)  # Rescale input for compatibility with transformer

    encoder_output = transformer_input
    for _ in range(num_layers):
        encoder_output = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size // num_heads)(encoder_output, encoder_output[:, :2, :])
        encoder_output = Dropout(0.1)(encoder_output)
        encoder_output = LayerNormalization(epsilon=1e-6)(encoder_output)

    output = GlobalAveragePooling1D()(encoder_output[:, :2, :])
    output = Dense(emb_size, activation='sigmoid', kernel_initializer=glorot_uniform(seed=0))(output)

    model = Model(inputs=input_data, outputs=output, name=model_name)
    return model
'''


'''def create_model(input_shape=None, emb_size=None, model_name=''):
    input_data = Input(shape=input_shape)
    
    num_layers = 2
    num_heads = 4
    hidden_size = 256

    encoder_output = input_data
    for _ in range(num_layers):
        encoder_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size)(encoder_output, encoder_output)
        encoder_output = Dropout(0.1)(encoder_output)
        encoder_output = LayerNormalization(epsilon=1e-6)(encoder_output)
        #encoder_output = Dense(hidden_size, activation='relu')(encoder_output)

    try:
        output = Dense(emb_size, activation='sigmoid', kernel_initializer=glorot_uniform(seed=0))(encoder_output)
    except:
        output = Dense(emb_size, activation='sigmoid', kernel_initializer=RandomNormal(stddev=0.01, mean=0.0))(encoder_output)

    model = Model(inputs=input_data, outputs=output, name=model_name)
    return model
'''

import tensorflow as tf
from keras.layers import Input, Conv1D, ELU, MaxPooling1D, Dropout, LayerNormalization, Dense
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.layers import GlobalAveragePooling1D, Reshape, Rescaling
import numpy as np

def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    d_k = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    
    return output, attention_weights

def multihead_attention(query, key, value, num_heads, mask):
    d_model = tf.shape(key)[-1]
    depth = d_model // num_heads
    
    query = tf.reshape(query, (-1, tf.shape(query)[1], num_heads, depth))
    key = tf.reshape(key, (-1, tf.shape(key)[1], num_heads, depth))
    value = tf.reshape(value, (-1, tf.shape(value)[1], num_heads, depth))
    
    query = tf.transpose(query, perm=[0, 2, 1, 3])
    key = tf.transpose(key, perm=[0, 2, 1, 3])
    value = tf.transpose(value, perm=[0, 2, 1, 3])
    
    output, attention_weights = scaled_dot_product_attention(query, key, value, mask)
    
    output = tf.transpose(output, perm=[0, 2, 1, 3])
    output = tf.reshape(output, (-1, tf.shape(output)[2], d_model))
    
    return output, attention_weights


def create_model(input_shape=None, emb_size=None, model_name=''):
    # -----------------Entry flow -----------------
    #input_data = Input(shape=input_shape)
    #batch_size = 32
    input_data = Input(shape=input_shape)

    filter_num = ['None', 32, 64, 128, 256, 256]
    kernel_size = ['None', 8, 8, 8, 8, 8]
    conv_stride_size = ['None', 1, 1, 1, 1, 1]
    pool_stride_size = ['None', 4, 4, 4, 4, 4]
    pool_size = ['None', 8, 8, 8, 8, 8]
    
    
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
    model = Dropout(0.1, name='block4_dropout'+'_'+model_name)(model)
   
    model = Conv1D(filters=filter_num[5], kernel_size=kernel_size[5],
                   strides=conv_stride_size[5], padding='same', name='block5_conv2'+'_'+model_name)(model)
    model = Activation('relu', name='block5_act2'+'_'+model_name)(model)
    model = MaxPooling1D(pool_size=pool_size[5], strides=pool_stride_size[5],
                         padding='same', name='block5_pool'+'_'+model_name)(model)
    cnn_output = Dropout(0.1, name='block5_dropout'+'_'+model_name)(model)
    
    #cnn_output = Flatten()(model)
 
    # Transformer layers
    num_layers = 4
    num_heads = 2
    hidden_size = 256

    transformer_input = Rescaling(1.0 / emb_size)(model)  # Rescale input for compatibility with transformer

    encoder_output = transformer_input
    for _ in range(num_layers):
        '''batch_size = tf.shape(encoder_output)[0]
        query = encoder_output[:, :batch_size, :]  # Truncate the query tensor to match batch size
        value = encoder_output[:, :batch_size, :]
        encoder_output = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size // num_heads)(query, value)#(encoder_output, encoder_output[:, :2, :])
        encoder_output = Dropout(0.1)(encoder_output)
        encoder_output = LayerNormalization(epsilon=1e-4)(encoder_output)'''

        mask = None  # You can add a mask if needed
        encoder_output, _ = multihead_attention(encoder_output, encoder_output, encoder_output, num_heads, mask)
        encoder_output = Dropout(0.1)(encoder_output)
        encoder_output = LayerNormalization(epsilon=1e-4)(encoder_output)

    # Window Partitioning Layer
    target_product = 74240
    window_size = 11  # Define the window size based on your requirements
    num_windows = input_shape[0] // window_size
    window_size = target_product // (num_windows * emb_size)
    window_outputs = []

    for i in range(num_windows):
        start = i * window_size
        end = (i + 1) * window_size
        window_output = encoder_output[start:end, :, :]  # Slice the encoder output
        window_outputs.append(window_output)

    # Combine window outputs
    window_outputs = tf.stack(window_outputs)
    window_partition_output = Reshape((num_windows * window_size, emb_size))(window_outputs)
    #window_partition_output = Reshape((num_windows, window_size, emb_size))(window_outputs)
    #window_partition_output = Reshape((num_windows, window_size, emb_size))(window_partition_output)
    #flatten_output = Flatten()(window_partition_output)

    #output = GlobalAveragePooling1D()(encoder_output[:, :2, :])
    output = GlobalAveragePooling1D()(window_partition_output)
    output = Dense(emb_size, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(output)

    model = Model(inputs=input_data, outputs=output, name=model_name)
    return model

'''
def create_model(input_shape=None, emb_size=None, model_name=''):
    # -----------------Entry flow -----------------
    input_data = Input(shape=input_shape)

    filter_num = ['None', 32, 64, 128, 256, 256]
    kernel_size = ['None', 8, 8, 8, 8, 8]
    conv_stride_size = ['None', 1, 1, 1, 1, 1]
    pool_stride_size = ['None', 4, 4, 4, 4, 4]
    pool_size = ['None', 8, 8, 8, 8, 8]
    

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
    return shared_conv2

def create_model_2d(input_shape=None, emb_size=None):
    input_data = Input(shape=input_shape) # (None, 2, 371, 1)
    # OOM when allocating tensor with shape[400,750,2,342]
    model = Conv2D(64, kernel_size=(2, 30), strides=(2, 1), padding='valid', activation='relu', input_shape=input_shape, kernel_initializer=RandomNormal(stddev=0.01))(input_data) #(None, 2, 342, 750)
    model = MaxPooling2D(pool_size=(1, 5), strides=(1, 1), padding='valid')(model) #(None, 2, 338, 2000)
    model = Conv2D(32, kernel_size=(1, 10), strides=(1, 1), padding='valid', activation='relu', kernel_initializer=RandomNormal(stddev=0.01))(model) # (None, 2, 329, 1000)
    model = MaxPooling2D(pool_size=(1, 5), strides=(1, 1), padding='valid')(model) #(None, 2, 325, 1000)
    print(model._keras_shape) # (None, 2, 325, 1000)
    model = Flatten()(model)
    # model1_out = Reshape((-1,))(model1_out)
    print(model._keras_shape) # (None, 650000)
    # Issue: OOM when allocating tensor with shape[650000,3000]
    model = Dense(1024, activation='relu', kernel_initializer=RandomNormal(stddev=0.01, mean=0.0))(model)
    model = Dropout(rate=0.6)(model)
    model = Dense(800, activation='relu', kernel_initializer=RandomNormal(stddev=0.01, mean=0.0))(model)
    model = Dropout(rate=0.6)(model)
    model = Dense(100, activation='relu', kernel_initializer=RandomNormal(stddev=0.01, mean=0.0))(model)
    model = Dropout(rate=0.6)(model)
    model = Dense(emb_size, activation='linear', kernel_initializer=RandomNormal(stddev=0.01, mean=0.0))(model)
    shared_conv2 = Model(inputs=input_data, outputs=model)
    return shared_conv2
'''