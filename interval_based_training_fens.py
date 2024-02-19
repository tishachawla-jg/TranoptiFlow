from __future__ import print_function

try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle
from tensorflow.python.tools import module_util as _module_util
#from tensorflow.keras.callbacks import LambdaCallbacky
from tensorflow.python.keras import backend as ktf
from deepcoffea_model import create_model_e
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
#from tensorflow.keras.optimizers import legacy
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dot
from keras import optimizers
import tensorflow.keras.backend as K
import sys
import numpy as np
import argparse
import random
import pickle
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LayerNormalization, MultiHeadAttention, Dropout, Embedding, Reshape, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Reshape
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Layer
#import keras.backend as K
#### Stop the model training when 0.002 to get the best result in the paper!!!!

os.environ["CUDA_VISIBLE_DEVICES"] = "2";
loss_history = []
parser = argparse.ArgumentParser()
loss = 1

def get_session(gpu_fraction=0.85):
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,allow_growth=True)
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)
    return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

ktf.set_session(get_session())

# Load the dataset
with open('/home/tc9614/TranoptiFlow/interval_based_representation_interval_based.pickle', 'rb') as handle:
    traces = pickle.load(handle, encoding='latin1')

#only selecting partial dataset for now for making the code work
traces = traces[:80000]

traces_train, traces_test = train_test_split(traces, test_size=0.3, random_state=1242)

print(len(traces_train))

feature_count = 64

#simple neural network embedding 
# embedding_model = Sequential([
#     Dense(64, activation='relu', input_shape=(2,)),  # Assuming each interval is of shape (2,)
#     Dense(feature_count),
#     Reshape((1, feature_count))  # `feature_count` is the desired size of your embeddings
# ])

def transformer_encoder(inputs, embed_dim, num_heads, ff_dim, rate=0.1):
    # Multi-head self-attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attention_output = Dropout(rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed-forward layer
    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dropout(rate)(ff_output)
    ff_output = Dense(embed_dim)(ff_output)
    ff_output = LayerNormalization(epsilon=1e-6)(attention_output + ff_output)
    
    return ff_output

def create_transformer_embedding_model(input_shape, feature_count, embed_dim, num_heads, ff_dim, rate=0.1):
    inputs = Input(shape=input_shape)
    
    # Assuming input_shape like (2,) we first project it to a higher dimensional space
    x = Dense(embed_dim, activation="relu")(inputs)
    x = Reshape((1, embed_dim))(x)  # Reshape for the attention layer

    # Transformer encoder
    transformer_block = transformer_encoder(x, embed_dim, num_heads, ff_dim, rate)
    
    # You might want to flatten or pool the transformer output here depending on your needs
    x = Flatten()(transformer_block)
    x = Dense(feature_count)(x)  # Output layer to generate embeddings of size `feature_count`
    
    model = Model(inputs=inputs, outputs=x)
    return model

# Model configuration
input_shape = (2,)
feature_count = 64 
embed_dim = 64  
num_heads = 2  
ff_dim = 64  # Hidden layer size in feed forward network inside transformer

embedding_model = create_transformer_embedding_model(input_shape, feature_count, embed_dim, num_heads, ff_dim)
embedding_model.summary()

# Embed the data
traces_train_embedded = embedding_model.predict(traces_train)
traces_test_embedded = embedding_model.predict(traces_test)


def extend_model_with_cnn(base_model, sequence_length=64, num_channels=1):
    inputs = base_model.input  
    x = base_model.output  # Start from the base model output
    x = Reshape((sequence_length, num_channels))(x)

    # Add new layers
    x = Conv1D(filters=32, kernel_size=2, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Flatten()(x)  

    # Create a new model
    new_model = Model(inputs=inputs, outputs=x)
    return new_model


# Extend the embedding model with the new layers
extended_model = extend_model_with_cnn(embedding_model)

#extended_model.compile(optimizer='adam', loss='binary_crossentropy')

traces_train_embedded = embedding_model.predict(traces_train)
traces_test_embedded = embedding_model.predict(traces_test)


print('feature_count', feature_count)
print('dim red size', np.array(traces_test_embedded).shape)
feature_count = np.array(traces_test_embedded).shape[-1]

#Inputs for the triplet loss model
anchor = Input(shape=(2,), name='anchor')
positive = Input(shape=(2,), name='positive')
negative = Input(shape=(2,), name='negative')

shared_model = create_model_e(emb_size=64, model_name='shared')

def create_combined_model(embedding_model, shared_model):
    input_shape = (2,)  # Assuming the original input shape for your embeddings
    inputs = Input(shape=input_shape)
    
    # Generate embeddings
    embeddings = embedding_model(inputs)
    
    print('embeddings.shape', embeddings.shape)
    # Process embeddings through the shared model
    outputs = shared_model(embeddings)
    
    # Create and return the combined model
    combined_model = Model(inputs=inputs, outputs=outputs)
    return combined_model

print('combined model')
combined_model = create_combined_model(embedding_model, shared_model)
combined_model.summary()

a = combined_model(anchor)
p = combined_model(positive)
n = combined_model(negative)

print('a shape', a.shape)
print('p shape', p.shape)
print('n shape', n.shape)
pos_sim = Dot(axes=-1, normalize=True)([a, p])
neg_sim = Dot(axes=-1, normalize=True)([a, n])
print('pos_sim shape', pos_sim.shape)
print('neg_sim shape', neg_sim.shape)


class TripletLossLayer(Layer):
    def __init__(self, alpha=0.2, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        #a, p, n = inputs
        p_dist, n_dist = inputs
        #p_dist = K.sum(K.square(a - p), axis=-1)
        #n_dist = K.sum(K.square(a - n), axis=-1)
        return K.mean(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

# Assuming a, p, n are the embeddings from shared_model
loss_layer = TripletLossLayer(name='triplet_loss_layer')([pos_sim, neg_sim])

model_triplet = Model(
    inputs=[anchor, positive, negative],
    outputs=loss_layer  
)

#opt = optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
#opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
opt = tf.keras.optimizers.Adadelta(rho=0.95, epsilon=1e-07)

model_triplet.compile(optimizer=opt)

print('check for model compilation')

def generate_embeddings(data):
    #we have already embedded above
    embeddings = normalize(data)
    print('embeddings done')
    return data

#based on cosine similarity
def create_triplets(embeddings, num_triplets=1600):
    triplets = []
    dist_matrix = pairwise_distances(embeddings, metric='cosine')
    
    for _ in range(num_triplets):
        #print('loop gone inside triplets')
        anchor_idx = np.random.randint(0, len(embeddings))
        
        # Get distances to all other points
        distances = dist_matrix[anchor_idx]
        
        # Sort indices by distance; the closest will be at the beginning, after the anchor itself
        sorted_indices = np.argsort(distances)
        
        # Positive: Closest point (excluding the anchor itself)
        positive_idx = sorted_indices[1]  # [0] is the anchor
        
        # Negative: Pick a point farther than the positive but within some range
        # This example picks the 90th percentile distance among the sorted ones
        negative_idx = sorted_indices[int(len(sorted_indices) * 0.90)]
        
        triplets.append((anchor_idx, positive_idx, negative_idx))
    
    return np.array(triplets)


traces_train_array = np.array(traces_train_embedded)

print(traces_train_array.shape)

#flatten
X = traces_train_array.reshape(traces_train_array.shape[0], -1)

# print(X.shape)
embeddings = generate_embeddings(X)
triplets = create_triplets(embeddings)

print('check post triplets')

X_train_anchors = np.array([traces_train[i] for i in triplets[:, 0]])
X_train_positives = np.array([traces_train[j] for j in triplets[:, 1]])
X_train_negatives = np.array([traces_train[k] for k in triplets[:, 2]])


dummy_labels = np.zeros((len(X_train_anchors),))


print('check before fitting')
#  Fit the model
history = model_triplet.fit(
    [X_train_anchors, X_train_positives, X_train_negatives], 
    dummy_labels,
    epochs=50,  
    batch_size=64
    #validation_split=0.1  
)
