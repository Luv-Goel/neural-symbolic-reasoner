"""
Neural Network Module for Hybrid AI

This module contains neural network components that will be used for:
- Feature extraction
- Pattern recognition
- Learning representations
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

class FeatureExtractor(tf.keras.layers.Layer):
    """
    A simple neural network layer for extracting features from input data.
    """
    def __init__(self, embedding_dim=64, **kwargs):
        super(FeatureExtractor, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.dense1 = Dense(128, activation='relu')
        self.dropout1 = Dropout(0.3)
        self.dense2 = Dense(self.embedding_dim, activation='relu')
        super(FeatureExtractor, self).build(input_shape)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout1(x)
        return self.dense2(x)

def build_feature_extractor_model(input_shape, embedding_dim=64):
    """
    Builds a model specifically for feature extraction.
    """
    inputs = Input(shape=input_shape)
    features = FeatureExtractor(embedding_dim=embedding_dim)(inputs)
    model = Model(inputs=inputs, outputs=features)
    return model

if __name__ == '__main__':
    # Example usage:
    input_shape = (784,) # Example input shape (e.g., flattened image)
    model = build_feature_extractor_model(input_shape)
    model.summary()