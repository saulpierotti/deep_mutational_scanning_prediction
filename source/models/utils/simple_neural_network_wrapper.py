#!/usr/bin/env python3
"""
A module with utility functions that define a linear model, train it, and
obtain test predictions.
"""

import numpy as np
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_neural_network(nn_params):
    """
    Function that builds and returns a neural neural_network model
    """
    inputs_aa = tf.keras.Input(shape=2)
    x_aa = tf.keras.layers.Embedding(
        input_dim=20, output_dim=10, mask_zero=False
    )(inputs_aa)
    x_aa = tf.keras.layers.Flatten()(x_aa)
    inputs_n = tf.keras.Input(shape=22)
    x_comm = tf.keras.layers.Concatenate()([x_aa, inputs_n])
    x_comm = tf.keras.layers.Dense(100, activation="relu")(x_comm)
    x_comm = tf.keras.layers.Dropout(0.2)(x_comm)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x_comm)
    model = tf.keras.Model(inputs=(inputs_aa, inputs_n), outputs=outputs)
    model.compile(optimizer=nn_params["optimizer"], loss=nn_params["loss"])

    return model


def get_model(x_train, y_train, params):
    """
    Train an instance of the linear predictor using the
    data x and y. The model is returned.
    """

    possible_nn_params = ["optimizer", "loss"]

    nn_params = {
        key: value
        for key, value in params.items()
        if key in possible_nn_params
    }

    neural_network = tf.keras.wrappers.scikit_learn.KerasRegressor(
        build_fn=get_neural_network(nn_params)
    )
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), neural_network
    )
    model.fit(X=x_train, y=y_train.reshape(-1, 1))

    return model


def train_and_predict(x_train, y_train, x_test, params, groups=None):
    """
    Get the test predictions.
    Params and Groups is only for API compatibility.
    """
    del groups
    model = get_model(x_train, y_train, params)

    return model.predict(x_test).flatten()
