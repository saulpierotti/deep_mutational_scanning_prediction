#!/usr/bin/env python3
"""
A module with utility functions that define a linear model, train it, and
obtain test predictions.
"""

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR


def get_model(x_train, y_train, params):
    """
    Train an instance of the linear predictor using the
    data x and y. The model is returned.
    """

    possible_svm_params = ["kernel", "degree", "gamma", "C", "epsilon"]

    svm_params = {
        key: value
        for key, value in params.items()
        if key in possible_svm_params
    }

    if params["kernel"] == "linear":
        del svm_params["kernel"]
        svm_model = LinearSVR(**svm_params, max_iter=10000)
    else:
        svm_model = SVR(**svm_params, max_iter=1000000)

    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), svm_model
    )
    model.fit(X=x_train, y=y_train)

    return model


def train_and_predict(x_train, y_train, x_test, params, groups=None):
    """
    Get the test predictions.
    Params and Groups is only for API compatibility.
    """
    del groups
    model = get_model(x_train, y_train, params)

    return model.predict(x_test).flatten()
