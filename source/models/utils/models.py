#!/usr/bin/env python3
"""
A module that defines classes for all the models used in this project.
A sklearn-like fit and predict API is provided.
Parameters are given at construction time in params.
"""


import warnings

import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class GradientBoostedTreeModel:
    """
    Gradient boosted tree model constructor using the XGBoost implementation.
    """

    def __init__(self, params):
        possible_xgb_params = [
            "booster",
            "verbosity",
            "validate_parameters",
            "nthread",
            "disable_default_eval_metric",
            "num_pbuffer",
            "num_feature",
            "eta",
            "gamma",
            "max_depth",
            "min_child_weight",
            "max_delta_step",
            "subsample",
            "sampling_method",
            "colsample_bytree",
            "lambda",
            "alpha",
            "tree_method",
            "sketch_eps",
            "scale_pos_weight",
            "updater",
            "refresh_leaf",
            "process_type",
            "grow_policy",
            "max_leaves",
            "max_bin",
            "predictor",
            "num_parallel_tree",
            "single_precision_histogram",
            "deterministic_histogram",
        ]
        self.xgb_params = {
            key: value
            for key, value in params.items()
            if key in possible_xgb_params
        }
        self.num_rounds = params["num_rounds"]
        self.model = None

    def fit(self, x_vec, y_vec, ranking_group=None):
        """
        Fit the model on the given data, using ranking groups if provided
        """

        if ranking_group is not None:
            qid_raw = ranking_group.factorize()[0]
            # qid must be sorted, and I need to sort also the data accordingly
            qid_idx_sort = np.argsort(qid_raw)
            d_train = xgb.DMatrix(
                data=x_vec[qid_idx_sort],
                label=y_vec[qid_idx_sort],
                qid=qid_raw[qid_idx_sort],
            )
        else:
            d_train = xgb.DMatrix(
                data=x_vec,
                label=y_vec,
            )

        self.model = xgb.train(
            self.xgb_params, d_train, num_boost_round=self.num_rounds
        )

        return self

    def predict(self, x_vec):
        """
        Sklearn-like predict API
        """

        if self.model is None:
            raise AssertionError("Predict called in unfitted model.")

        d_mat = xgb.DMatrix(data=x_vec)

        return self.model.predict(d_mat).flatten()


class LinearRegressionModel:
    """
    Linear regression model constructor using the Sklearn implementation.
    """

    def __init__(self, params):
        # this model does not accept any parameter
        assert params is None
        self.model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearRegression(n_jobs=-1),
        )
        self.is_fitted = False

    def fit(self, x_vec, y_vec, ranking_group=None):
        """
        Fit the model on the given data. Ranking group is provided for
        compatibility, but it is ignored.
        """

        if ranking_group is not None:
            warnings.warn("Ranking group is ignored in LinearRegressionModel")
        self.model.fit(X=x_vec, y=y_vec.reshape(-1, 1))
        self.is_fitted = True

        return self

    def predict(self, x_vec):
        """
        Sklearn-like predict API
        """

        if not self.is_fitted:
            raise AssertionError("Predict called in unfitted model.")

        return self.model.predict(x_vec).flatten()
