#!/usr/bin/env python3
"""
Get the test results and feature importances for the LOPO linear regression
models.
The model does not have any parameter, so no file argument is needed.
"""

from utils import constants, models, preprocess_dms, testing

if __name__ == "__main__":
    testing.lopo_models_test_main(
        json_params_file=None,
        subdir_name="linear_regression",
        label_type="labels_quantile",
        ModelClass=models.LinearRegressionModel,
        get_x_y=preprocess_dms.get_x_y_onehot,
        groups=constants.df_dms[constants.df_dms_columnlabels["groups"]],
    )
