#!/usr/bin/env python3
"""
Get the test results and feature permutation importances for the single protein
models by position
"""


import sys

from utils import constants, models, preprocess_dms, testing

if __name__ == "__main__":
    testing.single_protein_models_test_main(
        json_params_file=sys.argv[1],
        subdir_name="by_position",
        label_type="labels",
        ModelClass=models.GradientBoostedTreeModel,
        get_x_y=preprocess_dms.get_x_y_onehot,
        groups=constants.df_dms[constants.df_dms_columnlabels["groups"]],
    )
