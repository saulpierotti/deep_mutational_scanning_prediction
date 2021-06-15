#!/usr/bin/env python3
"""
Get the test results and feature permutation importances for the naive
single protein models
"""


import sys

from utils import models, preprocess_dms, testing

if __name__ == "__main__":
    testing.single_protein_models_test_main(
        json_params_file=sys.argv[1],
        subdir_name="naive",
        label_type="labels",
        ModelClass=models.GradientBoostedTreeModel,
        get_x_y=preprocess_dms.get_x_y_onehot,
        groups=None,
    )
