#!/usr/bin/env python3
"""
Perform the cross_validation random search on the LOPO gradient boosted tree
models
"""

import sys

from utils import constants, cross_validation, models, preprocess_dms

if __name__ == "__main__":
    cross_validation.lopo_models_cv_main(
        json_params_file=sys.argv[1],
        subdir_name="gradient_boosted_trees",
        label_type="labels",
        ModelClass=models.GradientBoostedTreeModel,
        get_x_y=preprocess_dms.get_x_y_onehot,
        groups=constants.df_dms[constants.df_dms_columnlabels["groups"]],
    )
