#!/usr/bin/env python3
"""
Perform the cross_validation random search on the single protein models by
position
"""


import sys

from utils import constants, cross_validation, models, preprocess_dms

if __name__ == "__main__":
    cross_validation.single_protein_models_cv_main(
        json_params_file=sys.argv[1],
        ModelClass=models.GradientBoostedTreeModel,
        subdir_name="by_position",
        label_type="labels",
        get_x_y=preprocess_dms.get_x_y_onehot,
        groups=constants.df_dms[constants.df_dms_columnlabels["groups"]],
    )
