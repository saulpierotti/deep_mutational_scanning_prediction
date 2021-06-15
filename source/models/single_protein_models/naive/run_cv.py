#!/usr/bin/env python3
"""
Perform the cross_validation random search on the single protein models naive
"""


import sys

from utils import cross_validation, models, preprocess_dms

if __name__ == "__main__":
    cross_validation.single_protein_models_cv_main(
        json_params_file=sys.argv[1],
        ModelClass=models.GradientBoostedTreeModel,
        subdir_name="naive",
        label_type="labels",
        get_x_y=preprocess_dms.get_x_y_onehot,
        groups=None,
    )
