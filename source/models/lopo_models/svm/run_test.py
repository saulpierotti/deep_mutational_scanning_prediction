#!/usr/bin/env python3
"""
Get the test results
"""


import json
import os
import sys

from utils import constants, preprocess_dms, testing, svm_wrapper


def main(df_dms, columnlabels, params):
    """
    The main function
    """

    out_file = (
        "{}/testing_results/lopo_models/svm/{}".format(
            constants.root, testing.get_dump_name()
        )
    )
    assert not os.path.isfile(out_file)

    df_results = testing.get_lopo_test_results_df(
        params=params,
        df_dms=df_dms,
        labels_type="labels",
        columnlabels=columnlabels,
        train_and_predict_func=svm_wrapper.train_and_predict,
    )

    testing.dump_results(
        df_results, params, columnlabels["features"], out_file
    )

    print("Testing complete. Results dumped in {}".format(out_file))
    print(df_results)


if __name__ == "__main__":

    with open(sys.argv[1]) as handle:
        json_content = json.load(handle)

    # the hyperparameters to use
    PARAMS = json_content["params"]
    del json_content

    main(constants.df_dms, constants.df_dms_columnlabels, PARAMS)
