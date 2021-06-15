#!/usr/bin/env python3
"""
This is an utility script with hard-coded paths and should not be imported or
used for any other purpose. It performs a random search of hyperparameters for
an XGB predictor on a very specific dataset, with data specific pre-processing.
In order to change the search space and the number of iterations change the
variables at the bottom. This version is parallelized with MPI. Launch it with
mpirun -n <num_processes> python <script_name>.
To replicate the same train-test split shuffle with sklearn with random state 1
and then use GroupKFold with 2 folds. The first one is cv and the second is
test.
"""


import json
import os
import sys

from utils import constants, cross_validation, mpi_parallelization, svm_wrapper


def get_result_chunk(df_dms, columnlabels, params_chunk, metrics):
    """
    This is the parallel part of the pipeline, the work on a data chunk that
    needs then to be gathered.
    """

    result_chunk = []

    for i, curr_params in params_chunk:
        # this is the name used in the index to identify each param
        # combination. The combination can then be retrieved with
        # params_random_search[i]
        param_index = "param_" + str(i)
        result_chunk.extend(
            cross_validation.get_lopo_result(
                param_index=param_index,
                curr_params=curr_params,
                df_dms=df_dms,
                labels_type="labels",
                columnlabels=columnlabels,
                metrics=metrics,
                train_and_predict_func=svm_wrapper.train_and_predict,
            )
        )
        print("Completed param combination {}".format(i + 1))

    return result_chunk


def main(df_dms, columnlabels, params_to_explode, num_samples):
    """
    The main function
    """

    def preamble():
        assert not os.path.isfile(out_file)

    def get_params_list():
        params_random_search_gen = (
            cross_validation.get_random_search_param_gen(
                params_to_explode, num_samples
            )
        )

        params_random_search_enum_list = list(
            enumerate(params_random_search_gen)
        )

        return params_random_search_enum_list

    def conclude(params_list, results):
        # params_list and results are fished from the output of previous
        # functions in the MPI wrapper, the rest is obrained from the outer
        # scope of this main function
        corr_df = cross_validation.get_corr_df(results)
        cross_validation.dump_results(
            corr_df,
            params_list,
            columnlabels["features"],
            constants.metrics,
            out_file,
        )

    out_file = "{}/hyperparameter_search/lopo_models/svm/{}".format(
        constants.root, cross_validation.get_dump_name()
    )

    mpi_parallelization.main(
        preamble,
        get_params_list,
        lambda params_chunk: get_result_chunk(
            df_dms, columnlabels, params_chunk, constants.metrics
        ),
        conclude,
    )


if __name__ == "__main__":

    with open(sys.argv[1]) as handle:
        json_content = json.load(handle)

    # the hyperparameters to try
    PARAMS_TO_EXPLODE = json_content["params"]
    # the number of combinations of params to test
    # -1 for all the possible combinations
    NUM_SAMPLES = json_content["num_samples"]
    del json_content

    main(
        constants.df_dms,
        constants.df_dms_columnlabels,
        PARAMS_TO_EXPLODE,
        NUM_SAMPLES,
    )
