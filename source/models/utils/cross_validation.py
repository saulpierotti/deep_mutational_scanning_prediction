#!/usr/bin/env python3
"""
A module with utility functions for cross-validation
"""

import inspect
import itertools
import json
import os
import pathlib
import random
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from utils import constants, mpi_parallelization, preprocess_dms


def get_random_search_param_list(params_to_explode, num_samples_raw):
    """
    Obtain the parameters combinations to use for the random search.
    params_to_explode is dictionary with a key per hyperparameter and a list of
    values to be tested for each key.
    It returns a list of tuples.
    Each tuples contains the id of the combination (an integer) and a
    combination of parameter values as a dictionary.
    """

    def get_params_random_search_gen():
        """
        A generator cannot be sampled directly since it has unknown lenght. If
        I know for some reason the lenght of the generator I can sample indexes
        and use them to get elements from it. This avoids storing the whole
        generator in memory.
        This function is a generator of random samples from the generator
        grid_search_generator.
        """

        def sampler(num_samples):
            sampled_indexes = sorted(
                random.sample(range(big_generator_length), num_samples)
            )
            sampled_index = sampled_indexes.pop(0)
            complete = False

            for i, generator_element in enumerate(
                get_params_grid_search_gen()
            ):
                # sampled_indexes checks if it is empty or not

                if i == sampled_index:
                    if sampled_indexes:
                        sampled_index = sampled_indexes.pop(0)
                    else:
                        complete = True
                    yield generator_element

                if complete:
                    break

        random.seed(constants.RANDOM_SEED)
        big_generator_length = np.prod(
            [len(el) for el in params_to_explode.values()]
        )

        # I use -1 to indicate that I want a grid search, full sampling

        if num_samples_raw == -1:
            yield from sampler(big_generator_length)
        elif num_samples_raw <= big_generator_length:
            yield from sampler(num_samples_raw)
        else:

            for _ in range(num_samples_raw // big_generator_length):
                yield from sampler(big_generator_length)

            if num_samples_raw % big_generator_length != 0:
                yield from sampler(num_samples_raw % big_generator_length)

    def get_params_grid_search_gen():
        """
        Take a dictionary with a list of possible values for each key. Return a
        generator with all the possible combinations of values.
        """
        # I use a generator since this can be hard to fit in memory

        for param_tuple in itertools.product(*params_to_explode.values()):
            yield dict(zip(params_to_explode.keys(), param_tuple))

    return list(enumerate(get_params_random_search_gen()))


def get_cv_pred(cv_indexes, x_vec, y_vec, params, ModelClass, groups):
    """
    Perform a 5-fold cross validation while respecting groups (if given).
    Returns the predictions on the validation sets as a concatenated array.
    Every element in args is also returned in the same order after the
    prediction.
    """
    y_cv_pred_list = list()
    cv_indexes_reordered_list = list()

    if groups is not None:
        kfold_gen = GroupKFold(n_splits=5).split(
            cv_indexes, groups=groups[cv_indexes]
        )
    else:
        kfold_gen = KFold(n_splits=5).split(cv_indexes)

    for train_hyperindex, val_hyperindex in kfold_gen:
        train, val = cv_indexes[train_hyperindex], cv_indexes[val_hyperindex]
        y_cv_pred_list.append(
            ModelClass(params=params)
            .fit(x_vec[train], y_vec[train])
            .predict(x_vec[val])
        )
        cv_indexes_reordered_list.append(val)

    cv_indexes_reordered = np.concatenate(cv_indexes_reordered_list)
    y_cv_pred = np.concatenate(y_cv_pred_list)
    assert len(cv_indexes_reordered) == len(y_cv_pred)
    assert len(y_cv_pred) == len(cv_indexes)

    return y_cv_pred, cv_indexes_reordered


def get_param_result_single_protein(
    x_vec,
    y_vec,
    train_test_indexes_by_dataset_dict,
    ModelClass,
    param_index,
    params,
    groups,
):
    """
    Process a single params instance, given the data, the model, and the
    testing splits. Produce a result relative to that param instance and
    return it.
    This function is for the single protein models.

    train_test_indexes_by_dataset_dict is expected to be so organized:
    {
         "groups" : list of groups or None,
         "indexes": {
             <dataset_name> : {
                 "train": np array of indexes
                 "test": np array of indexes
                 },
             ... a key for each dataset with the same structure as above
         }
    }
    """
    result = []

    for (
        curr_dataset,
        curr_indexes,
    ) in train_test_indexes_by_dataset_dict["indexes"].items():
        y_cv_pred, cv_indexes_reordered = get_cv_pred(
            cv_indexes=curr_indexes["train"],
            x_vec=x_vec,
            y_vec=y_vec,
            params=params,
            ModelClass=ModelClass,
            groups=groups,
        )
        scores = dict()

        for metric_name, metric in constants.metrics.items():
            scores[metric_name] = metric(
                y_vec[cv_indexes_reordered], y_cv_pred
            )

        result.append([param_index, curr_dataset, scores])

    return result


def get_param_result_lopo(
    x_vec,
    y_vec,
    train_test_indexes_by_protein_dict,
    ModelClass,
    param_index,
    params,
):
    """
    Process a single params instance, given the data, the model, and the
    testing splits. Produce a result relative to that param instance and
    return it.
    This function is for the lopo models.

    train_test_indexes_by_protein_dict is expected to be so organized:
    {
        "groups" : list of groups or None,
        "indexes": {
            <protein_name> : {
                "train": np array of indexes
                "test": np array of indexes
                "val": np array of indexes
                },
            ... a key for each dataset with the same structure as above
        }
    }
    """
    result = []
    datasets_vec = constants.df_dms[constants.df_dms_columnlabels["datasets"]]

    for curr_indexes in train_test_indexes_by_protein_dict["indexes"].values():
        train, val = curr_indexes["train"], curr_indexes["val"]
        y_val_pred = (
            ModelClass(params=params)
            .fit(x_vec=x_vec[train], y_vec=y_vec[train])
            .predict(x_vec[val])
        )

        left_out_datasets = set(datasets_vec[val])
        assert len(left_out_datasets) == 1 or (
            len(left_out_datasets) == 2
            and "Ubiquitin" in left_out_datasets
            and "E1_Ubiquitin" in left_out_datasets
        )

        for curr_dataset in left_out_datasets:
            scores = dict()
            is_curr_dataset = datasets_vec[val] == curr_dataset

            for metric_name, metric in constants.metrics.items():
                scores[metric_name] = metric(
                    y_vec[val][is_curr_dataset], y_val_pred[is_curr_dataset]
                )
                result.append([param_index, curr_dataset, scores])

    return result


def get_result_chunk(
    get_curr_result,
    params_chunk,
):
    """
    This is the parallel part of the cv pipeline, the work on a data chunk that
    needs then to be gathered.
    It processes a chunk of parameter combinations and returns a dataframe of
    performances.
    get_curr_result must be just a function of params, and param_index.
    params is one of the items in params_chunk.
    Use a lambda wrapping to provide other parameters.
    """

    result_chunk = []

    for i, curr_params in params_chunk:
        # this is the name used in the index to identify each param
        # combination. The combination can then be retrieved with
        # params_random_search[i]
        param_index = "param_" + str(i)
        result_chunk.extend(
            get_curr_result(param_index=param_index, params=curr_params)
        )

    return result_chunk


def post_process_corr_df(corr_df):
    """
    Summarise corr_df adding new columns with average of row and standard error
    of the mean for row, and drop rows containing nan. Return
    the dataframe sorted in descending way using the average column as key.
    """
    # I am not interested in combinations of params that do not yield any
    # result for some datasets
    corr_df.dropna(axis=0, inplace=True)
    # The average correlation across datasets
    avg_series = corr_df.mean(axis=1)
    # The standard error of the previous average
    sem_series = corr_df.sem(axis=1)
    # I add them later otherwise I am including avg in the calculation of sem
    corr_df["average"] = avg_series
    corr_df["sem"] = sem_series
    # I sort by the average correlation, such that the first n entries are the
    # top N param combinations
    corr_df.sort_values(by="average", axis=0, inplace=True, ascending=False)


def get_corr_df(results):
    """
    Expects a list of tuples (or lists). Map the last element of each tuple
    in a dataframe at first_element, second_element (for the tuple). Return
    the dataframe.
    """
    corr_df = dict()

    for param_index, curr_dataset, scores in results:
        for metric_name in scores:
            if not metric_name in corr_df:
                corr_df[metric_name] = pd.DataFrame()
            corr_df[metric_name].at[param_index, curr_dataset] = scores[
                metric_name
            ]

    for key in corr_df:
        corr_df[key] = corr_df[key].astype(float)
        # this operates in place
        post_process_corr_df(corr_df[key])

    return corr_df


def get_dump_name():
    """
    Returns a standardised string to be used as a filename for the dump of the
    cross validation results.
    """
    time_string = time.strftime("%Y%m%d-%H%M%S")

    return "random_search_{}.joblib.xz".format(time_string)


def dump_results(corr_df, params_list, features, metrics, out_file):
    """
    Save a summary of the result in a joblib dump
    """
    # in order to retrieve later which combination of params and features
    # corresponds to the observed performance I need also to pickle these
    # variables

    # create the directory where to place out_file if it does not exists
    pathlib.Path(out_file).parents[0].mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "data": corr_df,
            "params": params_list,
            "features": features,
            "metrics": {
                key: inspect.getsource(value).strip()
                for key, value in metrics.items()
            },
        },
        out_file,
    )


def single_protein_models_cv_main(
    json_params_file, ModelClass, subdir_name, label_type, get_x_y, groups
):
    """
    Performs all the functions required to do a random hyperparameter search
    for the single protein models and saves the results.
    This function is independent of model type and validation strategy.
    Implements MPI parallelization.
    """

    def preamble(leader_vars):
        leader_vars[
            "out_file"
        ] = "{}/hyperparameter_search/single_protein_models/{}/{}".format(
            constants.root, subdir_name, get_dump_name()
        )
        assert not os.path.isfile(leader_vars["out_file"])

        with open(json_params_file) as handle:
            json_content = json.load(handle)

        # the hyperparameters to try
        leader_vars["params_to_explode"] = json_content["params"]
        # the number of combinations of params to test
        # -1 for all the possible combinations
        leader_vars["num_samples"] = json_content["num_samples"]
        del json_content

        leader_vars[
            "train_test_indexes_by_dataset_dict"
        ] = preprocess_dms.get_single_protein_split_indexes_by_dataset(
            groups=groups
        )

        return leader_vars

    def conclude(params_list, results, leader_vars):
        # params_list and results are fished from the output of previous
        # functions in the MPI wrapper, the rest is obrained from the outer
        # scope of this main function

        corr_df = get_corr_df(results)
        dump_results(
            corr_df,
            params_list,
            constants.df_dms_columnlabels["features"],
            constants.metrics,
            leader_vars["out_file"],
        )
        print(
            "Random search complete. Results dumped in {}".format(
                leader_vars["out_file"]
            )
        )
        print(corr_df)
        print(params_list)

    # this is exectued both by leader and workers
    (
        x_df,
        y_vec,
    ) = get_x_y(label_type=label_type)

    mpi_parallelization.main(
        preamble=preamble,
        get_params_list=lambda leader_vars: get_random_search_param_list(
            params_to_explode=leader_vars["params_to_explode"],
            num_samples_raw=leader_vars["num_samples"],
        ),
        get_result_chunk=lambda params_chunk, leader_vars: get_result_chunk(
            params_chunk=params_chunk,
            get_curr_result=lambda params, param_index: get_param_result_single_protein(
                x_vec=x_df.to_numpy(dtype=float),
                y_vec=y_vec,
                train_test_indexes_by_dataset_dict=leader_vars[
                    "train_test_indexes_by_dataset_dict"
                ],
                ModelClass=ModelClass,
                param_index=param_index,
                params=params,
                groups=groups,
            ),
        ),
        conclude=conclude,
    )


def lopo_models_cv_main(
    json_params_file, ModelClass, subdir_name, label_type, get_x_y, groups
):
    """
    Performs all the functions required to do a random hyperparameter search
    for the lopo models and saves the results.
    This function is independent of model type and validation strategy.
    Implements MPI parallelization.
    """

    def preamble(leader_vars):
        leader_vars[
            "out_file"
        ] = "{}/hyperparameter_search/lopo_models/{}/{}".format(
            constants.root, subdir_name, get_dump_name()
        )
        assert not os.path.isfile(leader_vars["out_file"])

        with open(json_params_file) as handle:
            json_content = json.load(handle)

        # the hyperparameters to try
        leader_vars["params_to_explode"] = json_content["params"]
        # the number of combinations of params to test
        # -1 for all the possible combinations
        leader_vars["num_samples"] = json_content["num_samples"]
        del json_content

        leader_vars[
            "train_test_indexes_by_protein_dict"
        ] = preprocess_dms.get_lopo_split_indexes_by_protein(groups=groups)

        return leader_vars

    def conclude(params_list, results, leader_vars):
        # params_list and results are fished from the output of previous
        # functions in the MPI wrapper, the rest is obrained from the outer
        # scope of this main function

        corr_df = get_corr_df(results)
        dump_results(
            corr_df,
            params_list,
            constants.df_dms_columnlabels["features"],
            constants.metrics,
            leader_vars["out_file"],
        )

        print(
            "Random search complete. Results dumped in {}".format(
                leader_vars["out_file"]
            )
        )
        print(corr_df)
        print(params_list)

    # this is exectued both by leader and workers
    (
        x_df,
        y_vec,
    ) = get_x_y(label_type=label_type)

    mpi_parallelization.main(
        preamble=preamble,
        get_params_list=lambda leader_vars: get_random_search_param_list(
            params_to_explode=leader_vars["params_to_explode"],
            num_samples_raw=leader_vars["num_samples"],
        ),
        get_result_chunk=lambda params_chunk, leader_vars: get_result_chunk(
            params_chunk=params_chunk,
            get_curr_result=lambda params, param_index: get_param_result_lopo(
                x_vec=x_df.to_numpy(dtype=float),
                y_vec=y_vec,
                train_test_indexes_by_protein_dict=leader_vars[
                    "train_test_indexes_by_protein_dict"
                ],
                ModelClass=ModelClass,
                param_index=param_index,
                params=params,
            ),
        ),
        conclude=conclude,
    )
