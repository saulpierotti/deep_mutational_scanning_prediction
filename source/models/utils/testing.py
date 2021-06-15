#!/usr/bin/env python3
"""
A module with utility functions for evaluating performances on the test set for
my models
"""

import json
import os
import pathlib
import time

import joblib
import numpy as np
import pandas as pd

from utils import constants, cross_validation, preprocess_dms


def get_test_results_df_and_trained_models_single_protein(
    x_vec, y_vec, train_test_indexes_by_dataset_dict, ModelClass, params
):
    """
     Takes in input the whole feature and label vectors, a dictionary of indexes
     to be used for the splits in training and testing, and model parameters.
     Returns a dataframe with cross_validation and testing prediction, and the
     trained model used on the test set.

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
    groups = train_test_indexes_by_dataset_dict["groups"]
    trained_models = dict()
    df_results = pd.DataFrame()

    for (
        curr_dataset,
        curr_indexes,
    ) in train_test_indexes_by_dataset_dict["indexes"].items():
        train, test = curr_indexes["train"], curr_indexes["test"]
        y_cv_pred, cv_indexes_reordered = cross_validation.get_cv_pred(
            cv_indexes=train,
            x_vec=x_vec,
            y_vec=y_vec,
            params=params,
            ModelClass=ModelClass,
            groups=groups,
        )
        curr_trained_model = ModelClass(params=params).fit(
            x_vec=x_vec[train], y_vec=y_vec[train]
        )
        y_test_pred = curr_trained_model.predict(x_vec[test])
        curr_results = {
            constants.df_dms_columnlabels["datasets"]: curr_dataset,
            "y_pred": np.concatenate([y_cv_pred, y_test_pred]),
            "y_true": np.concatenate(
                [y_vec[cv_indexes_reordered], y_vec[test]]
            ),
            "kind": ["validation"] * len(train) + ["test"] * len(test),
        }

        # this just adds aa1, aa2, position, ecc.

        for feature in constants.df_dms_columnlabels[
            "features_to_show_in_test_results"
        ]:
            curr_results[feature] = np.concatenate(
                [
                    constants.df_dms[feature].to_numpy()[cv_indexes_reordered],
                    constants.df_dms[feature].to_numpy()[test],
                ]
            )
        df_results = pd.concat([df_results, pd.DataFrame(curr_results)])
        trained_models[curr_dataset] = curr_trained_model

    return df_results, trained_models


def get_test_results_df_and_trained_models_lopo(
    x_vec, y_vec, train_test_indexes_by_protein_dict, ModelClass, params
):
    """
    Takes in input the whole feature and label vectors, a dictionary of indexes
    to be used for the splits in training and testing, and model parameters.
    Returns a dataframe with cross_validation and testing prediction, and the
    trained model used on the test set.

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
    trained_models = dict()
    df_results = pd.DataFrame()
    datasets_vec = constants.df_dms[constants.df_dms_columnlabels["datasets"]]

    for curr_indexes in train_test_indexes_by_protein_dict["indexes"].values():
        train, test, val = (
            curr_indexes["train"],
            curr_indexes["test"],
            curr_indexes["val"],
        )
        curr_trained_model = ModelClass(params=params).fit(
            x_vec=x_vec[train],
            y_vec=y_vec[train],
            ranking_group=datasets_vec[train],
        )
        y_test_pred = curr_trained_model.predict(x_vec[test])
        y_val_pred = curr_trained_model.predict(x_vec[val])
        curr_results = {
            constants.df_dms_columnlabels["datasets"]: np.concatenate(
                [datasets_vec[val], datasets_vec[test]]
            ),
            "y_pred": np.concatenate([y_val_pred, y_test_pred]),
            "y_true": np.concatenate([y_vec[val], y_vec[test]]),
            "kind": ["validation"] * len(val) + ["test"] * len(test),
        }

        # this just adds aa1, aa2, position, ecc.

        for feature in constants.df_dms_columnlabels[
            "features_to_show_in_test_results"
        ]:
            curr_results[feature] = np.concatenate(
                [
                    constants.df_dms[feature].to_numpy()[val],
                    constants.df_dms[feature].to_numpy()[test],
                ]
            )
        df_results = pd.concat([df_results, pd.DataFrame(curr_results)])

        left_out_datasets = set(datasets_vec[val])
        assert len(left_out_datasets) == 1 or (
            len(left_out_datasets) == 2
            and "Ubiquitin" in left_out_datasets
            and "E1_Ubiquitin" in left_out_datasets
        )

        for curr_dataset in set(left_out_datasets):
            trained_models[curr_dataset] = curr_trained_model

    return df_results, trained_models


def convert_protein_indexes_to_dataset_indexes(
    train_test_indexes_by_protein_dict,
):
    """
    Converts the LOPO split indexes arranged by protein and with train, val,
    and test categories in single protein-like indexes arranged by dataset and
    with only train and test categories.

    Input is expected to be:
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

    Output is produced as:
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
    datasets_vec = constants.df_dms[constants.df_dms_columnlabels["datasets"]]
    proteins_vec = constants.df_dms[constants.df_dms_columnlabels["protein"]]
    train_test_indexes_by_dataset_dict = {
        "groups": train_test_indexes_by_protein_dict["groups"]
    }
    old_indexes = train_test_indexes_by_protein_dict["indexes"]
    new_indexes = dict()

    for curr_protein, old_index_dict in old_indexes.items():
        datasets_for_protein = set(datasets_vec[proteins_vec == curr_protein])
        assert len(datasets_for_protein) == 1 or (
            len(datasets_for_protein) == 2
            and "Ubiquitin" in datasets_for_protein
            and "E1_Ubiquitin" in datasets_for_protein
        )

        for curr_dataset in datasets_for_protein:
            new_indexes[curr_dataset] = {
                "train": old_index_dict["train"],
                "test": old_index_dict["test"][
                    datasets_vec[old_index_dict["test"]] == curr_dataset
                ],
            }

    train_test_indexes_by_dataset_dict["indexes"] = new_indexes

    return train_test_indexes_by_dataset_dict


def get_permutation_importance(
    x_df,
    y_vec,
    models_dict,
    train_test_indexes_by_dataset_dict,
    scorer,
    num_repeats=5,
):
    """
    Get the permutation importance of the features for a fitted estimator.
    I do not use the sklearn implementation since I have correlated features
    and I want to permutate in groups.
    The model is expected to be wrapped in a sklearn API (just a
    wrapped_model.predict(X) method is expected).

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

    def initialize_importance_dict():
        importance_dict = {
            constants.df_dms_columnlabels["datasets"]: list(),
            "feature_group": list(),
        }

        for i in range(num_repeats):
            importance_dict["importance_{}".format(i + 1)] = list()

        return importance_dict

    def get_corrupted_x_vec(x_test_df, feature_list):
        x_test_corrupted_df = pd.DataFrame()

        for colname_raw in x_test_df.columns:
            # this is to reverse the dummies

            if "__" in colname_raw:
                colname = colname_raw.split("__")[0]
            else:
                colname = colname_raw

            if colname in feature_list:
                x_test_corrupted_df[
                    colname_raw
                ] = constants.np_rng.permutation(x_test_df[colname_raw])
            else:
                x_test_corrupted_df[colname] = x_test_df[colname]

        return x_test_corrupted_df.to_numpy(dtype=float)

    def add_importance_single_dataset(dataset, model):
        test = train_test_indexes_by_dataset_dict["indexes"][dataset]["test"]
        x_test_df = x_df.iloc[test]
        orig_perf = scorer(
            y_vec[test], model.predict(x_test_df.to_numpy(dtype=float))
        )

        for feature_group, feature_list in constants.df_dms_columnlabels[
            "feature_groups"
        ].items():
            importance_dict["feature_group"].append(feature_group)
            importance_dict[constants.df_dms_columnlabels["datasets"]].append(
                dataset
            )

            for i in range(num_repeats):
                importance_dict["importance_{}".format(i + 1)].append(
                    orig_perf
                    - scorer(
                        y_vec[test],
                        model.predict(
                            get_corrupted_x_vec(x_test_df, feature_list)
                        ),
                    )
                )

    importance_dict = initialize_importance_dict()

    for dataset, model in models_dict.items():
        add_importance_single_dataset(dataset=dataset, model=model)

    importance_df = pd.DataFrame(importance_dict)
    # calculate mean importance and sem
    avg_series = importance_df.mean(axis=1, numeric_only=True)
    sem_series = importance_df.sem(axis=1, numeric_only=True)
    # add them after calculation, to avoid influencing the sem results
    importance_df["importance_average"] = avg_series
    importance_df["importance_sem"] = sem_series

    return importance_df


def get_dump_name():
    """
    Returns a standardised string to be used as a filename for the dump of the
    test results.
    """
    time_string = time.strftime("%Y%m%d-%H%M%S")

    return "validation_and_test_{}.joblib.xz".format(time_string)


def dump_results(
    df_results,
    feature_importance_df,
    params,
    out_file,
):
    """
    Save a summary of the result in a joblib dump.
    """
    # create the directory where to place out_file if it does not exists
    pathlib.Path(out_file).parents[0].mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "data": df_results,
            "feature_importance": feature_importance_df,
            "params": params,
            "features": constants.df_dms_columnlabels["features"],
        },
        out_file,
    )


def single_protein_models_test_main(
    json_params_file,
    subdir_name,
    label_type,
    ModelClass,
    get_x_y,
    groups,
):
    """
    Performs all the functions required to train and test the single protein
    models and saves the results.
    This function is independent of model type and validation strategy.
    It computes also feature importance.
    If the model has no parameters set json_params_file to None.
    """
    out_file = "{}/testing_results/single_protein_models/{}/{}".format(
        constants.root, subdir_name, get_dump_name()
    )
    assert not os.path.isfile(out_file)

    if json_params_file is None:
        params = None
    else:
        with open(json_params_file) as handle:
            json_content = json.load(handle)

            # the hyperparameters to use
            params = json_content["params"]
            del json_content

    x_df, y_vec = get_x_y(label_type=label_type)
    train_test_indexes_by_dataset_dict = (
        preprocess_dms.get_single_protein_split_indexes_by_dataset(
            groups=groups
        )
    )
    (
        df_results,
        models_dict,
    ) = get_test_results_df_and_trained_models_single_protein(
        x_vec=x_df.to_numpy(dtype=float),
        y_vec=y_vec,
        train_test_indexes_by_dataset_dict=train_test_indexes_by_dataset_dict,
        ModelClass=ModelClass,
        params=params,
    )

    feature_importance_df = get_permutation_importance(
        x_df=x_df,
        y_vec=y_vec,
        models_dict=models_dict,
        train_test_indexes_by_dataset_dict=train_test_indexes_by_dataset_dict,
        scorer=constants.metrics["pearson"],
    )

    dump_results(
        df_results=df_results,
        feature_importance_df=feature_importance_df,
        params=params,
        out_file=out_file,
    )

    print("Testing complete. Results dumped in {}".format(out_file))
    print(df_results)
    print(feature_importance_df)


def lopo_models_test_main(
    json_params_file,
    subdir_name,
    label_type,
    ModelClass,
    get_x_y,
    groups,
):
    """
    Performs all the functions required to train and test the lopo
    models and saves the results.
    This function is independent of model type and validation strategy.
    It computes also feature importance.
    If the model has no parameters set json_params_file to None.
    """
    out_file = "{}/testing_results/lopo_models/{}/{}".format(
        constants.root, subdir_name, get_dump_name()
    )
    assert not os.path.isfile(out_file)

    if json_params_file is None:
        params = None
    else:
        with open(json_params_file) as handle:
            json_content = json.load(handle)

            # the hyperparameters to use
            params = json_content["params"]
            del json_content

    x_df, y_vec = get_x_y(label_type=label_type)
    train_test_indexes_by_protein_dict = (
        preprocess_dms.get_lopo_split_indexes_by_protein(groups=groups)
    )
    df_results, models_dict = get_test_results_df_and_trained_models_lopo(
        x_vec=x_df.to_numpy(dtype=float),
        y_vec=y_vec,
        train_test_indexes_by_protein_dict=train_test_indexes_by_protein_dict,
        ModelClass=ModelClass,
        params=params,
    )
    train_test_indexes_by_dataset_dict = (
        convert_protein_indexes_to_dataset_indexes(
            train_test_indexes_by_protein_dict
        )
    )
    feature_importance_df = get_permutation_importance(
        x_df=x_df,
        y_vec=y_vec,
        models_dict=models_dict,
        train_test_indexes_by_dataset_dict=train_test_indexes_by_dataset_dict,
        scorer=constants.metrics["pearson"],
    )
    dump_results(
        df_results=df_results,
        feature_importance_df=feature_importance_df,
        params=params,
        out_file=out_file,
    )

    print("Testing complete. Results dumped in {}".format(out_file))
    print(df_results)
    print(feature_importance_df)
