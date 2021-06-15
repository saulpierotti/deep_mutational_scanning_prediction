#!/usr/bin/env python3
"""
Functions for manipulating the DMS data
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

from utils import constants


def get_x_y_onehot(label_type):
    """
    Extracts features and labels from the dataframe.
    Labels are returned as a numpy array.
    Features are one-hot encoded if categorical and returned as a pandas
    dataframe
    """

    # get dummies is an amazing function that automagically converts to 1-hot
    # the categorical columns, giving sensible names to the new columns
    x_df = pd.get_dummies(
        constants.df_dms[constants.df_dms_columnlabels["features"]],
        prefix_sep="_dummy_",
    ).astype(float)
    y_vec = constants.df_dms[
        constants.df_dms_columnlabels[label_type]
    ].to_numpy(dtype="float")

    return x_df, y_vec


def get_single_protein_split_indexes_by_dataset(groups=None):
    """
    Obtain a dictionary with the possible datasets as keys.
    Each values is itself a dictionary returned by the provided function
    get_split_indexes.
    The key of each dictionary entry corresponds to the dataset in testing.

    The returned object is so organized:
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

    def get_split_indexes(indexes, groups_all):
        # works in place
        constants.np_rng.shuffle(indexes)

        if groups_all is not None:
            kfold_gen = GroupKFold(n_splits=2).split(
                indexes, groups=groups_all[indexes]
            )
        else:
            kfold_gen = KFold(n_splits=2).split(indexes)

        # this are indexes for the indexes array!
        train_hyperindex, test_hyperindex = next(kfold_gen)

        return {
            "train": indexes[train_hyperindex],
            "test": indexes[test_hyperindex],
        }

    datasets_vec = constants.df_dms[constants.df_dms_columnlabels["datasets"]]
    indexes = np.arange(len(constants.df_dms))
    split_indexes_dict = {
        "groups": groups,
        "indexes": dict(),
    }

    for curr_dataset in sorted(set(datasets_vec)):
        split_indexes_dict["indexes"][curr_dataset] = get_split_indexes(
            indexes=indexes[datasets_vec == curr_dataset],
            groups_all=groups,
        )
        assert (
            len(split_indexes_dict["indexes"][curr_dataset]["train"])
            + len(split_indexes_dict["indexes"][curr_dataset]["test"])
        ) == sum(datasets_vec == curr_dataset)

    assert sum(
        [
            len(vec)
            for dataset_dict in split_indexes_dict["indexes"].values()
            for vec in dataset_dict.values()
        ]
    ) == len(constants.df_dms)

    return split_indexes_dict


def get_lopo_split_indexes_by_protein(groups=None):
    """
    Obtain a dictionary with the possible datasets as keys.
    Each values is itself a dictionary returned by the provided function
    get_split_indexes.
    The key of each dictionary entry corresponds to the dataset in testing.
    The training set consists of all the proteins except the one in testing.
    The protein in testing is split in half in training and validation,
    respecting groups if provided.
    Note that the keys here are proteins, not datasets.
    groups refer to the division between test and validaiton.

    The returned object is so organized:
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

    def get_split_indexes(train_indexes, val_and_test_indexes, groups_all):
        # works in place
        constants.np_rng.shuffle(val_and_test_indexes)

        if groups_all is not None:
            kfold_gen = GroupKFold(n_splits=2).split(
                val_and_test_indexes, groups=groups_all[val_and_test_indexes]
            )
        else:
            kfold_gen = KFold(n_splits=2).split(val_and_test_indexes)

        # this are indexes for the indexes array!
        val_hyperindex, test_hyperindex = next(kfold_gen)

        return {
            "train": train_indexes,
            "test": val_and_test_indexes[test_hyperindex],
            "val": val_and_test_indexes[val_hyperindex],
        }

    proteins_vec = constants.df_dms[constants.df_dms_columnlabels["protein"]]
    indexes = np.arange(len(constants.df_dms))
    split_indexes_dict = {"groups": groups, "indexes": dict()}
    index_lopo_gen = LeaveOneGroupOut().split(indexes, groups=proteins_vec)

    for train, val_and_test in index_lopo_gen:
        assert len(set(proteins_vec[val_and_test])) == 1
        curr_protein = set(proteins_vec[val_and_test]).pop()
        split_indexes_dict["indexes"][curr_protein] = get_split_indexes(
            train_indexes=train,
            val_and_test_indexes=val_and_test,
            groups_all=groups,
        )
        assert (
            len(split_indexes_dict["indexes"][curr_protein]["train"])
            + len(split_indexes_dict["indexes"][curr_protein]["test"])
            + len(split_indexes_dict["indexes"][curr_protein]["val"])
        ) == len(constants.df_dms)

    return split_indexes_dict
