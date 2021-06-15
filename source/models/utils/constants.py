#!/usr/bin/env python
"""
Import all the constants that are reused across the module
"""

import git
import numpy as np
import pandas as pd
from scipy import stats

# find the git root
repo = git.Repo(".", search_parent_directories=True)
root = repo.working_tree_dir

# random seed used by all the random processes
RANDOM_SEED = 1
np_rng = np.random.default_rng(RANDOM_SEED)

# This is the main dataframe with all the features and response variables,
# and also additional information
df_dms_filereader = pd.read_csv(
    "{}/dataset/dms_training.csv".format(root), low_memory=False
)
df_dms = pd.DataFrame(df_dms_filereader)

df_dms_columnlabels = {
    # which column of df_dms to use as label for the regression
    "labels": "reported_fitness",
    # which column of df_dms to use as label for the regression
    # when I want a normalized label
    "labels_quantile": "reported_fitness_quantile",
    # columns of df identifying datasets that should be processed
    # separately
    "datasets": "dms_id",
    # columns of df identifying the protein, for lopo training
    "protein": "uniprot_id",
    # column of df to use to differentiate groups that should not end up in
    # the same fold
    "groups": "feature_position",
    # the features that I want my predictor to use. Comment out the ones that
    # need to be excluded
    "features": [
        "aa1",
        "aa2",
        "ev_frequency",
        "ev_conservation",
        "ev_independent",
        "ev_epistatic",
        "netsurf_rsa",
        "netsurf_asa",
        "netsurf_p_q3_H",
        "netsurf_p_q3_E",
        "netsurf_p_q3_C",
        "netsurf_p_q8_G",
        "netsurf_p_q8_H",
        "netsurf_p_q8_I",
        "netsurf_p_q8_B",
        "netsurf_p_q8_E",
        "netsurf_p_q8_S",
        "netsurf_p_q8_T",
        "netsurf_p_q8_C",
        "netsurf_phi",
        "netsurf_psi",
        "netsurf_disorder",
        "hmm_pssm_A",
        "hmm_pssm_C",
        "hmm_pssm_D",
        "hmm_pssm_E",
        "hmm_pssm_F",
        "hmm_pssm_G",
        "hmm_pssm_H",
        "hmm_pssm_I",
        "hmm_pssm_K",
        "hmm_pssm_L",
        "hmm_pssm_M",
        "hmm_pssm_N",
        "hmm_pssm_P",
        "hmm_pssm_Q",
        "hmm_pssm_R",
        "hmm_pssm_S",
        "hmm_pssm_T",
        "hmm_pssm_V",
        "hmm_pssm_W",
        "hmm_pssm_Y",
        "hmm_pssm_aa1_likelyhood",
        "hmm_pssm_aa2_likelyhood",
        "hmm_pssm_delta_likelyhood",
        "tr_rosetta_graph_closeness_centrality",
        "tr_rosetta_graph_betweenness_centrality",
        "tr_rosetta_graph_degree_centrality",
        "tr_rosetta_graph_load_centrality",
        "tr_rosetta_graph_harmonic_centrality",
        "tr_rosetta_graph_clustering",
    ],
    # how I want to group the features for determining the permutattion
    # importance
    "feature_groups": {
        "tr_rosetta_centrality": [
            "tr_rosetta_graph_closeness_centrality",
            "tr_rosetta_graph_betweenness_centrality",
            "tr_rosetta_graph_degree_centrality",
            "tr_rosetta_graph_load_centrality",
            "tr_rosetta_graph_harmonic_centrality",
            "tr_rosetta_graph_clustering",
        ],
        "hmm_pssm": [
            "hmm_pssm_A",
            "hmm_pssm_C",
            "hmm_pssm_D",
            "hmm_pssm_E",
            "hmm_pssm_F",
            "hmm_pssm_G",
            "hmm_pssm_H",
            "hmm_pssm_I",
            "hmm_pssm_K",
            "hmm_pssm_L",
            "hmm_pssm_M",
            "hmm_pssm_N",
            "hmm_pssm_P",
            "hmm_pssm_Q",
            "hmm_pssm_R",
            "hmm_pssm_S",
            "hmm_pssm_T",
            "hmm_pssm_V",
            "hmm_pssm_W",
            "hmm_pssm_Y",
            "hmm_pssm_aa1_likelyhood",
            "hmm_pssm_aa2_likelyhood",
            "hmm_pssm_delta_likelyhood",
        ],
        "netsurf_solvent_accessibility": [
            "netsurf_rsa",
            "netsurf_asa",
        ],
        "netsurf_secondary_structure": [
            "netsurf_p_q3_H",
            "netsurf_p_q3_E",
            "netsurf_p_q3_C",
            "netsurf_p_q8_G",
            "netsurf_p_q8_H",
            "netsurf_p_q8_I",
            "netsurf_p_q8_B",
            "netsurf_p_q8_E",
            "netsurf_p_q8_S",
            "netsurf_p_q8_T",
            "netsurf_p_q8_C",
        ],
        "netsurf_disorder": ["netsurf_disorder"],
        "netsurf_torsion_angles": [
            "netsurf_phi",
            "netsurf_psi",
        ],
        "ev_couplings": [
            "ev_frequency",
            "ev_conservation",
            "ev_independent",
            "ev_epistatic",
        ],
        "aa1": ["aa1"],
        "aa2": ["aa2"],
    },
    # this columns will be added to the test result scores dataframe
    "features_to_show_in_test_results": ["aa1", "aa2", "position"],
}


metrics = {
    "pearson": lambda y_true, y_pred: stats.pearsonr(y_true, y_pred)[0],
    "spearman": lambda y_true, y_pred: stats.spearmanr(y_true, y_pred)[0],
    "kendall": lambda y_true, y_pred: stats.kendalltau(y_true, y_pred)[0],
}

CONSTANTS_ARE_LOADED = True
