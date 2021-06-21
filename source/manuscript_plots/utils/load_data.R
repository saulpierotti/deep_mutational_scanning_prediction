#!/usr/bin/Rscript
# Load data
#
# Author: Saul Pierotti
# Mail: saulpierotti.bioinfo@gmail.com
# Last updated: 10/06/2021
#
# This script contains data imports
# Paths are relative to the parent folder.
###############################################################################
source("./utils/functions.R")
library("Biostrings")
library("tidyverse")


# Dataset exploration
###############################################################################
uniprot_compo <- read_csv(
  "../../dataset/uniprot_composition.csv",
  col_types = cols_only(aa = readr::col_factor(levels = residues), perc = "d")
)

training_tb <-
  read_csv(
    "../../dataset/dms_training.csv",
    col_types = cols_only(
      dms_id = readr::col_factor(levels = datasets),
      position = "i",
      aa1 = readr::col_factor(levels = residues),
      aa2 = readr::col_factor(levels = residues),
      reported_fitness = "d",
      hmm_pssm_C = "d",
      hmm_pssm_G = "d",
      hmm_pssm_P = "d",
      hmm_pssm_A = "d",
      hmm_pssm_V = "d",
      hmm_pssm_L = "d",
      hmm_pssm_I = "d",
      hmm_pssm_M = "d",
      hmm_pssm_F = "d",
      hmm_pssm_Y = "d",
      hmm_pssm_W = "d",
      hmm_pssm_T = "d",
      hmm_pssm_S = "d",
      hmm_pssm_N = "d",
      hmm_pssm_Q = "d",
      hmm_pssm_D = "d",
      hmm_pssm_E = "d",
      hmm_pssm_H = "d",
      hmm_pssm_K = "d",
      hmm_pssm_R = "d",
      hmm_pssm_aa1_likelyhood = "d",
      hmm_pssm_aa2_likelyhood = "d",
      hmm_pssm_delta_likelyhood = "d",
      netsurf_rsa = "d",
      netsurf_asa = "d",
      netsurf_p_q8_G = "d",
      netsurf_p_q8_H = "d",
      netsurf_p_q8_I = "d",
      netsurf_p_q8_B = "d",
      netsurf_p_q8_E = "d",
      netsurf_p_q8_S = "d",
      netsurf_p_q8_T = "d",
      netsurf_p_q8_C = "d",
      netsurf_p_q3_H = "d",
      netsurf_p_q3_E = "d",
      netsurf_p_q3_C = "d",
      netsurf_phi = "d",
      netsurf_psi = "d",
      netsurf_disorder = "d",
      dssp_rsa = "d",
      dssp_asa = "d",
      dssp_sec_struct = readr::col_factor(levels = sec_struct_q8_levels),
      dssp_phi = "d",
      dssp_psi = "d",
      ev_independent = "d",
      ev_epistatic = "d",
      ev_conservation = "d",
      ev_frequency = "d",
      tr_rosetta_graph_closeness_centrality = "d",
      tr_rosetta_graph_betweenness_centrality = "d",
      tr_rosetta_graph_degree_centrality = "d",
      tr_rosetta_graph_load_centrality = "d",
      tr_rosetta_graph_harmonic_centrality = "d",
      tr_rosetta_graph_clustering = "d"
    )
  ) %>%
  normalize_scores() %>%
  subset(select = -c(reported_fitness))

# Features exploration
###############################################################################

data("BLOSUM100") # this is from the Biostrings package
blosum_tb <- BLOSUM100 %>%
  as_tibble(rownames = NA) %>%
  rownames_to_column("aa1") %>%
  pivot_longer(cols = -c(aa1), names_to = "aa2", values_to = "score") %>%
  filter(aa1 %in% residues & aa2 %in% residues) %>%
  mutate(
    aa1 = factor(aa1, levels = residues),
    aa2 = factor(aa2, levels = residues)
  )

average_wt_blosum <- (blosum_tb %>%
  filter(aa1 == aa2) %>%
  subset(select = c(score)) %>%
  summarise(score = mean(score)))[[1]]

blosum_tb <- blosum_tb %>%
  mutate(
    scaled_score = rescale(score) - rescale(average_wt_blosum,
      from = c(min(score), max(score))
    )
  )

distance_tb <-
  read_csv(
    "../../processing/structures/experimental_and_predicted_distances.csv",
    col_types = cols_only(
      dms_id = readr::col_factor(levels = datasets),
      position_1 = "i",
      position_2 = "i",
      tr_rosetta_distance = "d",
      experimental_distance = "d"
    )
  )

# Single protein models
###############################################################################
test_result_naive_tb <-
  get_test_result_tb(
    "../../testing_results/single_protein_models/naive/test_results.csv.xz"
  )

test_result_by_position_tb <-
  get_test_result_tb(
    "../../testing_results/single_protein_models/by_position/test_results.csv.xz"
  )

feature_importance_naive_tb <-
  read_csv(
    "../../testing_results/single_protein_models/naive/feature_importance.csv.xz"
  )

feature_importance_by_position_tb <-
  read_csv(
    "../../testing_results/single_protein_models/by_position/feature_importance.csv.xz"
  )

random_search_results_naive_tb <-
  get_random_search_results_tb_xgb(
    "../../hyperparameter_search/single_protein_models/naive/random_search_results.csv.xz"
  )

random_search_results_by_position_tb <-
  get_random_search_results_tb_xgb(
    "../../hyperparameter_search/single_protein_models/by_position/random_search_results.csv.xz"
  )

# LOPO models
###############################################################################

test_result_lopo_xgb_tb <-
  get_test_result_tb(
    "../../testing_results/lopo_models/gradient_boosted_trees/test_results.csv.xz"
  )

test_result_lopo_linear_tb <-
  get_test_result_tb(
    "../../testing_results/lopo_models/linear_regression/test_results.csv.xz"
  )

feature_importance_lopo_xgb_tb <-
  read_csv(
    "../../testing_results/lopo_models/gradient_boosted_trees/feature_importance.csv.xz"
  )

feature_importance_lopo_linear_tb <-
  read_csv(
    "../../testing_results/lopo_models/linear_regression/feature_importance.csv.xz"
  )

random_search_results_lopo_xgb_tb <-
  get_random_search_results_tb_xgb(
    "../../hyperparameter_search/lopo_models/gradient_boosted_trees/random_search_results.csv.xz"
  )

random_search_results_lopo_svm_tb <-
  get_random_search_results_tb_svm(
    "../../hyperparameter_search/lopo_models/svm/random_search_results.csv.xz"
  )

# LOPO models comparisons
###############################################################################

naive_correlation_tb <-
  test_result_naive_tb %>%
  get_test_correlations() %>%
  filter(kind == "test") %>%
  summarise(dms_id,
    model = "naive",
    method = name,
    correlation = value
  )

by_position_correlation_tb <-
  test_result_by_position_tb %>%
  get_test_correlations() %>%
  filter(kind == "test") %>%
  summarise(dms_id,
    model = "by_position",
    method = name,
    correlation = value
  )

lopo_models_xgb_correlation_tb <-
  test_result_lopo_xgb_tb %>%
  get_test_correlations() %>%
  filter(kind == "test") %>%
  summarise(dms_id,
    model = "lopo_models_xgb",
    method = name,
    correlation = value
  )

single_protein_comparison_tb <- naive_correlation_tb %>%
  bind_rows(by_position_correlation_tb)

xgb_models_comparison_tb <- single_protein_comparison_tb %>%
  bind_rows(lopo_models_xgb_correlation_tb)

linear_tb_pred <- test_result_lopo_linear_tb %>%
  group_by(dms_id) %>%
  summarise(
    dms_id,
    position,
    aa1,
    aa2,
    kind,
    y_pred_linear = y_pred,
    rank_true = rank(y_true)
  )
xgb_tb_pred <- test_result_lopo_xgb_tb %>%
  group_by(dms_id) %>%
  summarise(
    dms_id,
    position,
    aa1,
    aa2,
    kind,
    y_pred_xgb = y_pred,
    rank_true = rank(y_true)
  )

evcouplings_tb <- read_csv(
  "../../dataset/dms_training.csv",
  col_types = cols_only(
    dms_id = readr::col_factor(levels = datasets),
    ev_independent = "d",
    ev_epistatic = "d",
    reported_fitness = "d",
    position = "d",
    aa1 = "f",
    aa2 = "f"
  )
)

evcouplings_tb_pred <- evcouplings_tb %>%
  drop_na() %>%
  summarise(
    dms_id,
    position,
    aa1,
    aa2,
    y_pred_evcouplings = ev_epistatic
  )

general_models_rank_tb <- linear_tb_pred %>%
  inner_join(xgb_tb_pred,
    by = c("dms_id", "position", "aa1", "aa2", "kind", "rank_true")
  ) %>%
  filter(kind == "test") %>%
  select(-kind) %>%
  inner_join(evcouplings_tb_pred,
    by = c("dms_id", "position", "aa1", "aa2")
  ) %>%
  summarise(dms_id,
    # rebuild the ranks according to the left-out entries
    rank_true = rank(rank_true),
    rank_pred_linear = rank(y_pred_linear),
    rank_pred_xgb = rank(y_pred_xgb),
    rank_pred_evcouplings = rank(y_pred_evcouplings)
  )

envision_tb <-
  read_csv(
    "../../dataset/gray2018/envision_performances.csv",
    col_types = cols(dms_id = readr::col_factor(levels = datasets))
  ) %>%
  summarise(
    dms_id,
    model = "envision", correlation = spearman, method = "spearman"
  )

general_models_corr_tb <- general_models_rank_tb %>%
  group_by(dms_id) %>%
  summarise(
    dms_id,
    ev_epistatic = cor(rank_true,
      rank_pred_evcouplings,
      method = "spearman"
    ),
    lopo_models_linear = cor(rank_true,
      rank_pred_linear,
      method = "spearman"
    ),
    lopo_models_xgb = cor(rank_true,
      rank_pred_xgb,
      method = "spearman"
    )
  ) %>%
  distinct() %>%
  pivot_longer(
    cols = c(lopo_models_linear, lopo_models_xgb, ev_epistatic),
    names_to = "model",
    values_to = "correlation"
  ) %>%
  mutate(method = "spearman") %>%
  bind_rows(envision_tb)
