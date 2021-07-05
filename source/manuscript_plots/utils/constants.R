#!/usr/bin/Rscript
# Constants
#
# Author: Saul Pierotti
# Mail: saulpierotti.bioinfo@gmail.com
# Last updated: 13/05/2021
#
# This script contains constants that are used in various script for the
# generation of manuscript plots
# Paths are relative to the parent folder.
###############################################################################
library("tidyverse")


# commonly used vectors and variables
###############################################################################

residues <- # the order maximizes chemical similarity
  c(
    "C",
    "G",
    "P",
    "A",
    "V",
    "L",
    "I",
    "M",
    "F",
    "Y",
    "W",
    "T",
    "S",
    "N",
    "Q",
    "D",
    "E",
    "H",
    "K",
    "R"
  )

residues_monospaced <- c(
  "C" = "\\texttt{C}",
  "G" = "\\texttt{G}",
  "P" = "\\texttt{P}",
  "A" = "\\texttt{A}",
  "V" = "\\texttt{V}",
  "L" = "\\texttt{L}",
  "I" = "\\texttt{I}",
  "M" = "\\texttt{M}",
  "F" = "\\texttt{F}",
  "Y" = "\\texttt{Y}",
  "W" = "\\texttt{W}",
  "T" = "\\texttt{T}",
  "S" = "\\texttt{S}",
  "N" = "\\texttt{N}",
  "Q" = "\\texttt{Q}",
  "D" = "\\texttt{D}",
  "E" = "\\texttt{E}",
  "H" = "\\texttt{H}",
  "K" = "\\texttt{K}",
  "R" = "\\texttt{R}"
)

xgb_hyperparameters <- c(
  "alpha",
  "lambda",
  "gamma",
  "eta",
  "subsample",
  "colsample_bytree",
  "min_child_weight",
  "max_depth",
  "num_rounds"
)

datasets <- c(
  "beta-lactamase",
  "E1_Ubiquitin",
  "Ubiquitin",
  "gb1",
  "hsp90",
  "kka2_1:2",
  "Pab1",
  "PSD95pdz3",
  "WW_domain"
)

pssm_cols <- c(
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
  hmm_pssm_R = "d"
)

sec_struct_q8_levels <- c(
  "G",
  "H",
  "I",
  "T",
  "E",
  "B",
  "S",
  "C"
)

sec_struct_q3_levels <- c(
  "H",
  "E",
  "C"
)

q8_to_q3_vec <- c(
  H = "G",
  H = "H",
  H = "I",
  C = "T",
  E = "E",
  E = "B",
  C = "S",
  C = "C"
)

dms_dataset_labels <- c(
  `beta-lactamase` = "beta-lactamase",
  E1_Ubiquitin = "E1\\_Ubiquitin",
  Ubiquitin = "Ubiquitin",
  gb1 = "gb1",
  hsp90 = "hsp90",
  `kka2_1:2` = "kka2\\_1:2",
  Pab1 = "Pab1",
  PSD95pdz3 = "PSD95pdz3",
  WW_domain = "WW\\_domain"
)

feature_group_labels <- c(
  `ev_couplings` = "EVcouplings",
  `tr_rosetta_centrality` = "trRosetta",
  `netsurf_solvent_accessibility` = "NetsurfP-2 solvent accessibility",
  `netsurf_disorder` = "NetsurfP-2 disorder",
  `netsurf_torsion_angles` = "NetsurfP-2 torsion angles",
  `netsurf_secondary_structure` = "NetsurfP-2 secondary structure",
  `hmm_pssm` = "PSSM",
  `aa1` = "Wild-type residue",
  `aa2` = "Mutated residue"
)

models_labels <- c(
  `envision` = "Envision",
  `ev_epistatic` = "EVmutation",
  `lopo_models_linear` = "Linear regression",
  `lopo_models_xgb` = "Gradient boosted trees",
  `naive` = "Naive",
  `by_position` = "By position"
)

# color palette and plots constants
###############################################################################
okabe <-
  c(
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7"
  )
bologna_red <- "#bb2e29"
bologna_ligh_red <- "#FF3838"
bologna_yellow <- "#FFFD52"
bologna_cyan <- "#1FA8FF"
bologna_gray <- "#808080"
bologna_palette <- c(bologna_gray, bologna_ligh_red, bologna_cyan, bologna_yellow)
boxplot_gray <- "gray15"
barplot_gray <- "gray30"
text_size <- 7 # for plots
latex_textwidth <- 5.78851 # this is the \textwidth of my latex thesis
latex_textheight <- 8.1866 # this is the \textheight of my latex thesis
latex_beamer_width <- 4.76357
latex_beamer_height <- 3.62505

# optimal parameters
###############################################################################

selected_param_values_naive <- c(
  alpha = 0,
  lambda = 4,
  gamma = 0.1,
  eta = 0.01,
  num_rounds = 1000,
  subsample = 0.6,
  `colsample_bytree` = 1,
  `min_child_weight` = 0,
  `max_depth` = 6
)

selected_param_values_by_position <- c(
  alpha = 0.1,
  lambda = 1,
  gamma = 0.1,
  eta = 0.01,
  subsample = 0.5,
  `colsample_bytree` = 0.25,
  `min_child_weight` = 2,
  `max_depth` = 6,
  `num_rounds` = 1000
)

selected_param_values_lopo_xgb <- c(
  alpha = 0,
  lambda = 1,
  gamma = 0,
  eta = 1e-4,
  num_rounds = 1000,
  subsample = 0.2,
  `colsample_bytree` = 0.4,
  `min_child_weight` = 0,
  `max_depth` = 6
)

# ggplot utils
###############################################################################

dms_facet_free_x <- facet_wrap(
  facets = vars(dms_id),
  labeller = labeller(dms_id = as_labeller(dms_dataset_labels)),
  scales = "free_x"
)

dms_facet_free <- facet_wrap(
  facets = vars(dms_id),
  labeller = labeller(dms_id = as_labeller(dms_dataset_labels)),
  scales = "free"
)

dms_facet <- facet_wrap(
  facets = vars(dms_id),
  labeller = labeller(dms_id = as_labeller(dms_dataset_labels))
)

facet_theme <- theme(panel.spacing.y = unit(1, "lines"))

x_axis_theme <- theme(axis.title.x = element_text(
  face = "bold",
  margin = margin(10, 0, 0, 0)
))

y_axis_theme <- theme(axis.title.y = element_text(
  face = "bold",
  margin = margin(0, 10, 0, 0)
))

latex_y_safe <- scale_y_discrete(
  labels = rev(dms_dataset_labels),
  limits = rev
)
