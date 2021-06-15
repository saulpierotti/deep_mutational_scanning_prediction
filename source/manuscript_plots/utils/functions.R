#!/usr/bin/Rscript
# Functions
#
# Author: Saul Pierotti
# Mail: saulpierotti.bioinfo@gmail.com
# Last updated: 13/05/2021
#
# This script contains functions that are used in various script for the
# generation of manuscript plots
# Paths are relative to the parent folder.
###############################################################################
library("tidyverse")
library("tikzDevice")
library("cowplot")
library("scales")
library("rlang")
library("caret")
library("circular")
library("Directional")
library("reshape2")

source("./utils/constants.R")

# plots for the manuscript
###############################################################################
create_tikz <- function(plot,
                        filename,
                        height = 3,
                        width = latex_textwidth) {
  tikz_dir <- "./tikz"
  tikzDevice::tikz(
    file = paste(tikz_dir, filename, sep = "/"),
    engine = "luatex",
    width = width,
    height = height
  )
  print(plot)
  dev.off()
}

# data processing
###############################################################################

normalize_scores <- function(my_tb) {
  out_tb <- my_tb %>%
    group_by(dms_id) %>%
    mutate(score = (rescale(reported_fitness) -
      rescale(0, from = c(
        min(reported_fitness),
        max(reported_fitness)
      )))) %>%
    ungroup()
  return(out_tb)
}

get_test_result_tb <- function(filepath) {
  test_result_tb <- read_csv(
    filepath,
    col_types = cols(
      aa1 = readr::col_factor(levels = residues),
      aa2 = readr::col_factor(levels = residues),
      position = "i",
      dms_id = readr::col_factor(levels = datasets),
      y_pred = "d",
      y_true = "d",
      kind = "f"
    )
  )
  return(test_result_tb)
}

get_test_correlations <- function(test_result_tb) {
  test_correlations <- test_result_tb %>%
    group_by(dms_id, kind) %>%
    mutate(pearson = cor(y_true, y_pred, method = "pearson")) %>%
    mutate(spearman = cor(y_true, y_pred, method = "spearman")) %>%
    mutate(kendall = cor(y_true, y_pred, method = "kendall")) %>%
    ungroup() %>%
    subset(select = c(dms_id, kind, pearson, spearman, kendall)) %>%
    distinct() %>%
    pivot_longer(cols = c(pearson, spearman, kendall)) %>%
    mutate(name = factor(name,
      levels = c("pearson", "spearman", "kendall")
    )) %>%
    mutate(kind = recode_factor(kind,
      `valid` = "validation",
      `validation` = "validation",
      `test` = "test"
    ))
  return(test_correlations)
}

get_random_search_results_tb_xgb <- function(filepath) {
  random_search_results_tb <- read_csv(filepath,
    col_types = cols(
      .default = "d",
      tree_method = "f",
      objective = "f"
    )
  )
  return(random_search_results_tb)
}

get_random_search_results_tb_svm <- function(filepath) {
  random_search_results_tb <- read_csv(filepath,
    col_types = cols(
      .default = "d",
      kernel = "f",
      C = "d",
      degree = "d",
      epsilon = "d",
      gamma = "d"
    )
  )
  return(random_search_results_tb)
}

get_corr_values <- function(tb, column1, column2) {
  column1 <- enquo(column1)
  column2 <- enquo(column2)
  tb <- tb %>%
    select(!!column1, !!column2) %>%
    distinct() %>%
    drop_na() %>%
    summarise(
      "variable1" = as_name(column1),
      "variable2" = as_name(column2),
      pearson = cor(!!column1, !!column2, method = "pearson"),
      spearman = cor(!!column1, !!column2, method = "spearman"),
      kendall = cor(!!column1, !!column2, method = "kendall")
    )
  return(tb)
}

filter_outliers <- function(tb, column) {
  column <- enquo(column)
  tb <- tb %>%
    mutate(
      low = boxplot.stats(!!column)$stats[1],
      up = boxplot.stats(!!column)$stats[5]
    ) %>%
    ungroup() %>%
    filter(!!column >= low & !!column <= up) %>%
    subset(select = -c(up, low))
  return(tb)
}

get_pssm_scores <- function(tb) {
  tb <- tb %>%
    pivot_longer(
      cols = starts_with("hmm_pssm_"),
      names_prefix = "hmm_pssm_",
      names_to = "hmm_pssm_name",
      values_to = "hmm_pssm_value"
    ) %>%
    filter(aa2 == hmm_pssm_name)
  return(tb)
}

get_categorical_sec_struct <- function(tb) {
  tb <- tb %>%
    rowid_to_column() %>%
    pivot_longer(
      cols = starts_with("netsurf_p_q8_"),
      names_to = "netsurf_q8_name",
      values_to = "netsurf_q8_proba",
      names_prefix = "netsurf_p_q8_"
    ) %>%
    mutate(netsurf_q8_name = factor(
      netsurf_q8_name,
      levels = sec_struct_q8_levels
    )) %>%
    group_by(rowid) %>%
    filter(netsurf_q8_proba == max(netsurf_q8_proba)) %>%
    dplyr::rename(netsurf_q8 = netsurf_q8_name) %>%
    pivot_longer(
      cols = starts_with("netsurf_p_q3_"),
      names_to = "netsurf_q3_name",
      values_to = "netsurf_q3_proba",
      names_prefix = "netsurf_p_q3_"
    ) %>%
    mutate(netsurf_q3_name = factor(
      netsurf_q3_name,
      levels = sec_struct_q3_levels
    )) %>%
    group_by(rowid) %>%
    filter(netsurf_q3_proba == max(netsurf_q3_proba)) %>%
    dplyr::rename(netsurf_q3 = netsurf_q3_name) %>%
    dplyr::rename(dssp_q8 = dssp_sec_struct) %>%
    ungroup() %>%
    subset(select = -c(rowid, netsurf_q8_proba)) %>%
    mutate(
      dssp_q3 = fct_recode(dssp_q8, !!!q8_to_q3_vec),
      netsurf_q3_from_q8 = fct_recode(netsurf_q8, !!!q8_to_q3_vec)
    ) %>%
    mutate(
      dssp_q3 = factor(dssp_q3, levels = sec_struct_q3_levels),
      netsurf_q3_from_q8 = factor(netsurf_q3_from_q8,
        levels = sec_struct_q3_levels
      )
    )
  return(tb)
}

get_categorical_metrics <- function(tb, pred_col, true_col) {
  pred_col <- enquo(pred_col)
  true_col <- enquo(true_col)
  tb <- tb %>%
    select(!!pred_col, !!true_col, dms_id, position) %>%
    distinct() %>%
    drop_na()
  pred_series <- tb %>% pull(!!pred_col)
  true_series <- tb %>% pull(!!true_col)
  conf_mat <- confusionMatrix(data = pred_series, reference = true_series)
  return(conf_mat)
}

get_categorical_continuous_metrics <- function(tb,
                                               column_continuous,
                                               column_categorical) {
  column_continuous <- enquo(column_continuous)
  column_categorical <- enquo(column_categorical)
  tb <- tb %>%
    select(!!column_continuous, !!column_categorical, dms_id, position) %>%
    distinct() %>%
    drop_na()
  continuous_series <- tb %>% pull(!!column_continuous)
  categorical_series <- tb %>% pull(!!column_categorical)
  results <- kruskal.test(continuous_series ~ categorical_series)
  return(results)
}

get_circular_corr <- function(tb, column1, column2) {
  column1 <- enquo(column1)
  column2 <- enquo(column2)
  tb <- tb %>%
    select(dms_id, position, !!column1, !!column2) %>%
    drop_na() %>%
    distinct() %>%
    mutate(
      var1_circular = circular(!!column1, units = "degrees"),
      var2_circular = circular(!!column2, units = "degrees")
    )
  vec1 <- tb %>% pull(var1_circular)
  vec2 <- tb %>% pull(var2_circular)
  corr <- cor.circular(vec1, vec2)
  return(corr)
}

get_circular_linear_corr <- function(tb, circular, linear) {
  circular <- enquo(circular)
  linear <- enquo(linear)
  tb <- tb %>%
    select(dms_id, position, !!circular, !!linear) %>%
    drop_na() %>%
    distinct() %>%
    mutate(var_circular = circular(!!circular, units = "degrees"))
  circular_vec <- tb %>% pull(var_circular)
  linear_vec <- tb %>% pull(!!linear)
  corr <- circlin.cor(circular_vec, linear_vec)
  return(corr)
}

get_top_L_stats <- function(in_tb) {
  get_top_L_stats_single <- function(tb, n, range, thr) {
    tb <- tb %>%
      drop_na() %>%
      group_by(dms_id) %>%
      mutate(L = max(position_1) - min(position_1) + 1) %>%
      filter(position_1 - position_2 >= range) %>%
      mutate(topL_n_tot = floor(L / n)) %>%
      arrange(tr_rosetta_distance) %>%
      mutate(n_contact = row_number()) %>%
      filter(n_contact <= topL_n_tot) %>%
      mutate(topL_n_true = sum(experimental_distance <= thr)) %>%
      ungroup() %>%
      summarise(dms_id, topL_n_perc = topL_n_true / topL_n_tot) %>%
      distinct() %>%
      summarise(
        dms_id,
        n = n,
        range = range,
        topL_n_perc
      )
    return(tb)
  }
  n_list <- c(5, 2, 1)
  range_list <- c(24, 12)
  thr <- 8
  stats_tb <- tibble()
  for (n in n_list) {
    for (range in range_list) {
      stats_tb <- stats_tb %>%
        bind_rows(get_top_L_stats_single(in_tb, n, range, thr))
    }
  }
  stats_tb <- stats_tb %>%
    pivot_wider(
      names_from = c(n, range),
      values_from = topL_n_perc,
      names_prefix = "top_L/",
      names_sep = ", s<="
    ) %>%
    relocate(
      "dms_id",
      "top_L/5, s<=12",
      "top_L/2, s<=12",
      "top_L, s<=12" = "top_L/1, s<=12",
      "top_L/5, s<=24",
      "top_L/2, s<=24",
      "top_L, s<=24" = "top_L/1, s<=24"
    )
  return(stats_tb)
}

# Plots
###############################################################################

get_corr_plot <- function(tb, column1, column2, color = NULL) {
  column1 <- enquo(column1)
  column2 <- enquo(column2)
  color <- enquo(color)
  plot <-
    ggplot(
      data = tb,
      mapping = aes(x = !!column1, y = !!column2)
    )
  if (!quo_is_null(color)) {
    plot <- plot +
      geom_point(
        data = tb %>% filter(!(!!color)),
        color = "black",
        shape = 4,
        alpha = 0.5
      ) +
      geom_point(
        data = tb %>% filter(!!color),
        color = okabe[1],
        shape = 4
      )
  }
  else {
    plot <- plot +
      geom_point(
        color = "black",
        shape = 4,
        alpha = 0.5
      )
  }
  plot <- plot +
    linear_regression_line +
    theme_cowplot(text_size) +
    x_axis_theme +
    y_axis_theme
  return(plot)
}

get_categorical_heatmap_plot <- function(tb, column1, column2) {
  column1 <- enquo(column1)
  column2 <- enquo(column2)
  tb <- tb %>%
    select(!!column1, !!column2, dms_id, position) %>%
    distinct() %>%
    drop_na() %>%
    group_by(!!column1, !!column2, .drop = FALSE) %>%
    summarise(count = n())

  plot <- ggplot(data = tb, mapping = aes(x = !!column1, y = !!column2)) +
    geom_tile(aes(fill = count)) +
    geom_text(aes(label = count)) +
    theme_minimal(text_size) +
    x_axis_theme +
    y_axis_theme +
    theme(legend.position = "none") +
    scale_fill_gradient(low = "white", high = "gray20")
  return(plot)
}

get_score_ss_violin_plot <- function(tb, dssp_netsurf, q3_q8, fill = "gray60") {
  tb <- tb %>%
    group_by(dms_id, position) %>%
    mutate(score_median = median(score)) %>%
    ungroup() %>%
    get_categorical_sec_struct() %>%
    select(dms_id, score_median, dssp_q3, netsurf_q3, dssp_q8, netsurf_q8) %>%
    drop_na() %>%
    pivot_longer(
      cols = c(dssp_q3, netsurf_q3, dssp_q8, netsurf_q8),
      names_to = "ss_name", values_to = "ss_value"
    ) %>%
    separate(ss_name, sep = "_", into = c("source", "qn")) %>%
    distinct()

  plot <- ggplot(data = tb %>%
    filter(qn == q3_q8 & source == dssp_netsurf)) +
    geom_violin(mapping = aes(x = ss_value, y = score_median), fill = fill)
  return(plot)
}

get_random_search_results_plot_xgb <- function(random_search_results_tb,
                                               selected_param_values) {
  hyperparam_labeller <- as_labeller(
    c(
      `alpha_log_scale` = "L1 regularization ($\\log_{10}$)",
      `alpha_normal_scale` = "L1 regularization",
      `lambda_log_scale` = "L2 regularization ($\\log_{10}$)",
      `gamma_log_scale` = "Minimum loss reduction ($\\log_{10}$)",
      `eta_log_scale` = "Learning rate ($\\log_{10}$)",
      `eta_normal_scale` = "Learning rate",
      `num_rounds_log_scale` = "Number of iterations ($\\log_{10}$)",
      `num_rounds_normal_scale` = "Number of iterations",
      `subsample_normal_scale` = "Fraction of subsampled rows",
      `colsample_bytree_normal_scale` = "Fraction of subsampled columns",
      `min_child_weight_normal_scale` = "Minimum child weight",
      `max_depth_log_scale` = "Maximum tree depth ($\\log_{10}$)"
    )
  )

  curr_tb <- random_search_results_tb %>%
    pivot_longer(
      cols = all_of(xgb_hyperparameters),
      names_to = "parameter_name",
      values_to = "parameter_value"
    ) %>%
    mutate(round_index = row_number()) %>%
    subset(select = c(
      round_index, parameter_name, parameter_value, average, sem
    )) %>%
    distinct() %>%
    group_by(parameter_name, parameter_value) %>%
    summarise(round_index, avg_max = max(average)) %>%
    group_by(parameter_name) %>%
    mutate(
      selected_parameter_value = selected_param_values[parameter_name]
    ) %>%
    ungroup() %>%
    pivot_longer(
      cols = c(parameter_value, selected_parameter_value),
      names_to = "value_kind",
      values_to = "parameter_value"
    ) %>%
    mutate(log_scale = log10(parameter_value)) %>%
    dplyr::rename(normal_scale = parameter_value) %>%
    pivot_longer(
      cols = c(normal_scale, log_scale),
      names_to = "scale",
      values_to = "value_temp"
    ) %>%
    filter(
      (parameter_name == "alpha" & scale == "log_scale") |
        (parameter_name == "alpha" & scale == "normal_scale") |
        (parameter_name == "colsample_bytree" &
          scale == "normal_scale") |
        (parameter_name == "gamma" & scale == "log_scale") |
        (parameter_name == "lambda" & scale == "log_scale") |
        (parameter_name == "max_depth" & scale == "log_scale") |
        (parameter_name == "subsample" &
          scale == "normal_scale") |
        (parameter_name == "min_child_weight" &
          scale == "normal_scale") |
        (parameter_name == "eta" &
          scale == "log_scale")
    ) %>%
    pivot_wider(names_from = value_kind, values_from = value_temp) %>%
    unite(parameter_name, c(parameter_name, scale)) %>%
    mutate(parameter_name = as_factor(parameter_name)) %>%
    mutate(parameter_name = factor(
      parameter_name,
      levels = c(
        "alpha_normal_scale",
        "alpha_log_scale",
        "eta_log_scale",
        "subsample_normal_scale",
        "lambda_log_scale",
        "colsample_bytree_normal_scale",
        "gamma_log_scale",
        "min_child_weight_normal_scale",
        "max_depth_log_scale"
      )
    ))

  plot <- ggplot(data = curr_tb) +
    geom_point(
      mapping = aes(x = parameter_value, y = avg_max),
      size = 0.8
    ) +
    geom_line(
      mapping = aes(x = parameter_value, y = avg_max),
      linetype = "dashed"
    ) +
    geom_vline(aes(xintercept = selected_parameter_value), color = okabe[1]) +
    theme_cowplot(text_size) +
    facet_wrap(
      facets = ~parameter_name,
      scales = "free",
      labeller = hyperparam_labeller
    ) +
    xlab("Hyperparameter value") +
    ylab("Maximum average Pearson r") +
    x_axis_theme +
    y_axis_theme

  return(plot)
}

get_learning_curves_plot_xgb <-
  function(random_search_results_tb,
           selected_param_values) {
    curr_tb <- random_search_results_tb %>%
      mutate(iteration = row_number()) %>%
      pivot_longer(
        cols = all_of(xgb_hyperparameters),
        names_to = "parameter_name",
        values_to = "parameter_value"
      ) %>%
      subset(select = c(
        iteration,
        parameter_name,
        parameter_value,
        average,
        sem
      )) %>%
      group_by(parameter_name) %>%
      mutate(
        selected_parameter_value = selected_param_values[parameter_name]
      ) %>%
      group_by(iteration) %>%
      filter(
        parameter_value == selected_parameter_value |
          parameter_name == "num_rounds" | parameter_name == "eta"
      ) %>%
      filter(n() == length(xgb_hyperparameters)) %>%
      filter(parameter_name == "num_rounds" |
        parameter_name == "eta") %>%
      subset(select = -c(selected_parameter_value)) %>%
      pivot_wider(
        names_from = parameter_name,
        values_from = parameter_value
      ) %>%
      subset(select = -c(iteration)) %>%
      mutate(
        eta_recoded = recode_factor(
          eta,
          `1e-4` = "$1 \\cdot 10^{-4}$",
          `0.001` = "$1 \\cdot 10^{-3}$",
          `0.005` = "$5 \\cdot 10^{-3}$",
          `0.01` = "$1 \\cdot 10^{-2}$",
          `0.05` = "$5 \\cdot 10^{-2}$",
          `0.1` = "$1 \\cdot 10^{-1}$",
          `0.5` = "$5 \\cdot 10^{-1}$",
          `1` = "$1$"
        )
      ) %>%
      mutate(eta_factor = factor(eta))

    plot <-
      ggplot(
        data = curr_tb,
        mapping = aes(
          x = log10(num_rounds),
          y = average,
          color = eta_recoded
        )
      ) +
      geom_line(linetype = "dashed") +
      geom_point() +
      scale_x_continuous(
        labels = c(
          "$1$", "$10$", "$10^2$", "$10^3$", "$10^4$", "$10^5$"
        ),
        breaks = c(0, 1, 2, 3, 4, 5)
      ) +
      theme_cowplot(text_size) +
      scale_color_manual(values = c("black", okabe), name = "Learning rate") +
      xlab("Number of iterations") +
      ylab("Average Pearson r") +
      x_axis_theme +
      y_axis_theme

    return(plot)
  }

get_test_result_by_dataset_plot <- function(test_result_tb) {
  plot <- ggplot(data = test_result_tb, mapping = aes(x = y_true, y = y_pred)) +
    geom_point(
      data = filter(test_result_tb, (aa1 != aa2) & (kind == "test")),
      mapping = aes(color = "gray15"),
      alpha = 0.5,
      shape = 4
    ) +
    geom_point(
      data = filter(test_result_tb, (aa1 == aa2) & (kind == "test")),
      mapping = aes(color = okabe[1]),
      shape = 4
    ) +
    xlab("Experimental fitenss score") +
    ylab("Predicted fitness score") +
    scale_color_identity(
      name = "Mutation type",
      labels = c("Missense", "Synonymous"),
      guide = "legend",
      limits = rev
    ) +
    theme_cowplot(text_size) +
    guides(colour = guide_legend(override.aes = list(alpha = 1, stroke = 1))) +
    x_axis_theme +
    y_axis_theme +
    theme(legend.position = "none") +
    dms_facet_free
  return(plot)
}

get_test_result_summary_plot <- function(test_result_tb,
                                         correlation_type = "pearson",
                                         show_numbers = TRUE,
                                         text_xshift = 0.25,
                                         text_yshift = 0.04) {
  curr_tb <- get_test_correlations(test_result_tb)
  correlation_label <- ifelse(correlation_type == "pearson",
    "Pearson correlation coefficient",
    "NULL"
  )
  correlation_label <- ifelse(correlation_type == "spearman",
    "Spearman correlation coefficient",
    correlation_label
  )


  plot <- ggplot(data = curr_tb %>% filter(name == correlation_type)) +
    geom_point(aes(x = dms_id, y = value, color = kind)) +
    ylim(c(0, 1)) +
    theme_cowplot(text_size) +
    xlab("Dataset") +
    ylab(correlation_label) +
    y_axis_theme +
    x_axis_theme +
    scale_x_discrete(labels = dms_dataset_labels) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_color_manual(values = c(okabe[1], "black")) +
    theme(legend.position = "none")

  if (show_numbers) {
    plot <- plot +
      geom_text(
        data = filter(curr_tb, name == correlation_type & kind == "test"),
        aes(
          x = as.numeric(dms_id) + text_xshift,
          y = value + text_yshift,
          label = sprintf("%.2f", value)
        ),
        size = 2.5
      )
  }

  return(plot)
}

get_feature_importance_by_dataset_plot <- function(feature_importance_tb) {
  plot <- ggplot(data = feature_importance_tb, mapping = aes(
    x = importance_average,
    y = fct_reorder(feature_group, importance_average, )
  )) +
    geom_bar(stat = "identity") +
    dms_facet_free_x +
    theme_minimal_vgrid(text_size) +
    xlab("Permutation importance") +
    ylab("Feature group") +
    x_axis_theme +
    y_axis_theme +
    scale_y_discrete(labels = feature_group_labels)
  return(plot)
}

get_performance_comparison_plot <- function(comparison_tb) {
  plot <- ggplot(
    data = comparison_tb,
    mapping = aes(x = dms_id, y = correlation, color = model)
  ) +
    geom_jitter(width = 0.1, height = 0) +
    ylim(c(-0.1, 0.93)) +
    theme_cowplot(text_size) +
    xlab("Dataset") +
    theme(strip.background = element_blank()) +
    x_axis_theme +
    y_axis_theme +
    scale_x_discrete(labels = dms_dataset_labels) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_color_manual(values = c("black", okabe), name = "Model", labels = models_labels)
  return(plot)
}
