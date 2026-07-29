#!/usr/bin/env Rscript
# Compute ape::birthdeath() estimates for the 10 sim6 true trees and write a
# JSON summary keyed by seed. Used to seed curveMaxY (lambda_BD) and yV (mu_BD)
# in the MoSSE XMLs.
#
# Usage:
#   Rscript compute_bd.R <sim_dir> <out_json>

suppressPackageStartupMessages({
  library(ape)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
sim_dir <- if (length(args) >= 1) args[1] else
  "/Users/thomaswong/workplaces/mosse_no_punc/simulations6_no_punc"
out_json <- if (length(args) >= 2) args[2] else
  "/Users/thomaswong/workplaces/mosse_no_punc/strict_clock/bd_estimates.json"

# Generalized: match tree<num>_<tag>_<seed>.tre for any epsilon/sig tag (0.2, 0.001, ...)
tree_files <- list.files(sim_dir, pattern = "^tree\\d+_[0-9.]+_\\d+\\.tre$",
                          full.names = TRUE)
result <- list()
for (tf in tree_files) {
  base <- sub("\\.tre$", "", basename(tf))
  seed <- as.integer(sub("^tree\\d+_[0-9.]+_", "", base))  # trailing seed
  tr <- read.tree(tf)
  bd <- tryCatch(birthdeath(tr), error = function(e) e)
  if (inherits(bd, "error")) {
    result[[as.character(seed)]] <- list(seed = seed, error = conditionMessage(bd))
    next
  }
  d_over_b   <- as.numeric(bd$para[1])  # d/b
  b_minus_d  <- as.numeric(bd$para[2])  # b-d
  speciation <- b_minus_d / (1 - d_over_b)
  extinction <- speciation * d_over_b
  result[[as.character(seed)]] <- list(
    seed = seed,
    tree_file = tf,
    n_tips = length(tr$tip.label),
    root_height = max(node.depth.edgelength(tr)[1:length(tr$tip.label)]),
    d_over_b = d_over_b,
    b_minus_d = b_minus_d,
    lambda_BD = speciation,
    mu_BD = extinction,
    bd_log_lik = as.numeric(bd$dev) * -0.5
  )
  cat(sprintf("seed=%-3d  lambda=%.6f  mu=%.6f  (d/b=%.4f, b-d=%.4f)\n",
              seed, speciation, extinction, d_over_b, b_minus_d))
}

write_json(result, out_json, auto_unbox = TRUE, pretty = TRUE)
cat(sprintf("wrote %s\n", out_json))
