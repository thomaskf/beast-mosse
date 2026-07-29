#!/usr/bin/env Rscript
# Driver for the sim8 no-punctuation MoSSE simulation (beta=1, sig/epsilon=0.2, 2000-bp).
# Portable: cwd = this script's own directory (works wherever the repo is cloned).
.args <- commandArgs(trailingOnly = FALSE)
.self <- sub("^--file=", "", .args[grep("^--file=", .args)])
if (length(.self) > 0) setwd(dirname(normalizePath(.self)))
cat("cwd:", getwd(), "\n")
source("tree.mosse.withpunc.R")
source("simSeq.mosse.withpunc.R")
source("prune.mosse.R")

# Loop until we get a 10-unique inner-seed draw.
old_seeds <- c(1,7,26,42,48,53,62,68,73,97,99)
for (os in 20262000:20263000) {
  set.seed(os)
  rexp(1, 0.15); LaplacesDemon::rdirichlet(1, alpha=c(1,1,1,1))
  cand <- round(runif(10, 1, 100))
  if (length(unique(cand)) == 10 && !any(cand %in% old_seeds)) {
    cat("Picked outer seed:", os, " ->", cand, "\n")
    chosen <- os
    break
  }
}
set.seed(chosen)
source("main.nopunc.R")
cat("\nSeeds used:", seed, "\n")
cat("Files written:\n")
print(list.files(pattern = "^(seq|tree|trait)\\d+_[0-9.]+_[0-9]+\\.(tre|fasta|csv)$"))
