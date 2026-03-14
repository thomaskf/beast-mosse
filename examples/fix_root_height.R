#!/usr/bin/env Rscript
# fix_root_height.R
#
# Usage:
#   Rscript fix_root_height.R input.xml [output.xml]
#
# If output.xml is not specified, the input file is overwritten in-place.
#
# What it does:
#   1. Reads the newick string from the <tree ... newick='...'> tag in the XML
#   2. Computes the root height from the tree (= max tip-to-root distance)
#   3. Replaces the value of <parameter name="mean" ...> ONLY inside the
#      <distr id="Normal.rootHeight" ...> ... </distr> block
#   4. Writes the corrected XML to the output file

library(ape)

# ── argument handling ────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript fix_root_height.R input.xml [output.xml]")
}

input_xml  <- args[1]
output_xml <- if (length(args) >= 2) args[2] else input_xml

if (!file.exists(input_xml)) {
  stop(paste("Input file not found:", input_xml))
}

cat("Input XML : ", input_xml,  "\n")
cat("Output XML: ", output_xml, "\n\n")

# ── read XML: keep line-by-line AND as single string ────────────────────────
xml_lines <- readLines(input_xml, warn = FALSE)
xml_text  <- paste(xml_lines, collapse = "\n")

# ── extract the newick string ────────────────────────────────────────────────
newick_pattern <- "newick=['\"]([^'\"]+)['\"]"
newick_match   <- regmatches(xml_text, regexpr(newick_pattern, xml_text, perl = TRUE))

if (length(newick_match) == 0 || newick_match == "") {
  stop("Could not find a newick='...' attribute in the XML file.")
}

newick_str <- sub(newick_pattern, "\\1", newick_match, perl = TRUE)
cat("Newick string found (first 80 chars):\n")
cat(substr(newick_str, 1, 80), "...\n\n")

# ── parse the tree and compute root height ───────────────────────────────────
tree <- tryCatch(
  read.tree(text = newick_str),
  error = function(e) stop(paste("Failed to parse newick string:", e$message))
)

tip_distances <- node.depth.edgelength(tree)
tip_only      <- tip_distances[1:length(tree$tip.label)]
root_height   <- max(tip_only)

cat(sprintf("Number of tips      : %d\n",     length(tree$tip.label)))
cat(sprintf("Computed root height: %.10f\n\n", root_height))

# ── locate the Normal.rootHeight block using line indices ────────────────────
# Find the line containing the opening tag
open_tag_pattern  <- 'id="Normal\\.rootHeight"'
close_tag_pattern <- '</distr>'

open_line <- grep(open_tag_pattern, xml_lines)
if (length(open_line) == 0) {
  stop('Could not find a line containing id="Normal.rootHeight" in the XML.')
}
if (length(open_line) > 1) {
  warning(sprintf("Found %d lines matching Normal.rootHeight; using the first.", length(open_line)))
  open_line <- open_line[1]
}

# Find the first </distr> that comes after the opening tag line
close_candidates <- grep(close_tag_pattern, xml_lines)
close_line <- close_candidates[close_candidates > open_line][1]
if (is.na(close_line)) {
  stop("Could not find closing </distr> tag after the Normal.rootHeight opening tag.")
}

cat(sprintf("Normal.rootHeight block spans lines %d to %d:\n", open_line, close_line))
cat(paste(xml_lines[open_line:close_line], collapse = "\n"), "\n\n")

# ── replace mean value ONLY on lines within the block ───────────────────────
mean_pattern <- '(<parameter\\s+name="mean"\\s+spec="parameter\\.RealParameter"\\s+value=")[^"]*("\\s+estimate="false"\\s*/>)'

new_value  <- sprintf("%.10f", root_height)
block_lines <- xml_lines[open_line:close_line]

if (!any(grepl(mean_pattern, block_lines, perl = TRUE))) {
  stop('Could not find <parameter name="mean" ...> inside the Normal.rootHeight block.')
}

updated_block_lines <- gsub(mean_pattern,
                             paste0("\\1", new_value, "\\2"),
                             block_lines,
                             perl = TRUE)

cat("Updated Normal.rootHeight block:\n")
cat(paste(updated_block_lines, collapse = "\n"), "\n\n")

# ── splice updated lines back into the full XML ──────────────────────────────
xml_lines_new <- c(
  xml_lines[seq_len(open_line - 1)],
  updated_block_lines,
  xml_lines[seq(close_line + 1, length(xml_lines))]
)

# ── sanity check ─────────────────────────────────────────────────────────────
if (!any(grepl(new_value, xml_lines_new, fixed = TRUE))) {
  stop("Replacement did not produce the expected value in the output. Aborting.")
}

# ── write output ─────────────────────────────────────────────────────────────
writeLines(xml_lines_new, output_xml)

cat(sprintf("Root height prior mean updated to: %s\n", new_value))
cat(sprintf("Corrected XML written to: %s\n",          output_xml))
