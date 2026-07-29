#!/usr/bin/env python3
"""Extract per-seed posterior summaries from a strict-clock BEAST 2 run.

For one (.log, .trees) pair:
  - posterior-median clockRate
  - posterior-median kappa
  - representative tree (last sample of the .trees file; with the chain length
    we use, the last sample is well past convergence)

Writes a small JSON to stdout (or `--out`).

Usage:
    extract_strict_clock.py <run_dir> [--burnin 0.2] [--out summary.json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

TREEANNOTATOR = "/Applications/BEAST 2.7.7/bin/treeannotator"


def find_one(p: Path, glob: str) -> Path:
    hits = sorted(p.glob(glob))
    if not hits:
        raise FileNotFoundError(f"no {glob} under {p}")
    if len(hits) > 1:
        raise RuntimeError(f"multiple {glob} under {p}: {hits}")
    return hits[0]


def parse_log(p: Path, burnin: float):
    header = None
    rows: list[list[str]] = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if header is None:
            header = ln.split("\t")
            continue
        parts = ln.split("\t")
        if len(parts) != len(header):
            continue
        rows.append(parts)
    if not rows:
        raise ValueError(f"empty log {p}")
    drop = int(len(rows) * burnin)
    rows = rows[drop:]
    idx = {h: i for i, h in enumerate(header)}
    def col(name):
        return [float(r[idx[name]]) for r in rows]
    return rows, header, col


def parse_trees(p: Path):
    """Return all tree strings (one per logged sample) in order."""
    txt = p.read_text()
    trees: list[str] = []
    translate: dict[str, str] = {}
    in_translate = False
    for ln in txt.splitlines():
        s = ln.strip()
        if s.lower().startswith("translate"):
            in_translate = True
            continue
        if in_translate:
            if s.endswith(";"):
                in_translate = False
                continue
            m = re.match(r"(\d+)\s+([^,;\s]+)", s.rstrip(","))
            if m:
                translate[m.group(1)] = m.group(2)
            continue
        m = re.match(r"tree\s+\S+\s*=\s*(?:\[[^\]]*\])?\s*(.+);$", s)
        if m:
            trees.append(m.group(1))
    return trees, translate


def restore_labels(newick: str, translate: dict[str, str]) -> str:
    if not translate:
        return newick
    # Replace ONLY tip indices, not branch lengths. Tip indices appear after
    # '(' or ',' and are followed by ':' or ',' or ')'.
    def sub(m):
        n = m.group(1)
        return f"{m.group(0)[0]}{translate.get(n, n)}"
    return re.sub(r"([(,])(\d+)(?=[:,)])", lambda m: f"{m.group(1)}{translate.get(m.group(2), m.group(2))}", newick)


def compute_mcc_newick(tree_file: Path, burnin_pct: int = 20) -> str | None:
    """Run BEAST treeannotator to get the MCC tree (median heights). Return
    the newick string with tip labels restored, or None on failure."""
    mcc_out = tree_file.with_suffix(".mcc.tre")
    try:
        proc = subprocess.run(
            [TREEANNOTATOR, "-burnin", str(burnin_pct), "-height", "median",
             str(tree_file), str(mcc_out)],
            capture_output=True, text=True, timeout=240,
        )
    except Exception as exc:
        sys.stderr.write(f"treeannotator subprocess exception: {exc}\n")
        return None
    if proc.returncode != 0 or not mcc_out.exists():
        sys.stderr.write(f"treeannotator failed (code={proc.returncode}): "
                          f"{proc.stderr[:400]}\n")
        return None
    # MCC nexus comes back with [&...] BEAST annotations and integer tip
    # labels via the TRANSLATE block. Reuse the same logic we already have.
    raw_trees, translate = parse_trees(mcc_out)
    if not raw_trees:
        return None
    nw = raw_trees[0]
    # Strip [&...] annotations
    nw = re.sub(r"\[&[^\]]*\]", "", nw)
    return restore_labels(nw, translate) + ";"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--burnin", type=float, default=0.2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tree-mode", choices=["mcc", "last"], default="mcc",
                    help="Representative tree source: 'mcc' (treeannotator MCC, default) "
                         "or 'last' (final sampled tree).")
    args = ap.parse_args(argv[1:])

    rd = Path(args.run_dir)
    log_file = find_one(rd, "*.log")
    tree_file = find_one(rd, "*.trees")

    rows, header, col = parse_log(log_file, args.burnin)
    if "clockRate" not in header or "kappa" not in header:
        raise KeyError(f"log missing clockRate/kappa columns: {header}")
    clock = col("clockRate")
    kappa = col("kappa")
    posterior = col("posterior")

    trees, translate = parse_trees(tree_file)
    if not trees:
        raise ValueError(f"no trees parsed from {tree_file}")
    drop = int(len(trees) * args.burnin)
    keep = trees[drop:]

    if args.tree_mode == "mcc":
        rep_tree = compute_mcc_newick(tree_file, burnin_pct=int(args.burnin * 100))
        tree_source = "mcc"
        if rep_tree is None:
            sys.stderr.write("MCC fallback: using last sample\n")
            rep_tree = restore_labels(keep[-1], translate) + ";"
            tree_source = "last_fallback"
    else:
        rep_tree = restore_labels(keep[-1], translate) + ";"
        tree_source = "last"

    out = {
        "run_dir": str(rd),
        "log_file": str(log_file),
        "tree_file": str(tree_file),
        "n_log_samples_post_burnin": len(rows),
        "n_trees_post_burnin": len(keep),
        "burnin": args.burnin,
        "clockRate_median": statistics.median(clock),
        "clockRate_mean": statistics.fmean(clock),
        "clockRate_sd": statistics.pstdev(clock),
        "kappa_median": statistics.median(kappa),
        "kappa_mean": statistics.fmean(kappa),
        "kappa_sd": statistics.pstdev(kappa),
        "posterior_median": statistics.median(posterior),
        "representative_tree_source": tree_source,
        "representative_tree_newick": rep_tree,
    }
    txt = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).write_text(txt + "\n")
    else:
        print(txt)


if __name__ == "__main__":
    main(sys.argv)
