#!/usr/bin/env python3
"""Emit a per-dataset summary JSON for one (seq, tree, trait) replicate.

Computes:
  - taxa: list of taxon names from the FASTA (in file order)
  - n_taxa, n_sites
  - base_freqs: observed ACGT frequencies (lowercase fasta), normalised
  - newick: cleaned newick string (single line, trailing newline stripped)
  - root_height: maximum root-to-tip distance
  - traits: dict taxon -> trait value (matched against the FASTA taxon set)
  - trait_min, trait_max
  - internal_clades: list of {node_label, taxa: [...] } for every non-root
    internal node, derived from the parsed newick (used to build logger-only
    MRCAPrior blocks in the BEAST XML)

Usage:
    summarize_dataset.py <seq.fasta> <tree.nwk> <trait.csv> [--out file.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# FASTA
# ---------------------------------------------------------------------------

def parse_fasta(path: Path) -> list[tuple[str, str]]:
    seqs: list[tuple[str, str]] = []
    name: str | None = None
    buf: list[str] = []
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs.append((name, "".join(buf)))
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if name is not None:
            seqs.append((name, "".join(buf)))
    return seqs


def base_freqs(seqs: list[tuple[str, str]]) -> dict[str, float]:
    counts = {b: 0 for b in "acgt"}
    total = 0
    for _, s in seqs:
        for ch in s.lower():
            if ch in counts:
                counts[ch] += 1
                total += 1
    if total == 0:
        raise ValueError("no ACGT characters found in alignment")
    return {b: counts[b] / total for b in "acgt"}


# ---------------------------------------------------------------------------
# Trait CSV
# ---------------------------------------------------------------------------

def parse_traits(path: Path) -> dict[str, float]:
    traits: dict[str, float] = {}
    with path.open() as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # Skip header
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        name = parts[0].strip().strip('"')
        try:
            val = float(parts[1])
        except ValueError:
            continue
        traits[name] = val
    return traits


# ---------------------------------------------------------------------------
# Newick parsing
# ---------------------------------------------------------------------------

class TreeNode:
    __slots__ = ("label", "branch", "children", "parent")

    def __init__(self, label: str = "", branch: float = 0.0):
        self.label = label
        self.branch = branch
        self.children: list[TreeNode] = []
        self.parent: TreeNode | None = None

    def add(self, child: TreeNode) -> None:
        child.parent = self
        self.children.append(child)

    def is_leaf(self) -> bool:
        return not self.children


def parse_newick(text: str) -> TreeNode:
    text = text.strip().rstrip(";").strip() + ";"
    pos = 0
    n = len(text)
    root = TreeNode()
    cur = root
    # Stack of parent nodes: when we see '(' the current node becomes a
    # parent and we descend into its first child; when we see ',' we add
    # a new sibling under the same parent; when we see ')' we pop back.
    stack: list[TreeNode] = []

    while pos < n:
        ch = text[pos]
        if ch == "(":
            stack.append(cur)
            child = TreeNode()
            cur.add(child)
            cur = child
            pos += 1
        elif ch == ",":
            if not stack:
                raise ValueError(f"unexpected ',' at pos {pos}")
            child = TreeNode()
            stack[-1].add(child)
            cur = child
            pos += 1
        elif ch == ")":
            if not stack:
                raise ValueError(f"unexpected ')' at pos {pos}")
            cur = stack.pop()
            pos += 1
            m = re.match(r"([^(),:;]+)", text[pos:])
            if m:
                cur.label = m.group(1).strip()
                pos += len(m.group(0))
            if pos < n and text[pos] == ":":
                m2 = re.match(r":([-+0-9eE.]+)", text[pos:])
                if not m2:
                    raise ValueError(f"bad branch length at pos {pos}")
                cur.branch = float(m2.group(1))
                pos += len(m2.group(0))
        elif ch == ";":
            break
        elif ch.isspace():
            pos += 1
        else:
            m = re.match(r"([^(),:;]+)", text[pos:])
            if not m:
                raise ValueError(f"unexpected token at pos {pos}: {text[pos:pos+10]!r}")
            cur.label = m.group(1).strip()
            pos += len(m.group(0))
            if pos < n and text[pos] == ":":
                m2 = re.match(r":([-+0-9eE.]+)", text[pos:])
                if not m2:
                    raise ValueError(f"bad branch length at pos {pos}")
                cur.branch = float(m2.group(1))
                pos += len(m2.group(0))
    return root


def iter_internal(node: TreeNode):
    if not node.is_leaf():
        yield node
        for c in node.children:
            yield from iter_internal(c)


def leaves_under(node: TreeNode) -> list[str]:
    if node.is_leaf():
        return [node.label]
    out: list[str] = []
    for c in node.children:
        out.extend(leaves_under(c))
    return out


def root_height(node: TreeNode) -> float:
    # max root-to-tip distance
    best = 0.0
    stack = [(node, 0.0)]
    while stack:
        n, d = stack.pop()
        if n.is_leaf():
            if d > best:
                best = d
        else:
            for c in n.children:
                stack.append((c, d + c.branch))
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta")
    ap.add_argument("tree")
    ap.add_argument("trait")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fasta_path = Path(args.fasta)
    tree_path = Path(args.tree)
    trait_path = Path(args.trait)

    seqs = parse_fasta(fasta_path)
    taxa = [n for n, _ in seqs]
    n_sites = len(seqs[0][1]) if seqs else 0
    bf = base_freqs(seqs)

    traits_raw = parse_traits(trait_path)
    # keep only those present in the FASTA, in FASTA order
    traits = {t: traits_raw[t] for t in taxa if t in traits_raw}
    missing = [t for t in taxa if t not in traits_raw]
    if missing:
        print(f"WARN: missing traits for {missing}", file=sys.stderr)

    newick_text = tree_path.read_text().strip()
    # Strip any inline FigTree blocks
    newick_text = re.sub(r"\[[^\]]*\]", "", newick_text)
    newick_text = newick_text.replace("\n", "")
    root = parse_newick(newick_text)

    # Validate taxon set match
    leaves = leaves_under(root)
    if set(leaves) != set(taxa):
        only_t = set(taxa) - set(leaves)
        only_n = set(leaves) - set(taxa)
        print(f"WARN: taxon mismatch  fasta_only={only_t}  newick_only={only_n}",
              file=sys.stderr)

    h = root_height(root)

    # Internal clades (skip the root itself; we already have a global rootHeight prior)
    internal_clades = []
    for nd in iter_internal(root):
        if nd is root:
            continue
        lbl = nd.label or f"int{len(internal_clades)+1}"
        internal_clades.append({"label": lbl, "taxa": leaves_under(nd)})

    summary = {
        "fasta": str(fasta_path),
        "tree": str(tree_path),
        "trait": str(trait_path),
        "n_taxa": len(taxa),
        "n_sites": n_sites,
        "taxa": taxa,
        "base_freqs": bf,
        "newick": newick_text.rstrip(";") + ";",
        "root_height": h,
        "traits": traits,
        "trait_min": min(traits.values()) if traits else None,
        "trait_max": max(traits.values()) if traits else None,
        "internal_clades": internal_clades,
    }

    txt = json.dumps(summary, indent=2)
    if args.out:
        Path(args.out).write_text(txt + "\n")
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
