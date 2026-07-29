#!/usr/bin/env python3
"""Generate a BEAST 2 / MoSSE XML for one no-punctuation, no-gamma replicate.

Per-parameter initial values, priors and operators follow the table in
260603_param_table.docx. Tree topology is fixed; root height is pinned via a
tight MRCAPrior on the full taxon set; internal node heights are initialised
by perturbing the true heights with Uniform(0.5, 1.5) per internal node and
renormalising so that the root height equals the true value.

Usage:
    generate_xml.py <fasta> <tree> <trait> <output_xml>

Deterministic: the per-XML perturbation seed is the integer extracted from
the seed token in the input filenames (e.g. seq1_0.001_2.fasta -> seed=2).
"""

from __future__ import annotations

import math
import random
import re
import sys
from pathlib import Path

# Re-use the parsers in summarize_dataset.py (sibling module).
sys.path.insert(0, str(Path(__file__).parent))
from summarize_dataset import (
    TreeNode,
    parse_fasta,
    parse_newick,
    parse_traits,
    base_freqs,
    iter_internal,
    leaves_under,
    root_height,
)


# ---------------------------------------------------------------------------
# Tree manipulation
# ---------------------------------------------------------------------------

def node_height(node: TreeNode) -> float:
    """Distance from the given node to the deepest descendant tip."""
    if node.is_leaf():
        return 0.0
    best = 0.0
    for c in node.children:
        h = c.branch + node_height(c)
        if h > best:
            best = h
    return best


def label_internals(root: TreeNode) -> list[TreeNode]:
    """Ensure every internal node has a unique label; return them in pre-order
    (root first)."""
    counter = [1]
    seen: set[str] = set()
    internals: list[TreeNode] = []

    def walk(n: TreeNode):
        if not n.is_leaf():
            if not n.label or n.label in seen:
                lbl = f"nd{counter[0]}"
                while lbl in seen:
                    counter[0] += 1
                    lbl = f"nd{counter[0]}"
                n.label = lbl
            seen.add(n.label)
            counter[0] += 1
            internals.append(n)
            for c in n.children:
                walk(c)
    walk(root)
    return internals


MIN_BRANCH = 0.003  # hard floor on every branch in the initial tree (matches dt in MosseDistribution)


def perturb_internal_heights(root: TreeNode, true_root_h: float,
                              rng: random.Random,
                              lo: float = 0.5, hi: float = 1.5,
                              min_branch: float = MIN_BRANCH) -> None:
    """Multiplicatively perturb every non-root internal-node height by a draw
    from Uniform(lo, hi). After perturbation, rescale all internal heights
    (leaves stay at height 0) so that the deepest root-to-tip distance equals
    `true_root_h`. Topology and tip labels are preserved.

    Every branch (parent height - child height) is held at >= `min_branch` by
    clipping during perturbation, and any branch that still drops below the
    floor after rescaling is raised to `min_branch` as a safety net.
    """
    # original heights, in pre-order
    heights: dict[int, float] = {}

    def compute_h(n: TreeNode) -> float:
        if n.is_leaf():
            heights[id(n)] = 0.0
            return 0.0
        h = 0.0
        for c in n.children:
            ch = compute_h(c)
            cand = c.branch + ch
            if cand > h:
                h = cand
        heights[id(n)] = h
        return h

    compute_h(root)

    # propose new heights for non-root internals (root stays at its true h)
    new_heights = dict(heights)
    new_heights[id(root)] = heights[id(root)]  # placeholder, rescaled below

    internals = [n for n in iter_internal(root) if n is not root]

    def parent_height(n: TreeNode) -> float:
        # find the parent height in new_heights (parents always processed
        # earlier when we walk pre-order from the root downwards)
        return new_heights[id(n.parent)]

    def max_child_height(n: TreeNode) -> float:
        return max(new_heights[id(c)] for c in n.children)

    # We need parents before children, so walk pre-order from root.
    def walk_preorder(n: TreeNode):
        for c in n.children:
            if c.is_leaf():
                continue
            factor = rng.uniform(lo, hi)
            proposed = heights[id(c)] * factor
            ph = parent_height(c)
            mch = max_child_height(c)
            # clip strictly between (mch + min_branch, ph - min_branch) so every
            # branch is at least `min_branch` after perturbation
            lo_b = mch + min_branch
            hi_b = ph - min_branch
            if hi_b <= lo_b:
                # degenerate: not enough slack for the floor; use midpoint
                proposed = 0.5 * (mch + ph)
            else:
                if proposed < lo_b:
                    proposed = lo_b
                if proposed > hi_b:
                    proposed = hi_b
            new_heights[id(c)] = proposed
            walk_preorder(c)

    walk_preorder(root)

    # rescale so root height matches the true value exactly
    cur_root = new_heights[id(root)]
    if cur_root <= 0:
        return
    scale = true_root_h / cur_root
    for n in iter_internal(root):
        new_heights[id(n)] *= scale

    # write back as branch lengths: branch_to_child = parent_h - child_h
    def write_branches(n: TreeNode):
        for c in n.children:
            ch = new_heights[id(c)] if not c.is_leaf() else 0.0
            c.branch = new_heights[id(n)] - ch
            if c.branch < min_branch:
                c.branch = min_branch
            if not c.is_leaf():
                write_branches(c)

    write_branches(root)


def newick_str(node: TreeNode, decimals: int = 10) -> str:
    if node.is_leaf():
        return f"{node.label}:{node.branch:.{decimals}f}"
    inner = ",".join(newick_str(c, decimals) for c in node.children)
    lbl = node.label or ""
    if node.branch == 0.0 and node.parent is None:
        return f"({inner}){lbl}"
    return f"({inner}){lbl}:{node.branch:.{decimals}f}"


# ---------------------------------------------------------------------------
# XML emission
# ---------------------------------------------------------------------------

XML_NS = (
    "beast.base.evolution.alignment:beast.pkgmgmt:beast.base.core:"
    "beast.base.inference:beast.base.evolution.tree.coalescent:"
    "beast.pkgmgmt:beast.base.core:beast.base.inference.util:"
    "beast.evolution.nuc:beast.base.evolution.operator:"
    "beast.base.inference.operator:beast.base.evolution.sitemodel:"
    "beast.base.evolution.substitutionmodel:beast.base.evolution.likelihood:"
    "beast.base.core"
)


def fmt_freqs(bf: dict[str, float]) -> str:
    # Round to 6 decimals, then absorb any rounding residual into the last freq
    # so BEAST's "frequencies must sum to 1" check passes.
    rounded = [round(bf[b], 6) for b in "acgt"]
    rounded[-1] = round(1.0 - sum(rounded[:-1]), 6)
    return " ".join(f"{v:.6f}" for v in rounded)


def fmt_traits(traits: dict[str, float]) -> str:
    return ",".join(f"{t}={v:.15g}" for t, v in traits.items())


def build_xml(
    taxa: list[str],
    sequences: list[tuple[str, str]],
    newick: str,
    true_root_height: float,
    traits: dict[str, float],
    bf: dict[str, float],
    internal_clades: list[tuple[str, list[str]]],
) -> str:
    lines: list[str] = []
    A = lines.append

    A(f"<beast version='2.0' namespace='{XML_NS}'>")
    A("")
    A(f"    <!-- ntax={len(taxa)} nchar={len(sequences[0][1])} -->")
    A('    <data id="alignment" dataType="nucleotide">')
    for name, seq in sequences:
        A(f'        <sequence taxon="{name}">')
        A(f"            {seq}")
        A("        </sequence>")
    A("    </data>")
    A("")

    A('    <taxonset id="taxonset" spec="beast.base.evolution.alignment.TaxonSet">')
    for t in taxa:
        A(f'        <taxon id="{t}" spec="Taxon" />')
    A("    </taxonset>")
    A("")

    A("    <!-- Topology fixed to the simulated tree; internal heights perturbed (Uniform(0.5,1.5)) and renormalised so root height = true value. -->")
    A("    <tree id='tree' spec='beast.base.evolution.tree.TreeParser' IsLabelledNewick='true'")
    A(f"        newick='{newick}'")
    A("        taxa='@alignment' />")
    A("")

    # MoSSE tip likelihood (linear scale)
    A('    <MosseTipLikelihood id="mossetiplikelihood" spec="mosse.MosseTipLikelihood" logscale="false">')
    A('        <epsilon id="epsilon" spec="parameter.RealParameter">0.0</epsilon>')
    A('        <subst   id="subst"   spec="parameter.RealParameter">0.05</subst>')
    A('        <beta    id="beta"    spec="parameter.RealParameter">0.5</beta>')
    A("    </MosseTipLikelihood>")
    A("")

    # Fixed observed base frequencies
    A("    <!-- Base frequencies fixed to observed values from the alignment. -->")
    A(f'    <parameter id="freqs" dimension="4" estimate="false" value="{fmt_freqs(bf)}"/>')
    A('    <frequencies id="estimatedFreqs" spec="Frequencies" frequencies="@freqs"/>')
    A("")

    A('    <parameter id="kappa" lower="0.0" value="2.0"/>')
    A('    <substModel id="hky" spec="HKY" kappa="@kappa" frequencies="@estimatedFreqs"/>')
    A('    <SiteModel id="sitemodel" spec="beast.base.evolution.sitemodel.SiteModel" mutationRate="1.0" substModel="@hky" />')
    A("")

    # Tree model (drift / diffusion)
    A("    <MosseDistribution id=\"treemodel\" spec=\"mosse.MosseDistribution\" tree=\"@tree\"")
    A('        nx="1024" dt="0.003" width="5" resolution="4" threads="16">')
    A('        <drift     id="drift"     spec="parameter.RealParameter">0.05</drift>')
    A('        <diffusion id="diffusion" spec="parameter.RealParameter" lower="0.0">0.002</diffusion>')
    A("    </MosseDistribution>")
    A("")

    # Link function parameters
    A('    <parameter id="curveYBaseValue"   spec="beast.base.inference.parameter.RealParameter" lower="0.0">0.00001</parameter>')
    A('    <parameter id="curveMaxY"         spec="beast.base.inference.parameter.RealParameter" lower="0.0">1.0</parameter>')
    A('    <parameter id="linearGrowthRate"  spec="beast.base.inference.parameter.RealParameter" lower="0.0">10.0</parameter>')
    A('    <parameter id="yV"                spec="beast.base.inference.parameter.RealParameter" lower="0.0">0.01</parameter>')
    A("")
    A('    <LinearFunction    id="linearfunc" spec="mosse.LinearFunction"')
    A('        curveYBaseValue="@curveYBaseValue" curveMaxY="@curveMaxY" linearGrowthRate="@linearGrowthRate" />')
    A('    <ConstantLinkFn    id="constfunc"  spec="mosse.ConstantLinkFn" yV="@yV" />')
    A("")
    A('    <RPNcalculator id="curveYDiff" spec="beast.base.inference.util.RPNcalculator"')
    A('        expression="curveMaxY curveYBaseValue -">')
    A('        <parameter idref="curveMaxY"/>')
    A('        <parameter idref="curveYBaseValue"/>')
    A("    </RPNcalculator>")
    A("")

    # ---- MCMC ----
    A('    <run chainLength="2000000" id="mcmc" spec="MCMC">')
    A('        <state id="state" storeEvery="100">')
    for sn in ("epsilon", "beta", "subst", "tree", "kappa", "drift", "diffusion",
               "curveYBaseValue", "curveMaxY", "linearGrowthRate", "yV"):
        A(f'            <stateNode idref="{sn}" />')
    A("        </state>")
    A("")

    A('        <distribution id="posterior" spec="beast.base.inference.CompoundDistribution">')
    A("")
    A('            <distribution id="likelihood" spec="beast.base.inference.CompoundDistribution">')
    A('                <distribution id="mossetreelikelihood" spec="mosse.MosseTreeLikelihoodMT"')
    A('                    data="@alignment" tree="@tree" siteModel="@sitemodel"')
    A('                    tipModel="@mossetiplikelihood" treeModel="@treemodel"')
    A('                    lambdaFunc="@linearfunc" muFunc="@constfunc"')
    A('                    resolutionMode="low">')
    A('                    <traits id="traitset" spec="beast.base.evolution.tree.TraitSet"')
    A(f'                        traitname="trait0" taxa="@taxonset" value="{fmt_traits(traits)}" />')
    A("                </distribution>")
    A("            </distribution>")
    A("")

    A('            <distribution id="prior" spec="beast.base.inference.CompoundDistribution">')
    A("")
    # Root height prior (tight Normal pinning to the true value)
    A('                <distribution id="rootHeightPrior" spec="beast.base.evolution.tree.MRCAPrior"')
    A('                    tree="@tree" monophyletic="true">')
    A('                    <taxonset idref="taxonset"/>')
    A('                    <distr id="Normal.rootHeight" spec="beast.base.inference.distribution.Normal">')
    A(f'                        <parameter name="mean"  spec="parameter.RealParameter" value="{true_root_height:.10f}" estimate="false"/>')
    A('                        <parameter name="sigma" spec="parameter.RealParameter" value="1e-3" estimate="false"/>')
    A("                    </distr>")
    A("                </distribution>")
    A("")

    # Logger-only MRCA priors for every non-root internal node
    for clade_id, clade_taxa in internal_clades:
        A(f'                <distribution id="height.{clade_id}" spec="beast.base.evolution.tree.MRCAPrior" tree="@tree" monophyletic="true">')
        A(f'                    <taxonset id="taxonset.{clade_id}" spec="beast.base.evolution.alignment.TaxonSet">')
        for t in clade_taxa:
            A(f'                        <taxon idref="{t}"/>')
        A("                    </taxonset>")
        A("                </distribution>")
    A("")

    # Parameter priors
    A("                <!-- ===== parameter priors ===== -->")
    A('                <distribution id="kappaPrior" spec="beast.base.inference.distribution.Prior" x="@kappa">')
    A('                    <distr id="LogNormal.kappa" spec="beast.base.inference.distribution.LogNormalDistributionModel">')
    A('                        <parameter name="M" spec="parameter.RealParameter" value="1.0"  estimate="false"/>')
    A('                        <parameter name="S" spec="parameter.RealParameter" value="10.0" estimate="false"/>')
    A("                    </distr>")
    A("                </distribution>")
    A("")
    A('                <distribution id="epsilonPrior" spec="beast.base.inference.distribution.Prior" x="@epsilon">')
    A('                    <distr id="Exponential.epsilon" spec="beast.base.inference.distribution.Exponential">')
    A('                        <parameter name="mean" spec="parameter.RealParameter" value="0.01" estimate="false"/>')
    A("                    </distr>")
    A("                </distribution>")
    A("")
    A('                <distribution id="betaPrior" spec="beast.base.inference.distribution.Prior" x="@beta">')
    A('                    <distr id="Normal.beta" spec="beast.base.inference.distribution.Normal">')
    A('                        <parameter name="mean"  spec="parameter.RealParameter" value="0.0" estimate="false"/>')
    A('                        <parameter name="sigma" spec="parameter.RealParameter" value="1.0" estimate="false"/>')
    A("                    </distr>")
    A("                </distribution>")
    A("")
    A('                <distribution id="substPrior" spec="beast.base.inference.distribution.Prior" x="@subst">')
    A('                    <distr id="Exponential.subst" spec="beast.base.inference.distribution.Exponential">')
    A('                        <parameter name="mean" spec="parameter.RealParameter" value="0.05" estimate="false"/>')
    A("                    </distr>")
    A("                </distribution>")
    A("")
    A('                <distribution id="driftPrior" spec="beast.base.inference.distribution.Prior" x="@drift">')
    A('                    <distr id="Normal.drift" spec="beast.base.inference.distribution.Normal">')
    A('                        <parameter name="mean"  spec="parameter.RealParameter" value="0.0" estimate="false"/>')
    A('                        <parameter name="sigma" spec="parameter.RealParameter" value="0.5" estimate="false"/>')
    A("                    </distr>")
    A("                </distribution>")
    A("")
    A('                <distribution id="diffusionPrior" spec="beast.base.inference.distribution.Prior" x="@diffusion">')
    A('                    <distr id="Exponential.diffusion" spec="beast.base.inference.distribution.Exponential">')
    A('                        <parameter name="mean" spec="parameter.RealParameter" value="0.002" estimate="false"/>')
    A("                    </distr>")
    A("                </distribution>")
    A("")
    A('                <distribution id="curveYBaseValuePrior" spec="beast.base.inference.distribution.Prior" x="@curveYBaseValue">')
    A('                    <distr id="Exponential.curveYBaseValue" spec="beast.base.inference.distribution.Exponential">')
    A('                        <parameter name="mean" spec="parameter.RealParameter" value="0.01" estimate="false"/>')
    A("                    </distr>")
    A("                </distribution>")
    A("")
    A('                <distribution id="curveYDiffPrior" spec="beast.base.inference.distribution.Prior" x="@curveYDiff">')
    A('                    <distr id="Exponential.curveYDiff" spec="beast.base.inference.distribution.Exponential">')
    A('                        <parameter name="mean" spec="parameter.RealParameter" value="2.0" estimate="false"/>')
    A("                    </distr>")
    A("                </distribution>")
    A("")
    A('                <distribution id="linearGrowthRatePrior" spec="beast.base.inference.distribution.Prior" x="@linearGrowthRate">')
    A('                    <distr id="Exponential.linearGrowthRate" spec="beast.base.inference.distribution.Exponential">')
    A('                        <parameter name="mean" spec="parameter.RealParameter" value="50.0" estimate="false"/>')
    A("                    </distr>")
    A("                </distribution>")
    A("")
    A('                <distribution id="yVPrior" spec="beast.base.inference.distribution.Prior" x="@yV">')
    A('                    <distr id="Exponential.yV" spec="beast.base.inference.distribution.Exponential">')
    A('                        <parameter name="mean" spec="parameter.RealParameter" value="2.0" estimate="false"/>')
    A("                    </distr>")
    A("                </distribution>")
    A("")
    A("            </distribution>  <!-- prior -->")
    A("        </distribution>  <!-- posterior -->")
    A("")

    # ---- operators ----
    A('        <operator id="Uniform.t"        spec="Uniform"                tree="@tree"        weight="3.0"/>')
    A('        <operator id="KappaScaler"      spec="ScaleOperator"          parameter="@kappa"  scaleFactor="0.75" weight="3.0"/>')
    A('        <operator id="EpsilonScaler"    spec="ScaleOperator"          parameter="@epsilon" scaleFactor="0.5" weight="3.0"/>')
    A('        <operator id="DriftRandomWalk"  spec="RealRandomWalkOperator" parameter="@drift"   windowSize="0.01" weight="3.0"/>')
    A('        <operator id="YVScaler"         spec="ScaleOperator"          parameter="@yV"     scaleFactor="0.5" weight="3.0"/>')
    A("")
    A('        <operator id="UpDownGroups" spec="UpDownOperator" scaleFactor="0.75" weight="10.0">')
    A('            <up   idref="diffusion"/>')
    A('            <up   idref="linearGrowthRate"/>')
    A('            <down idref="curveMaxY"/>')
    A('            <down idref="beta"/>')
    A('            <down idref="subst"/>')
    A('            <down idref="curveYBaseValue"/>')
    A("        </operator>")
    A("")

    # ---- loggers ----
    A('        <logger id="tracelog" logEvery="500" fileName="$(filebase).log">')
    for nm in ("posterior", "likelihood", "prior",
               "beta", "epsilon", "subst", "drift", "diffusion", "kappa",
               "curveYBaseValue", "curveMaxY", "linearGrowthRate", "yV"):
        A(f'            <log idref="{nm}" />')
    A('            <log idref="rootHeightPrior" />')
    for clade_id, _ in internal_clades:
        A(f'            <log idref="height.{clade_id}" />')
    for nm in ("kappaPrior", "epsilonPrior", "betaPrior", "substPrior",
               "driftPrior", "diffusionPrior", "curveYBaseValuePrior",
               "curveYDiffPrior", "linearGrowthRatePrior", "yVPrior"):
        A(f'            <log idref="{nm}" />')
    A("        </logger>")
    A("")
    A('        <logger id="treelog" logEvery="500" fileName="$(filebase).trees" mode="tree">')
    A('            <log idref="tree" />')
    A("        </logger>")
    A("    </run>")
    A("")
    A("</beast>")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def derive_seed(path: Path) -> int:
    # seq1_0.001_<N>.fasta -> N
    m = re.search(r"_(\d+)\.[^.]+$", path.name)
    if not m:
        return 0
    return int(m.group(1))


def main(argv: list[str]) -> int:
    if len(argv) not in (5, 6):
        print(__doc__, file=sys.stderr)
        return 2
    fasta = Path(argv[1])
    tree = Path(argv[2])
    trait = Path(argv[3])
    out = Path(argv[4])
    init_tree = Path(argv[5]) if len(argv) == 6 else None

    sequences = parse_fasta(fasta)
    taxa = [n for n, _ in sequences]
    bf = base_freqs(sequences)
    traits_all = parse_traits(trait)
    # simulated trait CSV stores exp(log-trait); the model covariate is the log-trait
    traits = {t: math.log(traits_all[t]) for t in taxa if t in traits_all}
    if len(traits) != len(taxa):
        print(f"WARN: trait coverage {len(traits)}/{len(taxa)}", file=sys.stderr)

    newick_text = tree.read_text().strip().replace("\n", "")
    newick_text = re.sub(r"\[[^\]]*\]", "", newick_text)
    root = parse_newick(newick_text)
    true_h = root_height(root)
    label_internals(root)

    seed = derive_seed(fasta)
    if init_tree is not None:
        # use the given (strict-clock) initial tree as-is; no perturbation
        t = re.sub(r"\[[^\]]*\]", "", init_tree.read_text().strip().replace("\n", ""))
        newick_perturbed = t.rstrip(";")
    else:
        rng = random.Random(seed)
        perturb_internal_heights(root, true_h, rng)
        newick_perturbed = newick_str(root)

    internal_clades: list[tuple[str, list[str]]] = []
    for nd in iter_internal(root):
        if nd.parent is None:
            continue  # skip root (covered by rootHeightPrior)
        internal_clades.append((nd.label, leaves_under(nd)))

    xml = build_xml(
        taxa=taxa,
        sequences=sequences,
        newick=newick_perturbed,
        true_root_height=true_h,
        traits=traits,
        bf=bf,
        internal_clades=internal_clades,
    )

    out.write_text(xml)
    print(f"wrote {out}  (seed={seed}, root_h={true_h:.6f}, "
          f"n_taxa={len(taxa)}, n_sites={len(sequences[0][1])}, "
          f"n_internals_logged={len(internal_clades)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
