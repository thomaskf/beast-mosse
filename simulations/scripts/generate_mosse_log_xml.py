#!/usr/bin/env python3
"""Generate a sim6 MoSSE XML (log-scale, no punctuation, no gamma).

Per-parameter initial values follow the collaborator's strategy in
260603_param_table_update.docx:

    subst     = log(strict_clock_clockRate_median)        # log-rate scale
    kappa     = strict_clock_kappa_median
    beta      = 0
    drift     = 0
    diffusion = 0.005                                     # collaborator's value
    epsilon   = 0.1        (prior Exp(mean=0.1), brackets typical truth)
    curveYBaseValue   = lambda_BD   (ape::birthdeath on true tree)
    curveMaxY         = lambda_BD
    linearGrowthRate  = 0
    yV                = max(mu_BD, 0.001)
    base freqs        = observed (fixed, estimate=false)
    tree (TreeParser) = strict-clock representative tree (no perturbation)
                        with any branch < 0.003 floored to 0.003
    traits value      = log(trait_csv_value)              # logscale=true model
    root height       = MRCAPrior Normal(true_h, 1e-3)

Usage:
    generate_mosse_log_xml.py \
        <fasta> <true_tree> <trait_csv> \
        <strict_clock_summary.json> <bd_estimates.json> \
        <out.xml>
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from summarize_dataset import (
    parse_fasta, parse_newick, parse_traits, base_freqs,
    iter_internal, leaves_under, root_height,
)
from generate_xml import label_internals, newick_str, fmt_freqs

MIN_BRANCH = 0.003  # matches dt in MosseDistribution
DEFAULT_YV_FLOOR = 0.001  # for seeds whose ape::birthdeath() returned mu=0


XML_NS = (
    "beast.base.evolution.alignment:beast.pkgmgmt:beast.base.core:"
    "beast.base.inference:beast.base.evolution.tree.coalescent:"
    "beast.pkgmgmt:beast.base.core:beast.base.inference.util:"
    "beast.evolution.nuc:beast.base.evolution.operator:"
    "beast.base.inference.operator:beast.base.evolution.sitemodel:"
    "beast.base.evolution.substitutionmodel:beast.base.evolution.likelihood:"
    "beast.base.core"
)


def floor_branches(root, mn: float = MIN_BRANCH) -> None:
    """Walk the tree; for any branch shorter than `mn`, raise it to `mn`.
    Operates in-place on TreeNode.branch. May slightly inflate the root height.
    """
    def walk(n):
        for c in n.children:
            if c.branch < mn:
                c.branch = mn
            walk(c)
    walk(root)


def fmt_traits_log(traits: dict[str, float]) -> str:
    out = []
    for t, v in traits.items():
        if v <= 0:
            raise ValueError(f"trait for {t} = {v} is non-positive; "
                             f"cannot take log for the log-scale model")
        out.append(f"{t}={math.log(v):.15g}")
    return ",".join(out)


def build(*, taxa, sequences, newick, true_root_h, traits, bf,
          internal_clades, subst_init, kappa_init,
          lambda_bd, yv_init,
          chain_length: int) -> str:
    L: list[str] = []
    A = L.append

    A(f"<beast version='2.0' namespace='{XML_NS}'>")
    A("")
    A(f"    <!-- sim6 log-scale, ntax={len(taxa)} nchar={len(sequences[0][1])} -->")
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

    A("    <!-- Initial tree = strict-clock posterior representative tree (any branch < 0.003 floored). -->")
    A("    <tree id='tree' spec='beast.base.evolution.tree.TreeParser' IsLabelledNewick='true'")
    A(f"        newick='{newick}'")
    A("        taxa='@alignment' />")
    A("")

    # MoSSE tip likelihood -- LOG SCALE
    A('    <MosseTipLikelihood id="mossetiplikelihood" spec="mosse.MosseTipLikelihood" logscale="true">')
    A('        <epsilon id="epsilon" spec="parameter.RealParameter">0.1</epsilon>')
    A(f'        <subst   id="subst"   spec="parameter.RealParameter">{subst_init:.10f}</subst>')
    A('        <beta    id="beta"    spec="parameter.RealParameter">0.0</beta>')
    A("    </MosseTipLikelihood>")
    A("")

    A("    <!-- Base frequencies fixed to observed values from the alignment. -->")
    A(f'    <parameter id="freqs" dimension="4" estimate="false" value="{fmt_freqs(bf)}"/>')
    A('    <frequencies id="estimatedFreqs" spec="Frequencies" frequencies="@freqs"/>')
    A("")

    A(f'    <parameter id="kappa" lower="0.0" value="{kappa_init:.6f}"/>')
    A('    <substModel id="hky" spec="HKY" kappa="@kappa" frequencies="@estimatedFreqs"/>')
    A('    <SiteModel id="sitemodel" spec="beast.base.evolution.sitemodel.SiteModel" mutationRate="1.0" substModel="@hky" />')
    A("")

    # Tree model (drift / diffusion)
    A('    <MosseDistribution id="treemodel" spec="mosse.MosseDistribution" tree="@tree"')
    A('        nx="1024" dt="0.003" width="5" resolution="4" threads="24">')
    A('        <drift     id="drift"     spec="parameter.RealParameter">0.0</drift>')
    A('        <diffusion id="diffusion" spec="parameter.RealParameter" lower="0.0">0.005</diffusion>')
    A("    </MosseDistribution>")
    A("")

    # Link function parameters
    A(f'    <parameter id="curveYBaseValue"  spec="beast.base.inference.parameter.RealParameter" lower="0.0">{lambda_bd:.10f}</parameter>')
    A(f'    <parameter id="curveMaxY"        spec="beast.base.inference.parameter.RealParameter" lower="0.0">{lambda_bd*1.5:.10f}</parameter>')
    A('    <parameter id="linearGrowthRate"  spec="beast.base.inference.parameter.RealParameter" lower="0.0">1.0</parameter>')
    A(f'    <parameter id="yV"               spec="beast.base.inference.parameter.RealParameter" lower="0.0">{yv_init:.10f}</parameter>')
    A("")
    A('    <LinearFunction id="linearfunc" spec="mosse.LinearFunction"')
    A('        curveYBaseValue="@curveYBaseValue" curveMaxY="@curveMaxY" linearGrowthRate="@linearGrowthRate" />')
    A('    <ConstantLinkFn id="constfunc"  spec="mosse.ConstantLinkFn" yV="@yV" />')
    A("")
    A('    <RPNcalculator id="curveYDiff" spec="beast.base.inference.util.RPNcalculator"')
    A('        expression="curveMaxY curveYBaseValue -">')
    A('        <parameter idref="curveMaxY"/>')
    A('        <parameter idref="curveYBaseValue"/>')
    A("    </RPNcalculator>")
    A("")

    A(f'    <run chainLength="{chain_length}" id="mcmc" spec="MCMC">')
    A('        <state id="state" storeEvery="100">')
    for sn in ("epsilon", "beta", "subst", "tree", "kappa", "drift", "diffusion",
               "curveYBaseValue", "curveMaxY", "linearGrowthRate", "yV"):
        A(f'            <stateNode idref="{sn}" />')
    A("        </state>")
    A("")

    A('        <distribution id="posterior" spec="beast.base.inference.CompoundDistribution">')
    A('            <distribution id="likelihood" spec="beast.base.inference.CompoundDistribution">')
    A('                <distribution id="mossetreelikelihood" spec="mosse.MosseTreeLikelihoodMT"')
    A('                    data="@alignment" tree="@tree" siteModel="@sitemodel"')
    A('                    tipModel="@mossetiplikelihood" treeModel="@treemodel"')
    A('                    lambdaFunc="@linearfunc" muFunc="@constfunc"')
    A('                    resolutionMode="low">')
    A('                    <traits id="traitset" spec="beast.base.evolution.tree.TraitSet"')
    A(f'                        traitname="trait0" taxa="@taxonset" value="{fmt_traits_log(traits)}" />')
    A("                </distribution>")
    A("            </distribution>")
    A("")
    A('            <distribution id="prior" spec="beast.base.inference.CompoundDistribution">')

    # rootHeightPrior removed: root height is no longer pinned by a prior; it
    # stays at its initialised (true) value since UniformSelective skips the root.

    for clade_id, clade_taxa in internal_clades:
        A(f'                <distribution id="height.{clade_id}" spec="beast.base.evolution.tree.MRCAPrior" tree="@tree" monophyletic="true">')
        A(f'                    <taxonset id="taxonset.{clade_id}" spec="beast.base.evolution.alignment.TaxonSet">')
        for t in clade_taxa:
            A(f'                        <taxon idref="{t}"/>')
        A("                    </taxonset>")
        A("                </distribution>")
    A("")

    A("                <!-- parameter priors -->")
    A('                <distribution id="kappaPrior" spec="beast.base.inference.distribution.Prior" x="@kappa">')
    A('                    <distr id="LogNormal.kappa" spec="beast.base.inference.distribution.LogNormalDistributionModel">')
    A('                        <parameter name="M" spec="parameter.RealParameter" value="1.0"  estimate="false"/>')
    A('                        <parameter name="S" spec="parameter.RealParameter" value="10.0" estimate="false"/>')
    A("                    </distr>")
    A("                </distribution>")
    A("")
    A('                <distribution id="epsilonPrior" spec="beast.base.inference.distribution.Prior" x="@epsilon">')
    A('                    <distr id="Exponential.epsilon" spec="beast.base.inference.distribution.Exponential">')
    A('                        <parameter name="mean" spec="parameter.RealParameter" value="0.1" estimate="false"/>')
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
    A("                <!-- subst is log-rate (logscale=true); prior centred on log(0.02)=-3.91 -->")
    A('                <distribution id="substPrior" spec="beast.base.inference.distribution.Prior" x="@subst">')
    A('                    <distr id="Normal.subst" spec="beast.base.inference.distribution.Normal">')
    A('                        <parameter name="mean"  spec="parameter.RealParameter" value="-3.0" estimate="false"/>')
    A('                        <parameter name="sigma" spec="parameter.RealParameter" value="2.0"  estimate="false"/>')
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
    A('                        <parameter name="mean" spec="parameter.RealParameter" value="0.1" estimate="false"/>')
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
    A('        <operator id="UniformSelective.t" spec="beastlabs.evolution.operators.UniformSelective" tree="@tree" weight="3.0">')
    for clade_id, _ in internal_clades:
        A(f'            <restricted idref="height.{clade_id}"/>')
    A("        </operator>")
    A('        <operator id="KappaScaler"      spec="ScaleOperator"          parameter="@kappa"  scaleFactor="0.75" weight="3.0"/>')
    A('        <operator id="EpsilonScaler"    spec="ScaleOperator"          parameter="@epsilon" scaleFactor="0.5" weight="3.0"/>')
    A('        <operator id="DriftRandomWalk"  spec="RealRandomWalkOperator" parameter="@drift"   windowSize="0.05" weight="3.0"/>')
    A('        <operator id="YVScaler"         spec="ScaleOperator"          parameter="@yV"     scaleFactor="0.5" weight="3.0"/>')
    A("")
    A("        <!-- Individual random-walk mover for beta (so it can leave 0). -->")
    A('        <operator id="BetaRandomWalk"   spec="RealRandomWalkOperator" parameter="@beta"             windowSize="0.1"  weight="3.0"/>')
    A("")
    A("        <!-- Individual movers for diffusion, curve params, linearGrowthRate and subst. -->")
    A('        <operator id="DiffusionScaler"    spec="ScaleOperator"          parameter="@diffusion"        scaleFactor="0.75" weight="3.0"/>')
    A('        <operator id="CurveYBaseScaler"   spec="ScaleOperator"          parameter="@curveYBaseValue"  scaleFactor="0.75" weight="3.0"/>')
    A('        <operator id="CurveMaxYScaler"    spec="ScaleOperator"          parameter="@curveMaxY"        scaleFactor="0.75" weight="3.0"/>')
    A('        <operator id="LinearGrowthScaler" spec="ScaleOperator"          parameter="@linearGrowthRate" scaleFactor="0.75" weight="3.0"/>')
    A('        <operator id="SubstRandomWalk"    spec="RealRandomWalkOperator" parameter="@subst"            windowSize="0.1"   weight="3.0"/>')
    A("")
    A("        <!-- Joint correlation moves: LGR<->diffusion ridge + full 6-param up/down. -->")
    A('        <operator id="LGRDiffusionUp"     spec="UpDownOperator"         scaleFactor="0.75" weight="5.0">')
    A('            <up idref="linearGrowthRate"/>')
    A('            <up idref="diffusion"/>')
    A("        </operator>")
    A('        <operator id="UpDownGroups" spec="UpDownOperator" scaleFactor="0.75" weight="3.0">')
    A('            <up   idref="diffusion"/>')
    A('            <up   idref="linearGrowthRate"/>')
    A('            <down idref="curveMaxY"/>')
    A('            <down idref="beta"/>')
    A('            <down idref="subst"/>')
    A('            <down idref="curveYBaseValue"/>')
    A("        </operator>")

    # ---- loggers ----
    A('        <logger id="tracelog" logEvery="500" fileName="$(filebase).log">')
    for nm in ("posterior", "likelihood", "prior",
               "beta", "epsilon", "subst", "drift", "diffusion", "kappa",
               "curveYBaseValue", "curveMaxY", "linearGrowthRate", "yV"):
        A(f'            <log idref="{nm}" />')
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
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta")
    ap.add_argument("true_tree")
    ap.add_argument("trait_csv")
    ap.add_argument("strict_clock_summary")
    ap.add_argument("bd_estimates")
    ap.add_argument("out")
    ap.add_argument("--chain", type=int, default=1_000_000)
    ap.add_argument("--yv-floor", type=float, default=DEFAULT_YV_FLOOR)
    args = ap.parse_args(argv[1:])

    fasta = Path(args.fasta)
    sequences = parse_fasta(fasta)
    taxa = [n for n, _ in sequences]
    bf = base_freqs(sequences)
    traits_raw = parse_traits(Path(args.trait_csv))
    traits = {t: traits_raw[t] for t in taxa if t in traits_raw}

    # True tree -- only used for the root-height prior pin
    true_nw = Path(args.true_tree).read_text().strip().replace("\n", "")
    true_nw = re.sub(r"\[[^\]]*\]", "", true_nw)
    true_root_obj = parse_newick(true_nw)
    true_h = root_height(true_root_obj)

    # Starting tree = strict-clock representative
    sc = json.loads(Path(args.strict_clock_summary).read_text())
    sc_nw = sc["representative_tree_newick"]
    sc_nw = re.sub(r"\[[^\]]*\]", "", sc_nw).strip()
    sc_root = parse_newick(sc_nw)
    floor_branches(sc_root, MIN_BRANCH)
    label_internals(sc_root)

    # Per-seed BD estimates
    seed = int(re.search(r"_(\d+)\.[^.]+$", fasta.name).group(1))
    bd_all = json.loads(Path(args.bd_estimates).read_text())
    bd = bd_all.get(str(seed))
    if bd is None or "lambda_BD" not in bd:
        raise KeyError(f"no BD estimate for seed {seed} in {args.bd_estimates}")
    lambda_bd = float(bd["lambda_BD"])
    mu_bd = float(bd["mu_BD"])
    yv_init = max(mu_bd, args.yv_floor)

    subst_init = math.log(float(sc["clockRate_median"]))
    kappa_init = float(sc["kappa_median"])

    internal_clades = []
    for nd in iter_internal(sc_root):
        if nd.parent is None:
            continue
        internal_clades.append((nd.label, leaves_under(nd)))

    sc_nw_out = newick_str(sc_root)

    xml = build(
        taxa=taxa, sequences=sequences, newick=sc_nw_out,
        true_root_h=true_h, traits=traits, bf=bf,
        internal_clades=internal_clades,
        subst_init=subst_init, kappa_init=kappa_init,
        lambda_bd=lambda_bd, yv_init=yv_init,
        chain_length=args.chain,
    )
    Path(args.out).write_text(xml)
    print(f"wrote {args.out}  (seed={seed}, n_taxa={len(taxa)}, n_sites={len(sequences[0][1])}, "
          f"subst_init={subst_init:.4f}, kappa_init={kappa_init:.3f}, "
          f"lambda_BD={lambda_bd:.4f}, yV_init={yv_init:.4f})")


if __name__ == "__main__":
    main(sys.argv)
