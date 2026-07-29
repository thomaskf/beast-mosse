#!/usr/bin/env python3
"""Generate a BEAST 2 XML for a fast strict-clock analysis on one replicate.

Goal: get a posterior estimate of the substitution rate (clockRate), kappa,
and the internal node heights, with the topology and root height held to
their true values. The output of this run feeds the initial values of the
downstream MoSSE XML (see collaborator note in 260603_param_table_update.docx).

Configuration:
  - HKY (kappa free, base freqs fixed to observed)
  - Strict molecular clock (rate free)
  - Tree topology fixed via TreeParser
  - Root height pinned by tight MRCAPrior Normal(true_h, 1e-3) over all taxa
  - Per non-root internal node: logger-only MRCAPrior to trace heights
  - Operators: Uniform.t for node heights, ScaleOperator for kappa + clockRate
  - Chain length default 500_000; sample every 500

Usage:
    generate_strict_clock_xml.py <fasta> <tree> <out.xml> [--chain N]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from summarize_dataset import (
    parse_fasta, parse_newick, base_freqs,
    iter_internal, leaves_under, root_height,
)
from generate_xml import label_internals, newick_str


XML_NS = (
    "beast.base.evolution.alignment:beast.pkgmgmt:beast.base.core:"
    "beast.base.inference:beast.base.evolution.tree.coalescent:"
    "beast.pkgmgmt:beast.base.core:beast.base.inference.util:"
    "beast.evolution.nuc:beast.base.evolution.operator:"
    "beast.base.inference.operator:beast.base.evolution.sitemodel:"
    "beast.base.evolution.substitutionmodel:beast.base.evolution.likelihood:"
    "beast.base.core:beast.base.evolution.branchratemodel"
)


def fmt_freqs(bf):
    # Round to 6 decimals, then absorb any rounding residual into the last freq
    # so BEAST's "frequencies must sum to 1" check passes.
    rounded = [round(bf[b], 6) for b in "acgt"]
    rounded[-1] = round(1.0 - sum(rounded[:-1]), 6)
    return " ".join(f"{v:.6f}" for v in rounded)


def build(taxa, sequences, newick, true_root_h, bf, internal_clades, chain):
    L: list[str] = []
    A = L.append

    A(f"<beast version='2.0' namespace='{XML_NS}'>")
    A("")
    A(f"    <!-- strict-clock: ntax={len(taxa)} nchar={len(sequences[0][1])} -->")
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
    A("    <!-- Topology and branch lengths initialised from the true tree; topology stays fixed (no topology operators). -->")
    A("    <tree id='tree' spec='beast.base.evolution.tree.TreeParser' IsLabelledNewick='true'")
    A(f"        newick='{newick}'")
    A("        taxa='@alignment' />")
    A("")
    A(f'    <parameter id="freqs" dimension="4" estimate="false" value="{fmt_freqs(bf)}"/>')
    A('    <frequencies id="estimatedFreqs" spec="Frequencies" frequencies="@freqs"/>')
    A("")
    A('    <parameter id="kappa"     lower="0.0" value="2.0"/>')
    A('    <parameter id="clockRate" lower="0.0" value="0.005"/>')
    A("")
    A('    <substModel id="hky" spec="HKY" kappa="@kappa" frequencies="@estimatedFreqs"/>')
    A('    <SiteModel id="sitemodel" spec="beast.base.evolution.sitemodel.SiteModel"')
    A('        mutationRate="1.0" substModel="@hky"/>')
    A('    <branchRateModel id="clock" spec="beast.base.evolution.branchratemodel.StrictClockModel" clock.rate="@clockRate"/>')
    A("")

    A(f'    <run chainLength="{chain}" id="mcmc" spec="MCMC">')
    A('        <state id="state" storeEvery="500">')
    A('            <stateNode idref="tree"/>')
    A('            <stateNode idref="kappa"/>')
    A('            <stateNode idref="clockRate"/>')
    A("        </state>")
    A("")
    A('        <distribution id="posterior" spec="beast.base.inference.CompoundDistribution">')
    A('            <distribution id="likelihood" spec="beast.base.inference.CompoundDistribution">')
    A('                <distribution id="treeLikelihood" spec="beast.base.evolution.likelihood.TreeLikelihood"')
    A('                    data="@alignment" tree="@tree" siteModel="@sitemodel" branchRateModel="@clock"/>')
    A("            </distribution>")
    A("")
    A('            <distribution id="prior" spec="beast.base.inference.CompoundDistribution">')
    A("                <!-- Root height pinned to the true value -->")
    A('                <distribution id="rootHeightPrior" spec="beast.base.evolution.tree.MRCAPrior" tree="@tree" monophyletic="true">')
    A('                    <taxonset idref="taxonset"/>')
    A('                    <distr id="Normal.rootHeight" spec="beast.base.inference.distribution.Normal">')
    A(f'                        <parameter name="mean"  spec="parameter.RealParameter" value="{true_root_h:.10f}" estimate="false"/>')
    A('                        <parameter name="sigma" spec="parameter.RealParameter" value="1e-3" estimate="false"/>')
    A("                    </distr>")
    A("                </distribution>")
    A("")
    # logger-only MRCAPriors so we can read heights from the .log if needed
    for clade_id, clade_taxa in internal_clades:
        A(f'                <distribution id="height.{clade_id}" spec="beast.base.evolution.tree.MRCAPrior" tree="@tree" monophyletic="true">')
        A(f'                    <taxonset id="taxonset.{clade_id}" spec="beast.base.evolution.alignment.TaxonSet">')
        for t in clade_taxa:
            A(f'                        <taxon idref="{t}"/>')
        A("                    </taxonset>")
        A("                </distribution>")
    A("")
    A('                <distribution id="kappaPrior" spec="beast.base.inference.distribution.Prior" x="@kappa">')
    A('                    <distr id="LogNormal.kappa" spec="beast.base.inference.distribution.LogNormalDistributionModel">')
    A('                        <parameter name="M" spec="parameter.RealParameter" value="1.0"  estimate="false"/>')
    A('                        <parameter name="S" spec="parameter.RealParameter" value="10.0" estimate="false"/>')
    A("                    </distr>")
    A("                </distribution>")
    A('                <distribution id="clockPrior" spec="beast.base.inference.distribution.Prior" x="@clockRate">')
    A('                    <distr id="LogNormal.clockRate" spec="beast.base.inference.distribution.LogNormalDistributionModel">')
    A('                        <parameter name="M" spec="parameter.RealParameter" value="-3.0" estimate="false"/>')
    A('                        <parameter name="S" spec="parameter.RealParameter" value="2.0"  estimate="false"/>')
    A("                    </distr>")
    A("                </distribution>")
    A("            </distribution>  <!-- prior -->")
    A("        </distribution>  <!-- posterior -->")
    A("")
    A('        <operator id="Uniform.t"     spec="Uniform"       tree="@tree"        weight="10.0"/>')
    A('        <operator id="KappaScaler"   spec="ScaleOperator" parameter="@kappa"  scaleFactor="0.75" weight="3.0"/>')
    A('        <operator id="ClockScaler"   spec="ScaleOperator" parameter="@clockRate" scaleFactor="0.75" weight="3.0"/>')
    A("")
    A('        <logger id="tracelog" logEvery="500" fileName="$(filebase).log">')
    A('            <log idref="posterior" /><log idref="likelihood" /><log idref="prior" />')
    A('            <log idref="kappa" /><log idref="clockRate" />')
    A('            <log idref="rootHeightPrior" />')
    for clade_id, _ in internal_clades:
        A(f'            <log idref="height.{clade_id}" />')
    A("        </logger>")
    A('        <logger id="treelog" logEvery="500" fileName="$(filebase).trees" mode="tree">')
    A('            <log idref="tree" />')
    A("        </logger>")
    A('        <logger id="screen" logEvery="10000">')
    A('            <log idref="posterior" /><log idref="clockRate" /><log idref="kappa" />')
    A("        </logger>")
    A("    </run>")
    A("")
    A("</beast>")
    return "\n".join(L) + "\n"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta")
    ap.add_argument("tree")
    ap.add_argument("out")
    ap.add_argument("--chain", type=int, default=500_000)
    args = ap.parse_args(argv[1:])

    seqs = parse_fasta(Path(args.fasta))
    taxa = [n for n, _ in seqs]
    bf = base_freqs(seqs)

    nw_text = Path(args.tree).read_text().strip().replace("\n", "")
    nw_text = re.sub(r"\[[^\]]*\]", "", nw_text)
    root = parse_newick(nw_text)
    true_h = root_height(root)
    label_internals(root)

    internal_clades = []
    for nd in iter_internal(root):
        if nd.parent is None:
            continue
        internal_clades.append((nd.label, leaves_under(nd)))

    # Use the true Newick directly (no perturbation) for the strict-clock start.
    nw_out = newick_str(root)
    xml = build(taxa, seqs, nw_out, true_h, bf, internal_clades, args.chain)
    Path(args.out).write_text(xml)
    print(f"wrote {args.out}  (n_taxa={len(taxa)}, n_sites={len(seqs[0][1])}, "
          f"root_h={true_h:.6f}, internals_logged={len(internal_clades)}, chain={args.chain})")


if __name__ == "__main__":
    main(sys.argv)
