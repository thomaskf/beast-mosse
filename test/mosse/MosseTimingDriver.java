package mosse;

import java.util.ArrayList;
import java.util.List;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.substitutionmodel.JukesCantor;
import beast.base.evolution.tree.TraitSet;
import beast.base.evolution.tree.Tree;
import beast.base.inference.parameter.RealParameter;

/**
 * Standalone timing driver for Mosse / Mosse-punc verification.
 * Runs for several tree sizes with the same per-tree parameter setup,
 * works against both baseline and punc builds (we deliberately do NOT
 * set "a" in initByName so the punc build uses its default a=0).
 *
 * Usage:
 *   java mosse.MosseTimingDriver <label> [size]
 * size = 2 (default) | 4 | 8
 */
public class MosseTimingDriver {

    private static final int N_REPS = 5;

    private static Alignment buildAlignment(int n, String[] names, String[] seqs) {
        List<Sequence> list = new ArrayList<>();
        for (int i = 0; i < n; i++) list.add(new Sequence(names[i], seqs[i]));
        return new Alignment(list, "nucleotide");
    }

    /** A small fixed test problem for a given tree size. */
    private static class Problem {
        int n;
        String[] names, seqs;
        String newick;
        String traitValues;
    }

    private static Problem makeProblem(int size) {
        Problem p = new Problem();
        switch (size) {
            case 2:
                p.n      = 2;
                p.names  = new String[]{"t0", "t1"};
                p.seqs   = new String[]{"A", "C"};
                p.newick = "(t0: 0.4, t1: 0.4);";
                p.traitValues = "t0=0.15, t1=0.1";
                break;
            case 4:
                p.n      = 4;
                p.names  = new String[]{"t0", "t1", "t2", "t3"};
                p.seqs   = new String[]{"AG", "CT", "GG", "TC"};
                p.newick = "((t0:0.2,t1:0.2):0.2,(t2:0.3,t3:0.3):0.1);";
                p.traitValues = "t0=0.15, t1=0.1, t2=0.25, t3=0.2";
                break;
            case 8:
                p.n      = 8;
                p.names  = new String[]{"t0","t1","t2","t3","t4","t5","t6","t7"};
                p.seqs   = new String[]{"AG","CT","GG","TC","AT","CG","GA","TT"};
                p.newick = "(((t0:0.2,t1:0.2):0.2,(t2:0.3,t3:0.3):0.1):0.1," +
                           "((t4:0.2,t5:0.2):0.2,(t6:0.3,t7:0.3):0.1):0.1);";
                p.traitValues = "t0=0.15,t1=0.10,t2=0.25,t3=0.20," +
                                "t4=0.18,t5=0.12,t6=0.22,t7=0.14";
                break;
            default:
                throw new IllegalArgumentException("Unsupported tree size: " + size);
        }
        return p;
    }

    public static void main(String[] args) throws Exception {
        // Bootstrap BEAST 2.7's package classloader before any beast.base
        // class with a static initializer (e.g. Alignment) loads.
        beast.pkgmgmt.PackageManager.loadExternalJars();

        String label = (args.length > 0) ? args[0] : "(unlabelled)";
        int size     = (args.length > 1) ? Integer.parseInt(args[1]) : 2;

        Problem prob = makeProblem(size);
        Alignment alignment = buildAlignment(prob.n, prob.names, prob.seqs);

        Tree tree = new Tree(prob.newick);
        JukesCantor JC = new JukesCantor();
        SiteModel siteModel = new SiteModel();
        siteModel.initByName("mutationRate", "1.0",
                             "gammaCategoryCount", 1,
                             "substModel", JC);

        MosseTipLikelihood tipModel = new MosseTipLikelihood();
        tipModel.initByName("beta", new RealParameter(new Double[]{1.0}),
                            "subst", "0.0",
                            "epsilon", "0.01",
                            "logscale", "false");

        TraitSet trait1 = new TraitSet();
        trait1.initByName("traitname", "trait1",
                          "taxa", new TaxonSet(alignment),
                          "value", prob.traitValues);
        List<TraitSet> traits = new ArrayList<>();
        traits.add(trait1);

        LogisticFunction logFunc = new LogisticFunction();
        logFunc.initByName("curveYBaseValue", new RealParameter(new Double[]{0.1}),
                           "curveMaxY", new RealParameter(new Double[]{0.2}),
                           "sigmoidMidpoint", new RealParameter(new Double[]{0.0}),
                           "logisticGrowthRate", new RealParameter(new Double[]{2.5}));
        ConstantLinkFn constFunc = new ConstantLinkFn();
        constFunc.initByName("yV", new RealParameter(new Double[]{0.03}));

        // No "a" key here on purpose — works on both builds.
        MosseDistribution dist = new MosseDistribution();
        dist.initByName("tree", tree,
                        "nx", "1024",
                        "drift", "0.0",
                        "diffusion", "0.001",
                        "dt", "0.01",
                        "width", "5",
                        "resolution", "4");

        MosseTreeLikelihood lik = new MosseTreeLikelihood();
        lik.initByName("data", alignment,
                       "tree", tree,
                       "siteModel", siteModel,
                       "tipModel", tipModel,
                       "treeModel", dist,
                       "traits", traits,
                       "lambdaFunc", logFunc,
                       "muFunc", constFunc);

        // Warm-up
        double warm = lik.calculateLogP();

        double[] times = new double[N_REPS];
        double logp = 0;
        for (int i = 0; i < N_REPS; i++) {
            long t0 = System.nanoTime();
            logp = lik.calculateLogP();
            times[i] = (System.nanoTime() - t0) / 1e9;
        }
        double sum = 0, sumSq = 0, min = Double.POSITIVE_INFINITY, max = 0;
        for (double t : times) { sum += t; sumSq += t*t; if (t<min) min=t; if (t>max) max=t; }
        double mean = sum / N_REPS;
        double sd   = Math.sqrt(Math.max(0.0, sumSq / N_REPS - mean * mean));

        System.out.println("==============================================");
        System.out.println("MosseTimingDriver -- " + label + " -- " + prob.n + " taxa");
        System.out.println("----------------------------------------------");
        System.out.printf ("warm-up logP            = %.10f%n", warm);
        System.out.printf ("log-likelihood          = %.10f%n", logp);
        System.out.printf ("repetitions             = %d%n", N_REPS);
        System.out.printf ("mean time per call      = %.4f s%n", mean);
        System.out.printf ("sd                      = %.4f s%n", sd);
        System.out.printf ("min / max               = %.4f s / %.4f s%n", min, max);
        System.out.println("==============================================");
    }
}
