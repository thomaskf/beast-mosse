package mosse;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.alignment.Sequence;
import beast.base.evolution.alignment.TaxonSet;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.substitutionmodel.JukesCantor;
import beast.base.evolution.tree.TraitSet;
import beast.base.evolution.tree.Tree;
import beast.base.inference.parameter.BooleanParameter;
import beast.base.inference.parameter.RealParameter;

/**
 * Verification harness for the per-branch punctuation indicators e in {0,1}
 * (P = I + e*a*Q), added on top of the scalar punctuational amplitude a.
 *
 * Unlike {@link MossePuncVsBaselineTest}, this test needs no externally
 * captured baseline number: it evaluates the SAME tree several times in-process
 * under different (a, e) configurations and asserts the relationships that must
 * hold by construction:
 *
 *   1. all e = 1  with amplitude a   ==  no e input at all with amplitude a
 *      (every branch punctuating reproduces the original global-a behaviour).
 *   2. all e = 0  with amplitude a   ==  no e input at all with amplitude 0
 *      (turning every indicator off reproduces the a = 0 baseline — a is
 *      "not applicable" on any branch: neither the speciation matrix nor the
 *      along-branch substitution term contributes).
 *   3. Sanity: with a > 0, all-on and all-off give DIFFERENT log-likelihoods
 *      (so a genuinely affects the answer and the test is not vacuous), and a
 *      mixed indicator vector differs from both — i.e. e really is per-branch.
 *
 * Each configuration is also timed (one warm-up call, then N_REPS timed calls)
 * and reported, so the cost of punctuation-on vs punctuation-off is visible.
 *
 * Requires the native libtest library (as the other MosseDistribution tests do).
 *
 * @author Thomas Wong (with assistance)
 */
public class MosseBranchwidePuncTest {

    /**
     * Tolerance for the equality assertions. The all-e=0 vs a=0 comparison is
     * mathematically exact but takes two different numerical routes to
     * exp(Q*r*dt) — the a=0 baseline reads the Java-built per-bin eQ cache while
     * the all-e=0/a>0 run recomputes it in the native integrator (the cache is
     * disabled whenever the amplitude is non-zero) — so they agree only up to
     * floating-point round-off, not bit-for-bit. The all-e=1 vs global-a
     * comparison takes the identical route and is bit-identical.
     */
    private static final double TOLERANCE = 1e-6;

    /** A punctuational amplitude that is (a) within the validity bound
     *  a*max|Q_ii| < 1 for Jukes-Cantor and (b) large enough to move the
     *  log-likelihood appreciably away from the a = 0 baseline. */
    private static final double A_POS = 0.1;

    /** Number of timed repetitions per configuration (after one warm-up call). */
    private static final int N_REPS = 5;

    /** Holds the log-likelihood and per-call timing stats for one configuration. */
    private static final class Result {
        final double logp;
        final double meanSec, sdSec, minSec, maxSec;
        Result(double logp, double meanSec, double sdSec, double minSec, double maxSec) {
            this.logp = logp;
            this.meanSec = meanSec;
            this.sdSec = sdSec;
            this.minSec = minSec;
            this.maxSec = maxSec;
        }
    }

    private Alignment buildAlignment(String[] names, String[] sequences) {
        List<Sequence> seqList = new ArrayList<>();
        assert (names.length == sequences.length);
        for (int i = 0; i < names.length; i++) {
            seqList.add(new Sequence(names[i], sequences[i]));
        }
        return new Alignment(seqList, "nucleotide");
    }

    /**
     * Build and evaluate the Mosse tree likelihood for the fixed 3-taxon tree
     * below, returning the log-likelihood together with timing over N_REPS
     * calls (one warm-up call is made first and not timed).
     *
     * @param a       punctuational amplitude passed to the MosseDistribution.
     * @param eValues per-branch indicators (length must equal the node count,
     *                here 5); pass {@code null} to omit the e input entirely
     *                (the legacy global-a behaviour).
     */
    private Result run(double a, Boolean[] eValues) {
        // ----- Tree / data: 3 taxa, so there is an internal branch whose own
        // indicator gates punctuation (node count = 2*3-1 = 5). -----
        String[] names = {"t0", "t1", "t2"};
        String[] sequences = {"A", "C", "G"};
        String newick = "((t0:0.2, t1:0.2):0.2, t2:0.4);";
        String traitValues = "t0=0.15, t1=0.1, t2=0.05";

        Alignment alignment = buildAlignment(names, sequences);

        // GLM parameters
        Double[] betasArray = {1.0};
        double meanSubst = 0.0;
        double epsilon = 0.01;
        int numBins = 1024;

        // lambda / mu link-function parameters
        Double[] y0 = new Double[] { 0.1 };
        Double[] y1 = new Double[] { 0.2 };
        Double[] x0 = new Double[] { 0.0 };
        Double[] rParm = new Double[] { 2.5 };
        Double[] yValue = new Double[] { 0.03 };

        // PDE parameters
        double drift = 0.0;
        double diffusion = 0.001;
        double dt = 0.01;
        int width = 5;
        int resolution = 4;

        Tree tree = new Tree(newick);
        JukesCantor JC = new JukesCantor();
        SiteModel siteModel = new SiteModel();
        siteModel.initByName(
                "mutationRate", "1.0",
                "gammaCategoryCount", 1,
                "substModel", JC);

        MosseTipLikelihood tipModel = new MosseTipLikelihood();
        tipModel.initByName(
                "beta", new RealParameter(betasArray),
                "subst", Double.toString(meanSubst),
                "epsilon", Double.toString(epsilon),
                "logscale", "false");

        TraitSet trait1 = new TraitSet();
        trait1.initByName(
                "traitname", "trait1",
                "taxa", new TaxonSet(alignment),
                "value", traitValues);
        List<TraitSet> traitsList = new ArrayList<>();
        traitsList.add(trait1);

        LogisticFunction logFunc = new LogisticFunction();
        logFunc.initByName(
                "curveYBaseValue", new RealParameter(y0),
                "curveMaxY", new RealParameter(y1),
                "sigmoidMidpoint", new RealParameter(x0),
                "logisticGrowthRate", new RealParameter(rParm));
        ConstantLinkFn constFunc = new ConstantLinkFn();
        constFunc.initByName("yV", new RealParameter(yValue));

        MosseDistribution mosseDist = new MosseDistribution();
        mosseDist.initByName(
                "tree", tree,
                "nx", Integer.toString(numBins),
                "drift", Double.toString(drift),
                "diffusion", Double.toString(diffusion),
                "dt", Double.toString(dt),
                "width", Integer.toString(width),
                "resolution", Integer.toString(resolution),
                "a", Double.toString(a));

        // Assemble the likelihood inputs, optionally adding the per-branch e.
        List<Object> args = new ArrayList<>(Arrays.asList(
                "data", alignment,
                "tree", tree,
                "siteModel", siteModel,
                "tipModel", tipModel,
                "treeModel", mosseDist,
                "traits", traitsList,
                "lambdaFunc", logFunc,
                "muFunc", constFunc));
        if (eValues != null) {
            args.add("e");
            args.add(new BooleanParameter(eValues));
        }

        MosseTreeLikelihood likelihood = new MosseTreeLikelihood();
        likelihood.initByName(args.toArray());

        // One untimed warm-up absorbs FFTW planning and first-call overhead.
        double warm = likelihood.calculateLogP();
        assertTrue("warm-up call returned non-finite logP", Double.isFinite(warm));

        double[] times = new double[N_REPS];
        double logp = warm;
        for (int i = 0; i < N_REPS; i++) {
            long t0 = System.nanoTime();
            logp = likelihood.calculateLogP();
            times[i] = (System.nanoTime() - t0) / 1e9;
        }

        double sum = 0, sumSq = 0, min = Double.POSITIVE_INFINITY, max = 0;
        for (double t : times) {
            sum += t;
            sumSq += t * t;
            if (t < min) min = t;
            if (t > max) max = t;
        }
        double mean = sum / N_REPS;
        double sd = Math.sqrt(Math.max(0.0, sumSq / N_REPS - mean * mean));
        return new Result(logp, mean, sd, min, max);
    }

    private static Boolean[] filled(int n, boolean value) {
        Boolean[] e = new Boolean[n];
        Arrays.fill(e, value);
        return e;
    }

    private static void report(String label, Result r) {
        System.out.printf(
                "%-22s logP = %.10f | mean %.4f s  sd %.4f s  min %.4f s  max %.4f s%n",
                label, r.logp, r.meanSec, r.sdSec, r.minSec, r.maxSec);
    }

    @Test
    public void testBranchwidePunc() {
        final int nodeCount = 5; // 3 taxa => 2*3-1 nodes

        Result globalA    = run(A_POS, null);                     // legacy global a
        Result allOn       = run(A_POS, filled(nodeCount, true));  // every branch punctuates
        Result globalZero = run(0.0, null);                       // legacy a = 0 baseline
        Result allOff      = run(A_POS, filled(nodeCount, false)); // no branch punctuates

        // mixed: punctuate the two tip branches of the (t0,t1) clade (nodes 0,1)
        // but not the clade's stem (node 3), t2's branch (node 2), or the root (4).
        Boolean[] mixedE = filled(nodeCount, false);
        mixedE[0] = true;
        mixedE[1] = true;
        Result mixed = run(A_POS, mixedE);

        System.out.println("====================================================================");
        System.out.println("MosseBranchwidePuncTest  (3-taxon tree, a = " + A_POS + ", " + N_REPS + " timed reps)");
        System.out.println("--------------------------------------------------------------------");
        report("global a (no e)",     globalA);
        report("all e = 1",           allOn);
        report("global a = 0 (no e)", globalZero);
        report("all e = 0",           allOff);
        report("mixed e (tips on)",   mixed);
        System.out.printf("punc-on / punc-off mean-time ratio = %.2fx%n",
                allOff.meanSec > 0 ? allOn.meanSec / allOff.meanSec : Double.NaN);
        System.out.println("====================================================================");

        // All evaluations must be finite.
        assertTrue("global-a logP not finite", Double.isFinite(globalA.logp));
        assertTrue("all-on logP not finite",   Double.isFinite(allOn.logp));
        assertTrue("baseline logP not finite", Double.isFinite(globalZero.logp));
        assertTrue("all-off logP not finite",  Double.isFinite(allOff.logp));
        assertTrue("mixed logP not finite",    Double.isFinite(mixed.logp));

        // (1) all e = 1 reproduces the global-a behaviour.
        assertEquals("all-e=1 should match the global-a likelihood",
                globalA.logp, allOn.logp, TOLERANCE);

        // (2) all e = 0 reproduces the a = 0 baseline.
        assertEquals("all-e=0 should match the a=0 baseline likelihood",
                globalZero.logp, allOff.logp, TOLERANCE);

        // (3a) a actually matters: with a > 0, all-on != all-off.
        // (assertTrue on an explicit difference rather than assertNotEquals, so this
        //  compiles against JUnit 4.8.2 — assertNotEquals only arrived in 4.11.)
        assertTrue("punctuation amplitude has no effect (all-on == all-off)",
                Math.abs(allOn.logp - allOff.logp) > TOLERANCE);

        // (3b) e is genuinely per-branch: a partial vector differs from both extremes.
        assertTrue("mixed e equals all-on (per-branch gating not taking effect)",
                Math.abs(allOn.logp - mixed.logp) > TOLERANCE);
        assertTrue("mixed e equals all-off (per-branch gating not taking effect)",
                Math.abs(allOff.logp - mixed.logp) > TOLERANCE);
    }
}
