package mosse;

import java.util.ArrayList;
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
import beast.base.inference.parameter.RealParameter;

/**
 * Verification harness for the punctuational integrator.
 *
 * Runs a fixed small tree through the punc path with a = 0 and (a) prints
 * the resulting log-likelihood with timing, (b) optionally asserts equality
 * against a baseline log-likelihood obtained by running the equivalent test
 * against the original (no-punc) beast-mosse build.
 *
 * Workflow:
 *   1. With the ORIGINAL beast-mosse build installed (libtest.dylib without
 *      punctuational support), run an equivalent likelihood evaluation and
 *      record the printed log-likelihood. (See README.)
 *   2. Paste that value into EXPECTED_BASELINE_LOGP below.
 *   3. With this beast-mosse-punc build installed (libtest.dylib WITH punc
 *      support), run this test. It should report the same log-likelihood
 *      and pass the equality assertion. The test also prints a slowdown
 *      ratio you can divide by the baseline timing.
 *
 * @author Thomas Wong (with assistance)
 */
public class MossePuncVsBaselineTest {

    /**
     * Expected log-likelihood from the BASELINE (non-punc) build for the
     * same tree, set explicitly here so the punc-with-a=0 result can be
     * verified against it.
     *
     * Set to Double.NaN to skip the equality assertion (useful on the
     * very first run, before the baseline value has been captured).
     */
    private static final double EXPECTED_BASELINE_LOGP = -16.3544727367;

    /** Tolerance for the equality assertion. */
    private static final double TOLERANCE = 1e-6;

    /** Number of timed repetitions (after one warm-up call). */
    private static final int N_REPS = 5;

    /**
     * Build the same little 2-taxon tree as MosseTreeLikelihoodTest uses.
     * Kept inline so this test is self-contained.
     */
    private Alignment buildAlignment(int numLeaves, String[] names, String[] sequences) {
        List<Sequence> seqList = new ArrayList<>();
        assert (names.length == sequences.length);
        for (int i = 0; i < numLeaves; i++) {
            seqList.add(new Sequence(names[i], sequences[i]));
        }
        return new Alignment(seqList, "nucleotide");
    }

    @Test
    public void testPuncVsBaseline_aZero() {
        // ----- Tree / data (kept identical to MosseTreeLikelihoodTest) -----
        int numLeaves = 2;
        String[] names = {"t0", "t1"};
        String[] sequences = {"A", "C"};
        String newick = "(t0: 0.4, t1: 0.4);";
        int numTraits = 1;
        String trait1Values = "t0=0.15, t1=0.1";

        Alignment alignment = buildAlignment(numLeaves, names, sequences);

        // GLM parameters
        Double[] betasArray = {1.0};
        double meanSubst = 0.0;
        double epsilon = 0.01;
        int numBins = 1024;

        // λ / μ link-function parameters
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

        // ----- Models -----
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
                "logscale", "false"
        );

        TraitSet trait1 = new TraitSet();
        trait1.initByName(
                "traitname", "trait1",
                "taxa", new TaxonSet(alignment),
                "value", trait1Values);
        List<TraitSet> traitsList = new ArrayList<>(numTraits);
        traitsList.add(trait1);

        // λ as logistic, μ as constant
        LogisticFunction logFunc = new LogisticFunction();
        logFunc.initByName(
                "curveYBaseValue", new RealParameter(y0),
                "curveMaxY", new RealParameter(y1),
                "sigmoidMidpoint", new RealParameter(x0),
                "logisticGrowthRate", new RealParameter(rParm));
        ConstantLinkFn constFunc = new ConstantLinkFn();
        constFunc.initByName("yV", new RealParameter(yValue));

        // ----- MosseDistribution with PUNC parameter a = 0 -----
        MosseDistribution mosseDist = new MosseDistribution();
        mosseDist.initByName(
                "tree", tree,
                "nx", Integer.toString(numBins),
                "drift", Double.toString(drift),
                "diffusion", Double.toString(diffusion),
                "dt", Double.toString(dt),
                "width", Integer.toString(width),
                "resolution", Integer.toString(resolution),
                "a", "0.0"                                  // <-- THE KEY KNOB
        );

        MosseTreeLikelihood likelihood = new MosseTreeLikelihood();
        likelihood.initByName(
                "data", alignment,
                "tree", tree,
                "siteModel", siteModel,
                "tipModel", tipModel,
                "treeModel", mosseDist,
                "traits", traitsList,
                "lambdaFunc", logFunc,
                "muFunc", constFunc
        );

        // ----- Warm-up + timed runs -----
        // One untimed call absorbs FFTW planning and any first-call overhead.
        double warm = likelihood.calculateLogP();
        assertTrue("warm-up call returned non-finite logP", Double.isFinite(warm));

        double[] times = new double[N_REPS];
        double logp = 0;
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

        // ----- Report -----
        System.out.println("==============================================");
        System.out.println("MossePuncVsBaselineTest -- a = 0");
        System.out.println("----------------------------------------------");
        System.out.printf ("log-likelihood          = %.10f%n", logp);
        System.out.printf ("repetitions             = %d%n", N_REPS);
        System.out.printf ("mean time per call      = %.4f s%n", mean);
        System.out.printf ("sd                      = %.4f s%n", sd);
        System.out.printf ("min / max               = %.4f s / %.4f s%n", min, max);
        if (!Double.isNaN(EXPECTED_BASELINE_LOGP)) {
            System.out.printf("expected (baseline)     = %.10f%n", EXPECTED_BASELINE_LOGP);
            System.out.printf("difference              = %.3e%n",
                              logp - EXPECTED_BASELINE_LOGP);
        } else {
            System.out.println("expected (baseline)     = (not set; skipping assertEquals)");
            System.out.println("To enable the assertion, run the equivalent test against");
            System.out.println("the original beast-mosse build, copy its log-likelihood");
            System.out.println("into EXPECTED_BASELINE_LOGP at the top of this file, and");
            System.out.println("re-run.");
        }
        System.out.println("==============================================");

        // ----- Assertion (only if the baseline is set) -----
        if (!Double.isNaN(EXPECTED_BASELINE_LOGP)) {
            assertEquals(
                "punc-with-a=0 log-likelihood does not match the baseline",
                EXPECTED_BASELINE_LOGP, logp, TOLERANCE);
        }
    }
}
