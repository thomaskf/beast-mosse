package mosse;

import mosseapprox.RbarProvider;
import mosseapprox.SubstitutionModel;
import beast.base.core.Description;
import beast.base.evolution.tree.Node;
import java.util.function.DoubleUnaryOperator;
import java.util.function.DoubleBinaryOperator;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

// method A process likelihood: Mosse_like(t|M) from the sequence-free flat pass (ROOT_OBS +
// (1-E)^2 survival). Exposes a per-branch rate whose product with the branch time is the exact
// expected substitution count E[N_b] = int E[r(tau)] dtau, computed by a scalar F-downpass that
// re-solves each branch with Xia's moment hierarchy (SubstitutionModel, log-scale c2=exp(r)).
@Description("MoSSE process-only likelihood (method A): flat-pass Mosse_like(t|M), exposes E[N]/t.")
public class MosseProcessLikelihood extends MosseTreeLikelihood implements RbarProvider {

    protected double[] enBar; // per node: E[N_b]/t_b (rate*t = E[N_b])

    @Override
    public double calculateLogP() {
        if (paramsOutOfRange()) {
            logP = Double.NEGATIVE_INFINITY;
            return logP;
        }
        prepareProcessGrid(tree.getRoot());
        computeRbar = true;
        captureRootP = true;                         // makeRootFuncMosse fills survDenom, the (1-E)^2 term
        double lp = computeFlatTreeLogLikelihood();
        captureRootP = false;
        computeRbar = false;
        sharedRootP = null;
        if (Double.isInfinite(lp)) {
            logP = lp;
            return logP;
        }
        logP = lp - Math.log(survDenom);             // survival conditioning, as in the exact calcLogP
        if (enBar == null || enBar.length != tree.getNodeCount())
            enBar = new double[tree.getNodeCount()];
        computeEN(tree.getRoot());                    // fill enBar via the moment downpass
        return logP;
    }

    @Override
    public double getRbar(Node node) {
        return enBar[node.getNr()];
    }

    // Scalar process-density downpass on the low-resolution log-rate grid. Returns the node's
    // normalised F(r); for each non-root node solves the parent branch for E[N_b].
    // Sibling subtrees are independent, so the traversal fork-joins across them; the
    // per-branch CN solves and tip densities are pure (locals only), so this is safe.
    private static final ForkJoinPool EN_POOL = new ForkJoinPool(Math.max(1,
            Integer.getInteger("mosse.enThreads", Runtime.getRuntime().availableProcessors())));

    private double[] computeEN(Node node) {
        return EN_POOL.invoke(new ENTask(node));
    }

    private class ENTask extends RecursiveTask<double[]> {
        final Node node;
        ENTask(Node node) { this.node = node; }
        @Override
        protected double[] compute() {
            final int nE = treeModel.numEntries_l;
            double[] F;
            if (node.isLeaf()) {
                double start = startSubsRate_l + treeModel.padLeft_l * dx_l;
                F = tipModel.getTipLikelihoods(getTraits(node), nE, start, dx_l).clone();
            } else {
                ENTask left = new ENTask(node.getChild(0));
                left.fork();
                double[] fR = new ENTask(node.getChild(1)).compute();
                double[] fL = left.join();
                F = new double[nE];
                for (int i = 0; i < nE; i++) F[i] = lambdas_l[i] * fL[i] * fR[i]; // speciation combine
            }
            normalise(F);
            if (node.isRoot()) return F;
            return solveBranchEN(node, F, nE);
        }
    }

    private double[] solveBranchEN(Node node, double[] F, int nE) {
        final double[] xg = cached_x_l;
        final double t = node.getLength();
        if (t <= 0.0) { enBar[node.getNr()] = 0.0; return F; }

        final double[] fT = F;
        DoubleUnaryOperator fTfun = r -> interp(r, xg, fT, nE);
        DoubleBinaryOperator phi = (r, tau) -> treeModel.drift;
        DoubleBinaryOperator sig2 = (r, tau) -> treeModel.diffusion;
        DoubleUnaryOperator lam = r -> { double[] o = new double[1]; lambdaFunc.getY(new double[]{Math.exp(r)}, o); return o[0]; };
        int Nt = Math.max(8, (int) Math.ceil(t / deltaT));
        SubstitutionModel.Result res = SubstitutionModel.solveFMeanNVarN(
                phi, sig2, lam, mus_l[0], 0.0, t, xg[0], xg[nE - 1], nE, Nt, fTfun, logScale);

        double num = 0.0, den = 0.0;
        for (int i = 0; i < nE; i++) {
            double fi = res.fCurve[i], mi = res.meanNCurve[i];
            if (fi > 0.0 && Double.isFinite(mi)) { num += mi * fi; den += fi; } // avoid NaN*0 in tails
        }
        double en = (den > 0.0) ? num / den : Double.NaN; // E[N_b], density-weighted
        enBar[node.getNr()] = Double.isFinite(en) ? en / t : rbar[node.getNr()]; // fall back to r-bar
        double[] up = res.fCurve.clone();
        normalise(up);
        return up;
    }

    private static void normalise(double[] f) {
        double s = 0.0;
        for (double v : f) s += v;
        if (s > 0.0) for (int i = 0; i < f.length; i++) f[i] /= s;
    }

    private static double interp(double r, double[] xg, double[] f, int n) {
        if (r <= xg[0]) return f[0];
        if (r >= xg[n - 1]) return f[n - 1];
        int lo = 0, hi = n - 1;
        while (hi - lo > 1) { int m = (lo + hi) >>> 1; if (xg[m] <= r) lo = m; else hi = m; }
        double w = (r - xg[lo]) / (xg[hi] - xg[lo]);
        return f[lo] * (1.0 - w) + f[hi] * w;
    }
}
