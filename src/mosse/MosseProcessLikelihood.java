package mosse;

import mosseapprox.RbarProvider;
import beast.base.core.Description;
import beast.base.evolution.tree.Node;

// method A process likelihood: Mosse_like(t|M) from the sequence-free flat pass (ROOT_OBS +
// (1-E)^2 survival), exposing the per-branch downpass mean rate r̄_b. No per-site pruning.
@Description("MoSSE process-only likelihood (method A): flat-pass Mosse_like(t|M), exposes r-bar.")
public class MosseProcessLikelihood extends MosseTreeLikelihood implements RbarProvider {

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
        return logP;
    }

    @Override
    public double getRbar(Node node) {
        return rbar[node.getNr()];
    }
}
