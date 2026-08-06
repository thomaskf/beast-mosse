package mosseapprox;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.branchratemodel.BranchRateModel;
import beast.base.evolution.tree.Node;

// method A: per-branch rate = MoSSE downpass mean rate r-bar from an RbarProvider. The provider
// must be evaluated before this is queried (list the process likelihood ahead of the approx one).
@Description("Per-branch rate = MoSSE downpass mean rate r-bar (method A: l = r-bar*t).")
public class MosseRbarClockModel extends BranchRateModel.Base {

    final public Input<RbarProvider> providerInput = new Input<>("rbarProvider",
            "MoSSE process likelihood that supplies the per-branch mean rate", Input.Validate.REQUIRED);

    private RbarProvider provider;

    @Override
    public void initAndValidate() {
        provider = providerInput.get();
    }

    @Override
    public double getRateForBranch(Node node) {
        return provider.getRbar(node);
    }

    @Override
    protected boolean requiresRecalculation() {
        return true;
    }
}
