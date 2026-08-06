package mosseapprox;

import beast.base.evolution.tree.Node;

// method A: supplies the per-branch downpass mean rate r-bar (l_b = r-bar * t_b).
public interface RbarProvider {
    double getRbar(Node node);
}
