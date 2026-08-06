package mosseapprox;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.branchratemodel.BranchRateModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Distribution;
import beast.base.inference.State;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

@Description("Approximate phylogenetic likelihood P(Data|b) via the IQ-TREE branch-length "
        + "Hessian (dos Reis & Yang 2011): logP = C + (b-bhat).g + 0.5 (b-bhat).H (b-bhat). "
        + "Branch lengths b are read from the tree and matched to the Hessian record by "
        + "bipartition; the two root-child edges sum into the single merged (unrooted) branch.")
public class ApproxTreeLikelihood extends Distribution {

    final public Input<TreeInterface> treeInput = new Input<>("tree",
            "tree whose branch lengths are expected substitutions per site", Input.Validate.REQUIRED);
    final public Input<String> hessianFileInput = new Input<>("hessianFile",
            "IQ-TREE .mcmctree.hessian file (bhat, gradient, Hessian)", Input.Validate.REQUIRED);
    final public Input<String> branchMapInput = new Input<>("branchMap",
            "index<TAB>bipartition file (from branch_order.py)", Input.Validate.REQUIRED);
    final public Input<Double> constInput = new Input<>("constant",
            "additive constant C = ML log-likelihood at bhat (cancels in MCMC ratios)", 0.0);
    final public Input<BranchRateModel.Base> branchRateModelInput = new Input<>("branchRateModel",
            "optional per-branch rate model: branch subs = rate(branch) x branch time (method A). "
            + "If absent, node lengths are used directly as subs/site.");

    private int m;
    private double[] bhat, grad;
    private double[][] H;
    private double C;
    private String[] bipByIndex;
    private Map<String, Integer> bip2idx;
    private BranchRateModel.Base clock;

    @Override
    public void initAndValidate() {
        parseHessian(hessianFileInput.get());
        parseBranchMap(branchMapInput.get());
        if (bipByIndex.length != m)
            throw new IllegalArgumentException("branchMap has " + bipByIndex.length
                    + " entries but Hessian has " + m + " branches");
        bip2idx = new HashMap<>();
        for (int i = 0; i < m; i++) bip2idx.put(bipByIndex[i], i);
        C = constInput.get();
        clock = branchRateModelInput.get();
    }

    @Override
    public double calculateLogP() {
        TreeInterface tree = treeInput.get();
        List<String> allTaxa = new ArrayList<>();
        for (Node node : tree.getNodesAsArray()) if (node.isLeaf()) allTaxa.add(node.getID());

        double[] b = new double[m];
        for (Node node : tree.getNodesAsArray()) {
            if (node.isRoot()) continue;
            String bp = canonicalBip(leafSet(node), allTaxa);
            Integer idx = bip2idx.get(bp);
            if (idx == null)
                throw new IllegalArgumentException("no Hessian branch for bipartition " + bp);
            double subs = (clock == null) ? node.getLength()
                    : clock.getRateForBranch(node) * node.getLength();
            b[idx] += subs;                        // root children accumulate into merged branch
        }

        double[] d = new double[m];
        for (int i = 0; i < m; i++) d[i] = b[i] - bhat[i];
        double lp = C;
        for (int i = 0; i < m; i++) {
            lp += d[i] * grad[i];
            double s = 0;
            for (int j = 0; j < m; j++) s += H[i][j] * d[j];
            lp += 0.5 * d[i] * s;
        }
        logP = lp;
        return logP;
    }

    private Set<String> leafSet(Node node) {
        Set<String> s = new HashSet<>();
        if (node.isLeaf()) { s.add(node.getID()); return s; }
        for (Node leaf : node.getAllLeafNodes()) s.add(leaf.getID());
        return s;
    }

    // canonical bipartition label, matching branch_order.py bip(): the smaller side by
    // (size, then lexicographic), sorted, as "{a,b,c}".
    private String canonicalBip(Set<String> tips, List<String> allTaxa) {
        Set<String> comp = new HashSet<>(allTaxa);
        comp.removeAll(tips);
        List<String> a = new ArrayList<>(tips);
        List<String> b = new ArrayList<>(comp);
        Collections.sort(a);
        Collections.sort(b);
        List<String> side;
        if (a.size() < b.size()) side = a;
        else if (a.size() > b.size()) side = b;
        else side = (cmpList(a, b) <= 0) ? a : b;
        return "{" + String.join(",", side) + "}";
    }

    private static int cmpList(List<String> a, List<String> b) {
        for (int i = 0; i < a.size(); i++) {
            int c = a.get(i).compareTo(b.get(i));
            if (c != 0) return c;
        }
        return 0;
    }

    private void parseHessian(String path) {
        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            List<String> lines = new ArrayList<>();
            for (String ln; (ln = r.readLine()) != null; ) lines.add(ln);
            int i = 0, ntaxa = -1;
            while (i < lines.size()) {
                String s = lines.get(i++).trim();
                if (s.matches("\\d+")) { ntaxa = Integer.parseInt(s); break; }
            }
            m = 2 * ntaxa - 3;
            while (i < lines.size() && !lines.get(i).contains("(")) i++;
            i++;                                                   // skip the tree line
            List<Double> nums = new ArrayList<>();
            for (; i < lines.size(); i++) {
                if (lines.get(i).trim().toLowerCase().startsWith("hessian")) { i++; break; }
                for (String tok : lines.get(i).trim().split("\\s+"))
                    if (!tok.isEmpty()) nums.add(Double.parseDouble(tok));
            }
            bhat = new double[m];
            grad = new double[m];
            for (int k = 0; k < m; k++) bhat[k] = nums.get(k);
            for (int k = 0; k < m; k++) grad[k] = nums.get(m + k);
            List<Double> hn = new ArrayList<>();
            for (; i < lines.size() && hn.size() < m * m; i++)
                for (String tok : lines.get(i).trim().split("\\s+"))
                    if (!tok.isEmpty()) hn.add(Double.parseDouble(tok));
            H = new double[m][m];
            for (int a = 0; a < m; a++)
                for (int bb = 0; bb < m; bb++) H[a][bb] = hn.get(a * m + bb);
        } catch (Exception e) {
            throw new RuntimeException("failed to parse Hessian file " + path, e);
        }
    }

    private void parseBranchMap(String path) {
        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            Map<Integer, String> map = new HashMap<>();
            int max = -1;
            for (String ln; (ln = r.readLine()) != null; ) {
                ln = ln.trim();
                if (ln.isEmpty() || ln.startsWith("index")) continue;
                String[] p = ln.split("\\s+", 2);
                int idx = Integer.parseInt(p[0]);
                map.put(idx, p[1].trim());
                if (idx > max) max = idx;
            }
            bipByIndex = new String[max + 1];
            for (Map.Entry<Integer, String> e : map.entrySet()) bipByIndex[e.getKey()] = e.getValue();
        } catch (Exception e) {
            throw new RuntimeException("failed to parse branchMap file " + path, e);
        }
    }

    @Override protected boolean requiresRecalculation() { return true; }
    @Override public List<String> getArguments() { return null; }
    @Override public List<String> getConditions() { return null; }
    @Override public void sample(State state, Random random) { }
}
