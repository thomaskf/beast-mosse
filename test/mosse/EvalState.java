package mosse;

import java.io.*;
import java.util.*;
import java.util.concurrent.Executors;
import beast.base.parser.XMLParser;
import beast.base.inference.MCMC;
import beast.base.inference.State;
import beast.base.inference.StateNode;
import beast.base.inference.Distribution;
import beast.base.inference.CompoundDistribution;
import beast.base.inference.parameter.RealParameter;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeParser;
import beast.base.core.ProgramStatus;

// Deterministic single-state evaluation with DIAG breakdown.
// args: <xml> <statefile> [Nrepeat]
// statefile format (key=value lines):
//   newick=<newick with integer labels 1..N; mapped to spN>
//   beta=..  epsilon=..  subst=..  drift=..  diffusion=..  kappa=..
//   curveYBaseValue=..  curveMaxY=..  linearGrowthRate=..  yV=..
public class EvalState {
    public static void main(String[] args) throws Exception {
        int nrep = args.length > 2 ? Integer.parseInt(args[2]) : 3;
        int nthreads = Integer.getInteger("mosse.threads", 1);
        ProgramStatus.m_nThreads = nthreads;
        ProgramStatus.g_exec = Executors.newFixedThreadPool(nthreads);

        MCMC mcmc = (MCMC) new XMLParser().parseFile(new File(args[0]));
        State state = mcmc.startStateInput.get();
        Distribution posterior = mcmc.posteriorInput.get();

        Tree tree = null;
        Map<String,RealParameter> params = new HashMap<>();
        for (StateNode sn : state.stateNodeInput.get()) {
            String id = sn.getID();
            if (sn instanceof Tree && "tree".equals(id)) tree = (Tree) sn;
            else if (sn instanceof RealParameter) params.put(id, (RealParameter) sn);
        }
        if (tree == null) throw new RuntimeException("tree StateNode not found");

        // read state file
        Map<String,String> kv = new HashMap<>();
        BufferedReader br = new BufferedReader(new FileReader(args[1]));
        String line;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            int eq = line.indexOf('=');
            kv.put(line.substring(0,eq).trim(), line.substring(eq+1).trim());
        }
        br.close();

        // map integer-labelled newick -> spN labels
        String newick = kv.get("newick");
        // translate: token boundary integer -> spInteger. Newick uses (,):;
        // labels are integers immediately followed by ':'
        StringBuilder sb = new StringBuilder();
        int i = 0;
        while (i < newick.length()) {
            char c = newick.charAt(i);
            if ((c=='(' || c==',') ) {
                sb.append(c); i++;
                // read following integer label (digits) up to ':'
                int j = i;
                while (j < newick.length() && Character.isDigit(newick.charAt(j))) j++;
                if (j > i && j < newick.length() && newick.charAt(j)==':') {
                    sb.append("sp").append(newick.substring(i,j));
                    i = j;
                } 
            } else { sb.append(c); i++; }
        }
        String mapped = sb.toString();
        // strip internal-node numeric labels: ")<digits>" -> ")" so TreeParser
        // does not treat them as node numbers (they collide with node count)
        mapped = mapped.replaceAll("\\)\\d+", ")");

        // taxa ordering from the State tree
        List<String> taxa = new ArrayList<>();
        for (beast.base.evolution.tree.Node leaf : tree.getExternalNodes())
            taxa.add(leaf.getID());

        // transfer heights by clade matching (topology fixed)
        TreeParser tp = new TreeParser(taxa, mapped, 0, false);
        tp.initAndValidate();
        Map<String,Double> cladeHeight = new HashMap<>();
        collectHeights(tp.getRoot(), cladeHeight);
        applyHeights(tree.getRoot(), cladeHeight);

        // set params
        String[] ids = {"beta","epsilon","subst","drift","diffusion","kappa",
                        "curveYBaseValue","curveMaxY","linearGrowthRate","yV"};
        for (String id : ids) {
            if (!kv.containsKey(id)) throw new RuntimeException("missing param: "+id);
            if (!params.containsKey(id)) throw new RuntimeException("param StateNode not found: "+id);
            params.get(id).setValue(Double.parseDouble(kv.get(id)));
        }

        Distribution like = find(posterior, "likelihood");
        Distribution moss = find(posterior, "mossetreelikelihood");
        for (int r = 1; r <= nrep; r++) {
            state.setEverythingDirty(true);
            double post = state.robustlyCalcPosterior(posterior);
            System.out.println("eval#" + r + " posterior=" + post
                + " likelihood=" + (like!=null?like.getCurrentLogP():Double.NaN)
                + " mossetreelikelihood=" + (moss!=null?moss.getCurrentLogP():Double.NaN));
        }
        ProgramStatus.g_exec.shutdown();
    }

    static Distribution find(Distribution d, String id) {
        if (id.equals(d.getID())) return d;
        if (d instanceof CompoundDistribution)
            for (Distribution c : ((CompoundDistribution)d).pDistributions.get()) { Distribution rr = find(c,id); if (rr!=null) return rr; }
        return null;
    }
    static String collectHeights(beast.base.evolution.tree.Node n, Map<String,Double> out) {
        if (n.isLeaf()) { String k = n.getID(); out.put(k, n.getHeight()); return k; }
        List<String> keys = new ArrayList<>();
        for (beast.base.evolution.tree.Node c : n.getChildren()) keys.add(collectHeights(c, out));
        Collections.sort(keys);
        String key = String.join(",", keys);
        out.put(key, n.getHeight());
        return key;
    }
    static String applyHeights(beast.base.evolution.tree.Node n, Map<String,Double> src) {
        if (n.isLeaf()) return n.getID();
        List<String> keys = new ArrayList<>();
        for (beast.base.evolution.tree.Node c : n.getChildren()) keys.add(applyHeights(c, src));
        Collections.sort(keys);
        String key = String.join(",", keys);
        Double h = src.get(key);
        if (h == null) throw new RuntimeException("clade not matched: " + key);
        n.setHeight(h);
        return key;
    }
}
