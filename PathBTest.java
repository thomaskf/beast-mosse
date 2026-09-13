import beast.base.parser.XMLParser; import beast.base.core.BEASTInterface;
import beast.base.inference.MCMC; import beast.base.inference.State; import beast.base.inference.Distribution;
import beast.base.evolution.tree.Tree; import beast.base.evolution.tree.Node;
import java.io.File; import java.util.Map; import java.lang.reflect.Field;

// Confirms Path B (E[N] moment downpass) vs the padLeft-fixed fillRbar r-bar:
// per-branch rates should now agree (pre-fix ratio was exp(padLeft*dx) ~ 1.076),
// and reports the approximate likelihood AL under each choice of branch length.
public class PathBTest {
  static java.lang.reflect.Field findField(Class<?> c, String name) {
    for (; c != null; c = c.getSuperclass())
      try { return c.getDeclaredField(name); } catch (NoSuchFieldException e) {}
    return null;
  }
  public static void main(String[] a) throws Exception {
    XMLParser p = new XMLParser(); MCMC m = (MCMC) p.parseFile(new File(a[0]));
    Map<String, BEASTInterface> id = p.getIDMap();
    State st = m.startStateInput.get(); Distribution post = m.posteriorInput.get();
    st.initialise(); st.setPosterior(post);
    Distribution proc = (Distribution) id.get("proc");
    Distribution seq  = (Distribution) id.get("seq");
    Tree tree = (Tree) id.get("tree");

    // initial diffusion=0.005 fails the sd>=0.2*dx resolution guard; evaluate at a valid value
    beast.base.inference.parameter.RealParameter diff =
        (beast.base.inference.parameter.RealParameter) id.get("diffusion");
    diff.setValue(0, a.length > 1 ? Double.parseDouble(a[1]) : 0.2);
    st.setEverythingDirty(true);
    // refresh MosseDistribution's cached drift/diffusion fields (normally done by the MCMC loop)
    BEASTInterface tm = id.get("treemodel");
    java.lang.reflect.Method rr = tm.getClass().getDeclaredMethod("requiresRecalculation");
    rr.setAccessible(true); rr.invoke(tm);

    // dump grid geometry + guard verdict
    Object mtl = proc;
    for (String f : new String[]{"rmin","rmax","dx_h","dx_l"}) {
      java.lang.reflect.Field ff = findField(mtl.getClass(), f);
      if (ff != null) { ff.setAccessible(true); System.out.printf("%s=%.5f  ", f, ((Number) ff.get(mtl)).doubleValue()); }
    }
    System.out.println();
    java.lang.reflect.Method poor = mtl.getClass().getMethod("paramsOutOfRange");
    System.out.println("paramsOutOfRange = " + poor.invoke(mtl));

    double lpProc = proc.calculateLogP();
    Field fEn = proc.getClass().getDeclaredField("enBar"); fEn.setAccessible(true);
    double[] en = (double[]) fEn.get(proc);

    System.out.printf("proc logP (Mosse_like(t|M)) = %.4f%n%n", lpProc);
    System.out.println("node    t         E[N]/t(B)");
    double sE = 0;
    for (Node n : tree.getNodesAsArray()) {
      if (n.isRoot()) continue;
      int i = n.getNr(); double t = n.getLength();
      System.out.printf("%4d  %8.4f   %.6f%n", i, t, en[i]);
      sE += en[i] * t;
    }
    System.out.printf("%ntree totals: sum E[N] = %.5f%n%n", sE);

    // per-branch l = E[N] keyed by clade, so the vector can be matched to a newick outside
    System.out.println("CLADE_BL_START");
    for (Node n : tree.getNodesAsArray()) {
      if (n.isRoot()) continue;
      java.util.List<String> tips = new java.util.ArrayList<>();
      collectTips(n, tips);
      java.util.Collections.sort(tips);
      System.out.printf("BL\t%.10g\t%s%n", en[n.getNr()] * n.getLength(), String.join(",", tips));
    }
    System.out.println("CLADE_BL_END");

    double alB = seq.calculateLogP();
    System.out.printf("AL(l = E[N])        [Path B, wired]  = %.4f%n", alB);
    // timing + determinism check: repeated full proc evaluations
    int K = Integer.getInteger("bench.K", 5);
    long t0 = System.nanoTime();
    double acc = 0;
    for (int k = 0; k < K; k++) acc += proc.calculateLogP();
    double per = (System.nanoTime() - t0) / 1e9 / K;
    double[] en2 = (double[]) fEn.get(proc);
    double sumE = 0; for (Node n : tree.getNodesAsArray()) if (!n.isRoot()) sumE += en2[n.getNr()] * n.getLength();
    System.out.printf("BENCH: %.3f s/proc-eval (avg %d), logP=%.6f, sumEN=%.12f, enThreads=%s%n",
        per, K, acc / K, sumE, System.getProperty("mosse.enThreads", "auto"));
  }

  static void collectTips(Node n, java.util.List<String> out) {
    if (n.isLeaf()) { out.add(n.getID()); return; }
    for (Node c : n.getChildren()) collectTips(c, out);
  }
}
