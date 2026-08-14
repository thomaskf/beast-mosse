import beast.base.parser.XMLParser; import beast.base.core.BEASTInterface;
import beast.base.inference.MCMC; import beast.base.inference.State; import beast.base.inference.Distribution;
import beast.base.evolution.tree.Tree; import beast.base.evolution.tree.Node;
import java.io.File; import java.util.Map; import java.lang.reflect.*;
import java.util.function.DoubleUnaryOperator;
import mosseapprox.SubstitutionModel; import mosseapprox.FFTDiffusionModel;

// Benchmark: (a) phase split of one Path-B evaluation — native JNI/FFTW flat pass vs
// Java E[N] downpass (serial); (b) matched single-branch engine duel — Java CN
// (SubstitutionModel) vs Java FFT (FFTDiffusionModel) on identical grid/steps/inputs.
public class FFTBench {
  static Field f(Class<?> c, String n) {
    for (; c != null; c = c.getSuperclass())
      try { Field x = c.getDeclaredField(n); x.setAccessible(true); return x; } catch (NoSuchFieldException e) {}
    return null;
  }
  static Method m(Class<?> c, String n, Class<?>... a) {
    for (; c != null; c = c.getSuperclass())
      try { Method x = c.getDeclaredMethod(n, a); x.setAccessible(true); return x; } catch (NoSuchMethodException e) {}
    return null;
  }
  public static void main(String[] a) throws Exception {
    XMLParser p = new XMLParser(); MCMC mc = (MCMC) p.parseFile(new File(a[0]));
    Map<String, BEASTInterface> id = p.getIDMap();
    State st = mc.startStateInput.get(); st.initialise(); st.setPosterior(mc.posteriorInput.get());
    Distribution proc = (Distribution) id.get("proc");
    Tree tree = (Tree) id.get("tree");
    beast.base.inference.parameter.RealParameter diff =
        (beast.base.inference.parameter.RealParameter) id.get("diffusion");
    diff.setValue(0, 0.2);
    BEASTInterface tm = id.get("treemodel");
    Method rr = m(tm.getClass(), "requiresRecalculation"); rr.invoke(tm);

    int K = Integer.getInteger("bench.K", 5);
    // warm-up + phase split: total serial eval vs computeEN alone
    proc.calculateLogP();
    Method cEN = m(proc.getClass(), "computeEN", Node.class);
    long t0 = System.nanoTime();
    for (int k = 0; k < K; k++) proc.calculateLogP();
    double tTot = (System.nanoTime() - t0) / 1e9 / K;
    t0 = System.nanoTime();
    for (int k = 0; k < K; k++) cEN.invoke(proc, tree.getRoot());
    double tEN = (System.nanoTime() - t0) / 1e9 / K;
    System.out.printf("PHASES (serial, per evaluation): total=%.3f s  javaEN=%.3f s  nativeFlat+AL≈%.3f s%n",
        tTot, tEN, tTot - tEN);

    // matched single-branch duel: longest branch, identical grid/steps/inputs
    double[] xg = (double[]) f(proc.getClass(), "cached_x_l").get(proc);
    int nE = ((Number) f(tm.getClass(), "numEntries_l").get(tm)).intValue();
    double deltaT = ((Number) f(proc.getClass(), "deltaT").get(proc)).doubleValue();
    double[] mus = (double[]) f(proc.getClass(), "mus_l").get(proc);
    Object lambdaFunc = f(proc.getClass(), "lambdaFunc").get(proc);
    Method getY = m(lambdaFunc.getClass(), "getY", double[].class, double[].class);
    DoubleUnaryOperator lam = r -> {
      try { double[] o = new double[1]; getY.invoke(lambdaFunc, new double[]{Math.exp(r)}, o); return o[0]; }
      catch (Exception e) { throw new RuntimeException(e); }
    };
    Node longest = null;
    for (Node n : tree.getNodesAsArray())
      if (!n.isRoot() && (longest == null || n.getLength() > longest.getLength())) longest = n;
    double t = longest.getLength();
    int Nt = Math.max(8, (int) Math.ceil(t / deltaT));
    final double[] F0 = new double[nE];
    double c = (xg[0] + xg[nE-1]) / 2, s2 = 0.04;
    for (int i = 0; i < nE; i++) F0[i] = Math.exp(-(xg[i]-c)*(xg[i]-c)/(2*s2));
    DoubleUnaryOperator fT = r -> {
      if (r <= xg[0]) return F0[0];
      if (r >= xg[nE-1]) return F0[nE-1];
      int lo = 0, hi = nE-1;
      while (hi-lo > 1) { int mid = (lo+hi)>>>1; if (xg[mid] <= r) lo = mid; else hi = mid; }
      double w = (r-xg[lo])/(xg[hi]-xg[lo]);
      return F0[lo]*(1-w)+F0[hi]*w;
    };
    double drift = 0.0, sig2 = 0.2, mu = mus[0];
    int R = Integer.getInteger("bench.R", 50);
    // warm-up both
    SubstitutionModel.Result rc = SubstitutionModel.solveFMeanNVarN(
        (x,tau)->drift, (x,tau)->sig2, lam, mu, 0.0, t, xg[0], xg[nE-1], nE, Nt, fT, true);
    SubstitutionModel.Result rf = FFTDiffusionModel.solveFMeanNVarNFFT(
        drift, sig2, lam, mu, 0.0, t, xg[0], xg[nE-1], nE, Nt, fT);
    t0 = System.nanoTime();
    for (int k = 0; k < R; k++) SubstitutionModel.solveFMeanNVarN(
        (x,tau)->drift, (x,tau)->sig2, lam, mu, 0.0, t, xg[0], xg[nE-1], nE, Nt, fT, true);
    double tCN = (System.nanoTime()-t0)/1e9/R;
    t0 = System.nanoTime();
    for (int k = 0; k < R; k++) FFTDiffusionModel.solveFMeanNVarNFFT(
        drift, sig2, lam, mu, 0.0, t, xg[0], xg[nE-1], nE, Nt, fT);
    double tFFT = (System.nanoTime()-t0)/1e9/R;
    // agreement: E[N] from both (density-weighted mean of meanNCurve)
    double nc=0, dc=0, nf=0, df=0;
    for (int i = 0; i < nE; i++) {
      if (rc.fCurve[i] > 0 && Double.isFinite(rc.meanNCurve[i])) { nc += rc.meanNCurve[i]*rc.fCurve[i]; dc += rc.fCurve[i]; }
      if (rf.fCurve[i] > 0 && Double.isFinite(rf.meanNCurve[i])) { nf += rf.meanNCurve[i]*rf.fCurve[i]; df += rf.fCurve[i]; }
    }
    System.out.printf("DUEL (branch t=%.4f, Nr=%d, Nt=%d, avg of %d): javaCN=%.4f ms  javaFFT=%.4f ms  ratio(CN/FFT)=%.2f%n",
        t, nE, Nt, R, tCN*1e3, tFFT*1e3, tCN/tFFT);
    System.out.printf("AGREEMENT: E[N] CN=%.8f  FFT=%.8f  rel.diff=%.2e%n",
        nc/dc, nf/df, Math.abs(nc/dc - nf/df)/Math.abs(nc/dc));

    // MATCHED native branch solve: same branch, dt=0.003 -> ~same step count as the Java duel,
    // native 5-column (E + 4 D) FFTW integration at nx=1024 vs Java FFT's ~5 propagates/step.
    double[] lamL = (double[]) f(proc.getClass(), "lambdas_l").get(proc);
    double[] ratesL = (double[]) f(proc.getClass(), "rates_l").get(proc);
    double[] qF = (double[]) f(proc.getClass(), "qFlat").get(proc);
    double[] eV = (double[]) f(proc.getClass(), "eVal").get(proc);
    double[] eVc = (double[]) f(proc.getClass(), "eVec").get(proc);
    double[] iEv = (double[]) f(proc.getClass(), "iEvec").get(proc);
    boolean hasE = (Boolean) f(proc.getClass(), "hasEigen").get(proc);
    int nxL = ((Number) f(proc.getClass(), "numRateBins_l").get(proc)).intValue();
    int padL = ((Number) f(tm.getClass(), "padLeft_l").get(tm)).intValue();
    int padR = ((Number) f(tm.getClass(), "padRight_l").get(tm)).intValue();
    double dt = ((Number) f(tm.getClass(), "dt").get(tm)).doubleValue();
    int ntNative = (int) Math.floor(t / dt);
    double[] vars = new double[5 * nxL];
    for (int col = 1; col < 5; col++)
      for (int i = 0; i < nE; i++) vars[col * nxL + padL + i] = F0[i] * 0.25;
    Method doInt = m(tm.getClass(), "doIntegration", double[].class, double[].class, double[].class,
        double[].class, double[].class, double[].class, double[].class, double[].class, boolean.class,
        double[].class, long.class, double.class, double.class, int.class, double.class,
        int.class, int.class, boolean.class, int.class);
    doInt.invoke(tm, vars, lamL, mus, ratesL, qF, eV, eVc, iEv, hasE, null, Long.MIN_VALUE,
        0.0, 0.2, ntNative, dt, padL, padR, true, 0); // warm-up
    long tn0 = System.nanoTime();
    for (int k = 0; k < R; k++)
      doInt.invoke(tm, vars, lamL, mus, ratesL, qF, eV, eVc, iEv, hasE, null, Long.MIN_VALUE,
          0.0, 0.2, ntNative, dt, padL, padR, true, 0);
    double tNat = (System.nanoTime() - tn0) / 1e9 / R;
    System.out.printf("NATIVE MATCHED: branch t=%.4f, nt=%d steps, nx=%d, 5 cols: %.2f ms/branch (avg %d)%n",
        t, ntNative, nxL, tNat * 1e3, R);
  }
}
