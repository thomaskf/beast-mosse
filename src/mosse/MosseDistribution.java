package mosse;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.lang.ref.Cleaner;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.TreeDistribution;
import beast.base.inference.State;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;

/**
 * @author Kylie Chen
 * @author Thomas Wong
 */

@Description("Mosse tree model")
public class MosseDistribution extends TreeDistribution implements AutoCloseable {

	final public Input<IntegerParameter> nxInput = new Input<>("nx", "number of bins for substitution rate",
			new IntegerParameter("1024"));
	
	// change dx from an input value to a calculated value
	// final public Input<RealParameter> dxInput = new Input<>("dx", "distance between xs", new RealParameter("0.0001"));
	
	final public Input<RealParameter> driftInput = new Input<>("drift", "drift parameter", new RealParameter("0.0"));
	final public Input<RealParameter> diffusionInput = new Input<>("diffusion", "diffusion parameter",
			new RealParameter("0.001"));
	final public Input<RealParameter> dtInput = new Input<>("dt", "time interval dt", new RealParameter("0.01"));
	final public Input<IntegerParameter> widthInput = new Input<>("width", "width of the kernel for convolution",
			new IntegerParameter("10"));
	final public Input<IntegerParameter> resolutionInput = new Input<>("resolution",
			"scale factor for resolution of bins", new IntegerParameter("4"));

	// PUNC: punctuational-evolution amplitude. P = I + a*Q at speciation events.
	// When a = 0 the punc integrator reduces to the original Mosse algorithm.
	final public Input<RealParameter> aInput = new Input<>("a",
			"punctuational evolution amplitude (P = I + a*Q); a >= 0; default 0",
			new RealParameter("0.0"));
	
	// number of cpu threads (not for this class)
    final public Input<Integer> threadsInput =
            new Input<>("threads", "Number of threads for within-node parallelism", 1);
	
	final public int FLAG_FFTW3_DEFAULT = 0; // FFTW_MEASURE

	protected int resolution;
	protected int nx;
	protected double dx_h; // assigned by initFFTPtrs()
	protected int numRateBins_h;
	protected int numRateBins_l;
	
	protected int padLeft_h; // padLeft for high resolution
	protected int padLeft_l; // padLeft for low resolution
	protected int padRight_h; // padRight for high resolution
	protected int padRight_l; // padRight for low resolution
	protected int numEntries_h;
	protected int numEntries_l;

	protected double drift;
	protected double diffusion;
	protected double dt;
	protected int width;
	protected double a; // PUNC: punctuational amplitude (read in initAndValidate)
	
	// number of threads
	protected int numThreads;
	ArrayList<Long> ptr_l_pool; // ptr for low resolution
	ArrayList<Long> ptr_h_pool; // ptr for high resolution
	
	// for storing during mcmc
	protected double storedrift;
	protected double storedDiffusion;
	protected double storedA;
	protected int storedPadLeft_h;
	protected int storedPadLeft_l;
	protected int storedPadRight_h;
	protected int storedPadRight_l;
	protected int storedNumEntries_h;
	protected int storedNumEntries_l;

	private static final Cleaner CLEANER = Cleaner.create();
	private Cleaner.Cleanable cleanable;

	/**
	 * Static nested class that holds only the native pointer lists and a bridge to
	 * the native finalizer. Keeping this separate from MosseDistribution ensures
	 * the Cleaner-registered action holds NO reference to the enclosing instance,
	 * which is required for the Cleaner to ever fire (bug 1.1 fix).
	 */
	private static final class CleanerState implements Runnable {
		private final ArrayList<Long> lPool;
		private final ArrayList<Long> hPool;
		private final java.util.function.LongConsumer nativeFinalizer;

		CleanerState(ArrayList<Long> lPool, ArrayList<Long> hPool,
				java.util.function.LongConsumer nativeFinalizer) {
			this.lPool = lPool;
			this.hPool = hPool;
			this.nativeFinalizer = nativeFinalizer;
		}

		@Override
		public void run() {
			for (int i = 0; i < lPool.size(); i++) {
				nativeFinalizer.accept(lPool.get(i));
				nativeFinalizer.accept(hPool.get(i));
			}
			lPool.clear();
			hPool.clear();
		}
	}

	static {
		System.loadLibrary("test");
	}

	/**
	 * initialize mosse C object
	 *
	 * @param nx       number of bins for substitution rate
	 * @param dx       distance between xs
	 * @param array_nd plan dimensions for FFTW3 integration
	 * @param flags    flags for FFTW3 integration
	 * @return mosse object pointer
	 */
	private native long makeMosseFFT(int nx, double dx, int[] array_nd, int flags);

	/**
	 * destroy mosse C object
	 *
	 * @param obj_ptr mosse object pointer
	 */
	private native void mosseFinalize(long obj_ptr);

	/**
	 * NEW signature for the punctuational integrator.
	 * Compared to the original:
	 *   - r:  per-bin substitution rate vector (length numEntries), NEW
	 *   - a:  scalar punctuational amplitude (>= 0), NEW
	 *   - q:  single 4x4 substitution-rate matrix flattened to length 16
	 *         — replaces the old per-bin precomputed P stack.
	 */
	private native double[] doIntegrateMosse(long obj_ptr, double[] vars, double[] lambda, double[] mu,
			double[] r, double a,
			double drift, double diffusion,
			double[] q,
			double[] eVal, double[] eVec, double[] iEvec, boolean useEigen,
			double[] eQCache, long eigenGeneration,
			int nt, double dt, int pad_left, int pad_right);

	@Override
	public void initAndValidate() {
		if (resolutionInput.get() != null) {
			resolution = resolutionInput.get().getValue();
		} else {
			resolution = resolutionInput.defaultValue.getValue();
		}
		if (nxInput.get() != null) {
			nx = nxInput.get().getValue();
		} else {
			nx = nxInput.defaultValue.getValue();
		}
		
		numRateBins_h = nx * resolution;
		numRateBins_l = nx;

		drift = driftInput.get().getValue();
		diffusion = diffusionInput.get().getValue();
		dt = dtInput.get().getValue();
		a = aInput.get().getValue(); // PUNC

		width = widthInput.get().getValue();
		
		if (threadsInput.get() != null) {
			numThreads = threadsInput.get().intValue();
		}
	}
	
	/**
	 * Static bridge so CleanerState can invoke the native finalizer without
	 * holding a reference to the MosseDistribution instance.
	 * The MosseDistribution instance is passed explicitly only at the moment of
	 * the call; CleanerState stores only this method reference (which is a
	 * reference to this static-bridge method, not to 'this').
	 */
	private void mosseFinalizeBridge(long ptr) {
		mosseFinalize(ptr);
	}

	public void initFFTPtrs(double dx_h) {
		int[] nd = { 5 };
		int flags = FLAG_FFTW3_DEFAULT;
		ptr_l_pool = new ArrayList<Long>();
		ptr_h_pool = new ArrayList<Long>();
		for (int i = 0; i < numThreads; i++) {
			long ptr_l = makeMosseFFT(nx, dx_h * resolution, nd, flags);
			long ptr_h = makeMosseFFT(nx * resolution, dx_h, nd, flags);
			ptr_l_pool.add(ptr_l);
			ptr_h_pool.add(ptr_h);
		}
		this.dx_h = dx_h;

		// Reset closed so that a subsequent close() call (e.g. from
		// MosseTreeLikelihoodMT re-init) will actually release these new pointers.
		closed = false;

		// Register a Cleaner using the static-nested CleanerState so that the
		// cleanup action does NOT capture 'this' through an anonymous lambda
		// (which would prevent the object from ever becoming unreachable).
		cleanable = CLEANER.register(this,
				new CleanerState(ptr_l_pool, ptr_h_pool, this::mosseFinalizeBridge));
	}
	
	public synchronized void resizePtrPool(int newsize) {
		int[] nd = { 5 };
		int flags = FLAG_FFTW3_DEFAULT;
		if (newsize > ptr_l_pool.size()) {
			int k = newsize - ptr_l_pool.size();
			for (int i = 0; i < k; i++) {
				long ptr_l = makeMosseFFT(nx, dx_h * resolution, nd, flags);
				long ptr_h = makeMosseFFT(nx * resolution, dx_h, nd, flags);
				ptr_l_pool.add(ptr_l);
				ptr_h_pool.add(ptr_h);
			}
		}
	}

	public void computePadNumEntries(double dx) {
		double mean = drift * dt;
		double sd = Math.sqrt(diffusion * dt);
		
		// low resolution
		padLeft_l = Math.abs((int) Math.ceil(-(mean - width * sd) / dx / resolution));
		padRight_l = Math.abs((int) Math.ceil((mean + width * sd) / dx / resolution));
		// minimums value are 1
		if (padLeft_l < 1) padLeft_l = 1;
		if (padRight_l < 1) padRight_l = 1;	
		numEntries_l = numRateBins_l - padLeft_l - padRight_l - 1;
		
		// high resolution
		padLeft_h = Math.abs((int) Math.ceil(-(mean - width * sd) / dx / resolution) * resolution);
		padRight_h = Math.abs((int) Math.ceil((mean + width * sd) / dx / resolution) * resolution);
		// minimums value are 1 * resolution
		if (padLeft_h < resolution) padLeft_h = resolution;
		if (padRight_h < resolution) padRight_h = resolution;
		numEntries_h = numRateBins_h - padLeft_h - padRight_h - 1;
		
		// System.out.println("padLeft(l,h):" + padLeft_l + "," + padLeft_h);
		// System.out.println("padRight(l,h):" + padRight_l + "," + padRight_h);
		// System.out.println("numEntries(l,h):" + numEntries_l + "," + numEntries_h);
	}

	/*
	 * public void calculateBranchLogP(double branchTime, double[] vars, double[]
	 * lambda, double[] mu, double[] Q, double[] result, boolean lowResolution) { //
	 * double logP = 0.0; // getting parameter values int[] nd = {5};
	 *
	 * int nt = (int) Math.ceil(branchTime / dt); double[] the_result;
	 *
	 * if (lowResolution) { the_result = doIntegration(nx, dx * resolution, nd,
	 * FLAG_FFTW3_DEFAULT, vars, lambda, mu, drift, diffusion, Q, nt, dt, padLeft_l,
	 * padRight_l); } else { the_result = doIntegration(nx * resolution, dx, nd,
	 * FLAG_FFTW3_DEFAULT, vars, lambda, mu, drift, diffusion, Q, nt, dt, padLeft_h,
	 * padRight_h); }
	 *
	 * System.arraycopy(the_result, 0, result, 0, the_result.length); }
	 */

	/**
	 * Integrate one branch under the punctuational integrator.
	 *
	 * Compared to the non-punc version this method takes:
	 *   - r:  per-bin substitution rate vector (length numEntries), the rate axis
	 *         (e.g. r[i] = dx*(i+1) on linear scale)
	 *   - q:  single 4x4 substitution-rate matrix flattened to length 16,
	 *         instead of the old per-bin precomputed P stack.
	 */
	public double[] calculateBranchLogP(double branchTime, double[] vars, double[] lambda, double[] mu,
			double[] r, double[] q,
			double[] eVal, double[] eVec, double[] iEvec, boolean useEigen,
			double[] eQCache, long eigenGeneration,
			boolean lowResolution, int threadID) {
		int nt = (int) Math.ceil(branchTime / dt);
		double stepDt = dt;
		double[] eQCacheArg = eQCache;
		long genArg = eigenGeneration;
		if (branchTime < dt) { // sub-dt branch: one step of its true length, not a full dt step
			nt = 1;
			stepDt = branchTime;
			eQCacheArg = null;       // force native to rebuild exp(Q*r*stepDt) for this branch
			genArg = Long.MIN_VALUE; // distinct generation so the eQ-cache gate re-enters (valid=0)
		}
		// JNI doIntegrateMosse already returns a freshly-allocated jdoubleArray of
		// exactly the right length (obj->nx * nd == vars.length). Return it directly;
		// the previous defensive new double[vars.length] + System.arraycopy added
		// ~vars.length doubles of young-gen garbage per branch call (≈ 200 GB/step
		// of pure allocator pressure at 200-thread concurrency).
		if (lowResolution) {
			return doIntegration(vars, lambda, mu, r, q,
					eVal, eVec, iEvec, useEigen, eQCacheArg, genArg, drift, diffusion,
					nt, stepDt, padLeft_l, padRight_l, lowResolution, threadID);
		} else {
			return doIntegration(vars, lambda, mu, r, q,
					eVal, eVec, iEvec, useEigen, eQCacheArg, genArg, drift, diffusion,
					nt, stepDt, padLeft_h, padRight_h, lowResolution, threadID);
		}
	}

	/**
	 * Perform punctuational integration along a single branch.
	 *
	 * @param vars      array of tips or partials
	 * @param lambda    array of birth-rates
	 * @param mu        array of death-rates
	 * @param r         per-bin substitution rate (length numEntries) — NEW
	 * @param q         flat 4x4 substitution-rate matrix (length 16) — replaces old Q stack
	 * @param drift     drift parameter
	 * @param diffusion diffusion parameter
	 * @param nt        number of time steps
	 * @param dt_max    maximum time step
	 * @param pad_left  padding size left of the kernel (zero padding)
	 * @param pad_right padding size right of the kernel (zero padding)
	 * @return partial probabilities
	 */
	public double[] doIntegration(double[] vars, double[] lambda, double[] mu,
			double[] r, double[] q,
			double[] eVal, double[] eVec, double[] iEvec, boolean useEigen,
			double[] eQCache, long eigenGeneration,
			double drift, double diffusion,
			int nt, double dt_max, int pad_left, int pad_right,
			boolean lowResolution, int threadID) {

		long ptr;
		if (ptr_l_pool.size() <= threadID) {
			resizePtrPool(threadID + 1);
		}
		if (lowResolution)
			ptr = ptr_l_pool.get(threadID).longValue();
		else
			ptr = ptr_h_pool.get(threadID).longValue();

		double[] result = doIntegrateMosse(ptr, vars, lambda, mu,
				r, a,                       // rate vector + scalar amplitude
				drift, diffusion,
				q,                          // single 4x4 instead of per-bin P stack
				eVal, eVec, iEvec, useEigen, // eigendecomposition
				eQCache, eigenGeneration,   // per-likelihood-call eQ cache (a==0) + gen
				nt, dt_max, pad_left, pad_right);

		return result; // return non logged results
	}

	@Override
	public List<String> getArguments() {
		return null;
	}

	@Override
	public List<String> getConditions() {
		return null;
	}

	@Override
	public void sample(State state, Random random) {
		throw new UnsupportedOperationException();
	}

	private volatile boolean closed = false;

	@Override
	public synchronized void close() throws Exception {
	    if (closed) return;
	    closed = true;
	    if (cleanable != null) {
	        cleanable.clean(); // runs CleanerState.run() exactly once
	        cleanable = null;
	    }
	}

	@Override
	public void store() {
		// System.out.println("MosseDistribution.store()");
		super.store();
		storedrift = drift;
		storedDiffusion = diffusion;
		storedA = a;
		storedPadLeft_h = padLeft_h;
		storedPadLeft_l = padLeft_l;
		storedPadRight_h = padRight_h;
		storedPadRight_l = padRight_l;
		storedNumEntries_h = numEntries_h;
		storedNumEntries_l = numEntries_l;
	}

	@Override
	public void restore() {
		// System.out.println("MosseDistribution.restore()");
		super.restore();
		drift = storedrift;
		diffusion = storedDiffusion;
		a = storedA;
		padLeft_h = storedPadLeft_h;
		padLeft_l = storedPadLeft_l;
		padRight_h = storedPadRight_h;
		padRight_l = storedPadRight_l;
		numEntries_h = storedNumEntries_h;
		numEntries_l = storedNumEntries_l;
	}
	
	public void printParams() {
		System.out.println("drift = " + drift + "; diffusion = " + diffusion + "; nx = " + nx + "; nx = " + nx + "; dt = " + dt + "; width = " + width + "; resolution = " + resolution);
		System.out.println("padLeft_h = " + padLeft_h + "; padLeft_l = " + padLeft_l + "; padRight_h = " + padRight_h + "; padRight_l = " + padRight_l + "; numEntries_h = " + numEntries_h + "; numEntries_l = " + numEntries_l);
	}
	
	@Override
	protected boolean requiresRecalculation() {
		drift     = driftInput.get().getValue();
		diffusion = diffusionInput.get().getValue();
		// FIX: previously `a` was set only in initAndValidate, leaving treeModel.a
		// stuck at its initial XML value (typically 0). That silently disabled the
		// punctuational machinery — the puncA parameter still moved in the MCMC
		// (operator + prior) but had no effect on the likelihood. Sync it now so
		// puncA actually feeds into P = I + a*Q.
		a         = aInput.get().getValue();
	    return true;
	}
}
