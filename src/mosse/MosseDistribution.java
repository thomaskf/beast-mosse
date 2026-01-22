package mosse;

import java.util.List;
import java.util.Random;

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
	final public Input<RealParameter> dxInput = new Input<>("dx", "distance between xs", new RealParameter("0.0001"));
	final public Input<RealParameter> driftInput = new Input<>("drift", "drift parameter", new RealParameter("0.0"));
	final public Input<RealParameter> diffusionInput = new Input<>("diffusion", "diffusion parameter",
			new RealParameter("0.001"));
	final public Input<RealParameter> dtInput = new Input<>("dt", "time interval dt", new RealParameter("0.01"));
	final public Input<IntegerParameter> widthInput = new Input<>("width", "width of the kernel for convolution",
			new IntegerParameter("10"));
	final public Input<IntegerParameter> resolutionInput = new Input<>("resolution",
			"scale factor for resolution of bins", new IntegerParameter("4"));
	
	// number of cpu threads (not for this class)
    final public Input<Integer> threadsInput =
            new Input<>("threads", "Number of threads for within-node parallelism", 1);
	
	final public int FLAG_FFTW3_DEFAULT = 0;

	protected int resolution;
	protected int nx;
	protected double dx;
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
	
	// number of threads
	protected int numThreads;
	protected long[] ptr_l; // ptr for low resolution
	protected long[] ptr_h; // ptr for high resolution
	
	// for storing during mcmc
	protected double storedrift;
	protected double storedDiffusion;
//	protected int storedPadLeft_h;
//	protected int storedPadLeft_l;
//	protected int storedPadRight_h;
//	protected int storedPadRight_l;
//	protected int storedNumEntries_h;
//	protected int storedNumEntries_l;

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

	private native double[] doIntegrateMosse(long obj_ptr, double[] vars, double[] lambda, double[] mu, double drift,
			double diffusion, double[] Q, int nt, double dt, int pad_left, int pad_right);

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
		if (dxInput.get() != null) {
			dx = dxInput.get().getValue();
		} else {
			dx = dxInput.defaultValue.getValue();
		}
		
		numRateBins_h = nx * resolution;
		numRateBins_l = nx;

		drift = driftInput.get().getValue();
		diffusion = diffusionInput.get().getValue();
		dt = dtInput.get().getValue();

		width = widthInput.get().getValue();
		
		// compute padLeft, padRight, and numEntries
		computePadNumEntries();
		
		if (threadsInput.get() != null) {
			numThreads = threadsInput.get().intValue();
		}

		int[] nd = { 5 };
		int flags = FLAG_FFTW3_DEFAULT;
		ptr_l = new long[numThreads];
		ptr_h = new long[numThreads];
		for (int i = 0; i < numThreads; i++) {
			ptr_l[i] = makeMosseFFT(nx, dx * resolution, nd, flags);
			ptr_h[i] = makeMosseFFT(nx * resolution, dx, nd, flags);
		}
	}

	public void computePadNumEntries() {
		double mean = drift * dt;
		double sd = Math.sqrt(diffusion * dt);
		
		// low resolution
		padLeft_l = Math.abs((int) Math.ceil(-(mean - width * sd) / dx / resolution));
		padRight_l = Math.abs((int) Math.ceil((mean + width * sd) / dx / resolution));
		numEntries_l = numRateBins_l - padLeft_l - padRight_l - 1;
		// high resolution
		padLeft_h = Math.abs((int) Math.ceil(-(mean - width * sd) / dx / resolution) * resolution);
		padRight_h = Math.abs((int) Math.ceil((mean + width * sd) / dx / resolution) * resolution);
		numEntries_h = numRateBins_h - padLeft_h - padRight_h - 1;
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

	public double[] calculateBranchLogP(double branchTime, double[] vars, double[] lambda, double[] mu, double[] Q,
			boolean lowResolution, int threadID) {
		// double logP = 0.0;
		// getting parameter values
		int nt = (int) Math.ceil(branchTime / dt);
		double[] result_native;
		
		if (lowResolution) {
			result_native = doIntegration(vars, lambda, mu, drift, diffusion, Q, nt, dt, padLeft_l, padRight_l, lowResolution, threadID);
		} else {
			result_native = doIntegration(vars, lambda, mu, drift, diffusion, Q, nt, dt, padLeft_h, padRight_h, lowResolution, threadID);
		}
		
		double[] result = new double[vars.length];
		System.arraycopy(result_native, 0, result, 0, vars.length);
		
		return result;
	}

	/**
	 * perform integration along a single branch
	 *
	 * @param vars      array of tips or partials
	 * @param lambda    array of birth-rates
	 * @param mu        array of death-rates
	 * @param drift     drift parameter
	 * @param diffusion diffusion parameter
	 * @param Q         exponentiated form of the Q matrix
	 * @param nt        number of time steps
	 * @param dt_max    maximum time step
	 * @param pad_left  padding size left of the kernel (zero padding)
	 * @param pad_right padding size right of the kernel (zero padding)
	 * @param lq        log-scaling
	 * @return partial probabilities
	 */
	public double[] doIntegration(double[] vars, double[] lambda, double[] mu, double drift, double diffusion, double[] Q, int nt,
			double dt_max, int pad_left, int pad_right, boolean lowResolution, int threadID) {

		// make mosse fft object pointer
		long ptr;
		if (lowResolution)
			ptr = ptr_l[threadID];
		else
			ptr = ptr_h[threadID];
		
		// long ptr = makeMosseFFT(nx, dx, nd, flags);

		// integrate using C propagate x and propagate t
		double[] result = doIntegrateMosse(ptr, vars, lambda, mu, drift, diffusion, Q, nt, dt_max, pad_left, pad_right);

		// mosseFinalize(ptr); // destroy obj pointer
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
	    for (int i = 0; i < numThreads; i++) {
	        if (ptr_l[i] != 0) mosseFinalize(ptr_l[i]);
	        if (ptr_h[i] != 0) mosseFinalize(ptr_h[i]);
	        ptr_l[i] = 0;
	        ptr_h[i] = 0;
	    }
	}
	
	@Override
    protected void finalize() throws Throwable {
        close();
    }

	@Override
	public void store() {
		// System.out.println("MosseDistribution.store()");
		super.store();
		storedrift = drift;
		storedDiffusion = diffusion;
//		storedPadLeft_h = padLeft_h;
//		storedPadLeft_l = padLeft_l;
//		storedPadRight_h = padRight_h;
//		storedPadRight_l = padRight_l;
//		storedNumEntries_h = numEntries_h;
//		storedNumEntries_l = numEntries_l;
	}

	@Override
	public void restore() {
		// System.out.println("MosseDistribution.restore()");
		super.restore();
		drift = storedrift;
		diffusion = storedDiffusion;
//		padLeft_h = storedPadLeft_h;
//		padLeft_l = storedPadLeft_l;
//		padRight_h = storedPadRight_h;
//		padRight_l = storedPadRight_l;
//		numEntries_h = storedNumEntries_h;
//		numEntries_l = storedNumEntries_l;
	}
	
	public void printParams() {
		System.out.println("drift = " + drift + "; diffusion = " + diffusion);
	}
	
	@Override
	protected boolean requiresRecalculation() {
		drift = driftInput.get().getValue();
		diffusion = diffusionInput.get().getValue();
	    return true;
	}
}
