package mosse;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinWorkerThread;
import java.util.concurrent.RecursiveAction;
import java.util.stream.IntStream;

import org.jblas.DoubleMatrix;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Log;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.likelihood.TreeLikelihood;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.substitutionmodel.EigenDecomposition;
import beast.base.evolution.substitutionmodel.Frequencies;
import beast.base.evolution.substitutionmodel.HKY;
import beast.base.evolution.substitutionmodel.SubstitutionModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TraitSet;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;

/**
 * @author Kylie Chen
 * @author Thomas Wong
 */

@Description("Mosse likelihood class calculates the probability of sequence and trait data on a tree")
public class MosseTreeLikelihood extends TreeLikelihood {

	protected MosseLikelihoodCore mosseLikelihoodCore;
	// trait data
	final public Input<List<TraitSet>> traitListInput = new Input<>("traits", "list of traits", new ArrayList<>());
	// tip model, species diversification model and trait model
	final public Input<MosseTipLikelihood> tipModelInput = new Input<>("tipModel", "model of tip probabilities",
			Input.Validate.REQUIRED);
	final public Input<MosseDistribution> treeModelInput = new Input<>("treeModel", "species diversification model",
			Input.Validate.REQUIRED);

	// lambda and mu functions
	final public Input<LinkFn> lambdaFuncInput = new Input<>("lambdaFunc", "function for birth rate lambda",
			Input.Validate.REQUIRED);
	final public Input<LinkFn> muFuncInput = new Input<>("muFunc", "function for death rate mu",
			Input.Validate.REQUIRED);

	// root options
	final public Input<LinkFn> rootFuncInput = new Input<>("rootFunc", "function for root", Input.Validate.OPTIONAL);

	final public Input<IntegerParameter> rootOptionInput = new Input<>("rootOption", "option for root calculation",
			Input.Validate.OPTIONAL);

	// tc (i.e. all nodes below this height are using high resolution)
	final public Input<Double> tcInput = new Input<>("tc", "below this height for high resolution",
			Input.Validate.OPTIONAL);

	// resolution mode: "mixed" (default, tc-based), "low" (all low), "high" (all high)
	final public Input<String> resolutionModeInput = new Input<>("resolutionMode",
			"Resolution mode for the whole tree: \"mixed\" (default, tc-based split between high and low), " +
			"\"low\" (entire tree uses low resolution), or \"high\" (entire tree uses high resolution).",
			"mixed");

	// root options
	final public int ROOT_FLAT = 1;
	final public int ROOT_OBS = 2;
	final public int ROOT_EQUI = 3; // check test case for NAN
	final public int ROOT_GIVEN = 4; // check test case for NAN

	protected Alignment data;
	protected TreeInterface tree;
	protected int taxonCount;
	protected int nodeCount;
	protected int resolution;
	protected int patterns;
	protected int stateCount;
	protected double deltaT;
	protected int numPlan;

	protected int rootOption = 2; // default to ROOT_OBS
	protected double[] sharedRootP = null;
	protected boolean captureRootP = false;
	protected double lastFlatLogP = 0.0;
	protected double survDenom = 1.0;

	protected LinkFn rootFunc;
	protected LinkFn lambdaFunc;
	protected LinkFn muFunc;
	protected List<TraitSet> traits;
	protected MosseTipLikelihood tipModel;
	protected MosseDistribution treeModel;
	protected double tc; // time < tc for high resolution, while time >= tc for low resolution

	// Resolution mode constants and active mode
	public static final String RESOLUTION_MODE_MIXED = "mixed";
	public static final String RESOLUTION_MODE_LOW   = "low";
	public static final String RESOLUTION_MODE_HIGH  = "high";
	protected String resolutionMode; // one of the three constants above
	
	// rate for the bins
	protected double rmin; // minimum of rate
	protected double rmax; // maximum of rate

	// variables for high resolution
	protected int numRateBins_h;
	protected double dx_h;
	protected double startSubsRate_h;
	protected double[] lambdas_h;
	protected double[] mus_h;
	protected double[] flatTransitionMatrices_h; // legacy: now unused by punc path (kept to minimise diffs elsewhere)
	protected double[] rates_h;                  // Punctuational support: per-bin substitution rate (high resolution)

	// variables for low resolution
	protected int numRateBins_l;
	protected double dx_l;
	protected double startSubsRate_l;
	protected double[] lambdas_l;
	protected double[] mus_l;
	protected double[] flatTransitionMatrices_l; // legacy: now unused by punc path
	protected double[] rates_l;                  // Punctuational support: per-bin substitution rate (low resolution)
	protected double[] qFlat;                    // Punctuational support: single 4x4 substitution-rate matrix flattened

	// eigendecomposition of Q
	protected double[] eVal; // eigenvalues
	protected double[] eVec; // eigenvectors
	protected double[] iEvec; // inverses of eigenvectors
	protected boolean hasEigen;

	// per-bin eQ = exp(Q * r[ix] * dt) cache, built once per likelihood
	// call (alongside qFlat / eVal / eVec / iEvec) and passed to every JNI
	// branch call. Valid only when hasEigen && treeModel.a == 0 — otherwise
	// null and the C kernel falls back to per-step build (a > 0) or GSL.
	protected double[] eQCache_h;
	protected double[] eQCache_l;

	protected int numRateBins_max; // max{numRateBins_h,numRateBins_l}

	// for each node, store indices of taxa under subtree rooted at the node
	protected int[] taxaIndexUnderNode;
	// for each node, map global pattern index -> local sub-pattern index
	protected int[] pattern2SubpatnPerNode;
	// for each node, store numRateBins
	// protected int[] numRateBinsPerNode;
	
	// counter
	protected int count;

	// pool for threads
    protected ForkJoinPool pool;

	/**
	 * Run {@code job} inside {@link #pool}, joining the result. When the
	 * current thread is already a worker of {@code pool} we execute inline
	 * — submitting and blocking on {@code Future.get()} from a pool worker
	 * would consume one of the pool's threads and can deadlock once the pool
	 * is fully occupied by tree-level tasks. {@link IntStream#parallel()}
	 * called inside the job picks up the surrounding pool automatically,
	 * so nested parallelism still works.
	 */
	protected void runInPool(Runnable job) {
		if (pool == null) { job.run(); return; }
		Thread t = Thread.currentThread();
		if (t instanceof ForkJoinWorkerThread
				&& ((ForkJoinWorkerThread) t).getPool() == pool) {
			job.run();
			return;
		}
		try {
			pool.submit(job).get();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new RuntimeException(e);
		} catch (ExecutionException e) {
			throw new RuntimeException(e.getCause());
		}
	}

	/**
	 * Post-order tree task: fork the right subtree, recurse left in the
	 * current thread, join, then compute the node's partials. Independent
	 * sibling subtrees execute in parallel, which lifts the threads-idle
	 * problem at near-leaf nodes where the per-node subpattern count is small.
	 */
	private final class TraverseTask extends RecursiveAction {
		private static final long serialVersionUID = 1L;
		private final Node node;
		TraverseTask(Node node) { this.node = node; }
		@Override
		protected void compute() {
			if (node.isLeaf()) return;
			TraverseTask right = new TraverseTask(node.getRight());
			right.fork();
			new TraverseTask(node.getLeft()).compute();
			right.join();
			computePartialLikelihood(node);
		}
	}

	// Set to true when substitution model parameters or dx_h change; reset after
	// recomputing flatTransitionMatrices_{h,l} in traverseFull().
	protected boolean transitionMatricesDirty = true;

	// Monotonically-increasing counter that bumps every time the eigendata
	// (eVal/eVec/iEvec) or eQCache_{h,l} are rebuilt (i.e. whenever
	// transitionMatricesDirty triggers in traverseFull). Passed to the JNI
	// integrator so per-thread native plans can skip the eigendata copy when
	// the cached generation already matches — these arrays are constant for
	// every branch call within one likelihood evaluation, and the eQCache copy
	// alone is ~520 KB at high resolution.
	protected long eigenGeneration = 0L;

	// Computed once per computeLambdaMus() call; valid until pad params change.
	protected double[] cached_x_h;
	protected double[] cached_x_l;

	// Three arrays per thread: [0]=partialsLeft, [1]=partialsRight, [2]=patnPartials,
	// each of length numPlan * numRateBins_h (maximum possible partial size).
	protected ThreadLocal<double[][]> threadLocalScratch;
    
	// the format of the displayed log-likelihood value
	DecimalFormat df;
	
	// lower bound
	double beta_upbound = 3.0;
	double beta_lowbound = -3.0;
	double epsilon_lowbound = 1e-4;   // relaxed: allow exploration below true value
	double diffusion_lowbound = 1e-5; // relaxed: allow exploration below true value

	// trait range stored for grid-boundary checks (set in computeRminRmaxDx)
	protected double traitmin_global = Double.MAX_VALUE;
	protected double traitmax_global = -Double.MAX_VALUE;

	// when logScale is true, the bin axis is uniform in log r
	protected boolean logScale;

	@Override
	public void initAndValidate() {

		data = dataInput.get();
		tree = treeInput.get();
		taxonCount = data.getTaxonCount();
		nodeCount = tree.getNodeCount();
		patterns = data.getPatternCount();
		stateCount = data.getDataType().getStateCount();
		numPlan = stateCount + 1; // dimensions

		traits = traitListInput.get();
		tipModel = tipModelInput.get();
		treeModel = treeModelInput.get();
		resolution = treeModel.resolution;
		deltaT = treeModel.dtInput.get().getValue(); // 0.001; // dt
		logScale = tipModel.logScale;

		// compute the min and the max value of rate for the bins
		double x0 = tipModel.meanSubstitutionInput.get().getValue();
		computeRminRmaxDx(x0, treeModel.nx, resolution); // assign to the variables rmin, rmax, and dx_h
		
		// high resolution
		numRateBins_h = treeModel.numRateBins_h;
		startSubsRate_h = rmin;
		
		// low resolution
		numRateBins_l = treeModel.numRateBins_l;
		dx_l = dx_h * resolution;
		startSubsRate_l = rmin;

		// default initial epsilon to 2*dx (low-res) when not set (value <= 0)
		if (tipModel.epsilon.getValue() <= 0)
			tipModel.epsilon.setValue(0, 2 * dx_l);

		// raise initial diffusion above the sd/dx=0.2 floor when it starts below (analogous to epsilon)
		double diffFloorInit = 0.04 * dx_l * dx_l / deltaT; // sd/dx=0.2 => diffusion=(0.2*dx)^2/dt
		if (treeModel.diffusionInput.get().getValue() < diffFloorInit)
			treeModel.diffusionInput.get().setValue(0, 2 * diffFloorInit);
				
		// maximum value of numRateBins (always numRateBins_h)
		numRateBins_max = numRateBins_h;

		lambdaFunc = lambdaFuncInput.get();
		muFunc = muFuncInput.get();

		if (rootOptionInput.get() != null) {
			rootOption = rootOptionInput.get().getValue();
		}
		if (rootFuncInput.get() != null) {
			rootFunc = rootFuncInput.get();
		}

		// input checking
		if (taxonCount != tree.getLeafNodeCount()) {
			String message = String.format(
					"The number of leaves in tree (%d) does not match the number of sequences (%d).",
					tree.getLeafNodeCount(), taxonCount);
			throw new IllegalArgumentException(message);
		} else if (!(siteModelInput.get() instanceof SiteModel.Base)) {
			throw new IllegalArgumentException("siteModel input should be of type SiteModel.Base");
		} else if (branchRateModelInput.get() != null) {
			System.err.println("Ignoring clock model " + branchRateModelInput.get().getID());
		}

		beagle = null;

		if (tcInput.get() != null) {
			tc = tcInput.get().doubleValue();
		} else {
			tc = tree.getRoot().getHeight() / 10.0;
		}

		// read and validate the resolutionMode input
		resolutionMode = resolutionModeInput.get();
		if (!RESOLUTION_MODE_MIXED.equals(resolutionMode)
				&& !RESOLUTION_MODE_LOW.equals(resolutionMode)
				&& !RESOLUTION_MODE_HIGH.equals(resolutionMode)) {
			throw new IllegalArgumentException(
					"resolutionMode must be one of \"mixed\", \"low\", or \"high\", but got: \"" + resolutionMode + "\".");
		}
		m_siteModel = (SiteModel.Base) siteModelInput.get();
		m_siteModel.setDataType(dataInput.get().getDataType());
		substitutionModel = m_siteModel.substModelInput.get();

		// remove requirement for clock model
		branchRateModelInput.setRule(Input.Validate.OPTIONAL);
		branchRateModel = null;
		m_branchLengths = new double[nodeCount];
		storedBranchLengths = new double[nodeCount];

		// set likelihood core number of states and number of rate bins
		mosseLikelihoodCore = new MosseLikelihoodCore(stateCount, numRateBins_max);
		
		// logging likelihood class
		String className = getClass().getSimpleName();
		Log.info.println(className + "(" + getID() + ") uses " + mosseLikelihoodCore.getClass().getSimpleName());
		Log.info.println("  " + data.toString(true));

		setProportionInvariant(m_siteModel.getProportionInvariant());
		m_siteModel.setPropInvariantIsCategory(false);
		if (getProportionInvariant() > 0) {
			calcConstantPatternIndices(patterns, stateCount);
		}

		// set up likelihood core and initialize partials
		this.initCore();

		patternLogLikelihoods = new double[patterns];
		matrixSize = (stateCount + 1) * (stateCount + 1);
		probabilities = new double[(stateCount + 1) * (stateCount + 1)];
		Arrays.fill(probabilities, 1.0);

		if (dataInput.get().isAscertained) {
			useAscertainedSitePatterns = true;
		}

		// for each node, store indices of taxa under subtree rooted at the node
		taxaIndexUnderNode = new int[taxonCount * nodeCount];
		// for each node, map global pattern index -> local sub-pattern index
		pattern2SubpatnPerNode = new int[nodeCount * patterns];
		// for each node, store numRateBins
		// numRateBinsPerNode = new int[nodeCount];
		
		// initialize the fft pointers
		treeModel.initFFTPtrs(dx_h);
		
		// for this class, always single-threaded
		pool = null;
		
		// the format of the displayed log-likelihood value
		df = new DecimalFormat();
		df.setMaximumFractionDigits(7);
		df.setRoundingMode(RoundingMode.CEILING);
		
		count = 0;

		// possible partial array (numPlan * numRateBins_h). Three arrays per thread.
		final int scratchSize = numPlan * numRateBins_h;
		threadLocalScratch = ThreadLocal.withInitial(() -> new double[][] {
			new double[scratchSize], // [0] partialsLeft
			new double[scratchSize], // [1] partialsRight
			new double[scratchSize]  // [2] patnPartials (combined)
		});

		// first traversal.
		transitionMatricesDirty = true;
	}

	// compute the values of rmin, rmax, and dx
	protected void computeRminRmaxDx(double x0, int nx, int resolution) {
		List<Node> leaves = tree.getExternalNodes();
		List<Double> traitList = new ArrayList<>();
		for (int i = 0; i < leaves.size(); i++) {
			double[] traitValues = getTraits(leaves.get(i));
			for (double value : traitValues) {
				traitList.add(value);
			}
		}
		double traitmin = Collections.min(traitList).doubleValue();
		double traitmax = Collections.max(traitList).doubleValue();
		traitmin_global = traitmin;
		traitmax_global = traitmax;
		double w = 5.0;
		double betamin = -3.0; // lower bound of beta
		double betamax = 3.0; // upper bound of beta
		double rminmin = x0 - w/2 * betamin * traitmax + (w/2+1) * betamin * traitmin;
		double rminmax = x0 - w/2 * betamax * traitmax + (w/2+1) * betamax * traitmin;
		double rmaxmin = x0 + (w/2+1) * betamin * traitmax - w/2 * betamin * traitmin;
		double rmaxmax = x0 + (w/2+1) * betamax * traitmax - w/2 * betamax * traitmin;
		List<Double> x_array = new ArrayList<>();
		x_array.add(rminmin); x_array.add(rminmax); x_array.add(rmaxmin); x_array.add(rmaxmax);
		rmin = Collections.min(x_array).doubleValue();
		rmax = Collections.max(x_array).doubleValue();
		dx_h = (rmax-rmin) / (nx * resolution);
		
		int num_dec_pl = 5;
		dx_h = BigDecimal.valueOf(dx_h).setScale(num_dec_pl, RoundingMode.HALF_UP).doubleValue(); // round to 5 decimal places

		if (!logScale) {		
			if (rmin < 0) {
				int nBins = ((int) (Math.abs(rmin) / dx_h / resolution) ) * resolution ; // has to be divisible by 4
				rmin = - dx_h * nBins;
			} else {
				rmin = 0.0;
			}
		}
		
		rmax = rmin + dx_h * nx * resolution;
	}
	
	protected void computeLambdaMus() {
		
	    // Always sync drift/diffusion from BEAST2 parameter objects before computing pads.
	    treeModel.drift     = treeModel.driftInput.get().getValue();
	    treeModel.diffusion = treeModel.diffusionInput.get().getValue();

	    // update the values of pads and numEntries
		treeModel.computePadNumEntries(dx_h);

		// compute the padLeft, padRight, and numEntries for low and high resolution
		int padLeft_l = treeModel.padLeft_l;
		int numEntries_l = treeModel.numEntries_l;
		int padLeft_h = treeModel.padLeft_h;
		int numEntries_h = treeModel.numEntries_h;

		// get lambdas and mus
		lambdas_h = new double[numEntries_h];
		lambdas_l = new double[numEntries_l];
		mus_h = new double[numEntries_h];
		mus_l = new double[numEntries_l];

		// In linear-rate mode, cached_x_h (or cached_x_l) hold rate values directly and are
		// passed straight to the LinkFns. 
		// In log-scale mode, cached_x_h (or cached_x_l) hold log-rate values.
		cached_x_h = getSubstitutionRates(numEntries_h, startSubsRate_h, dx_h, padLeft_h);
		cached_x_l = getSubstitutionRates(numEntries_l, startSubsRate_l, dx_l, padLeft_l);
		double[] linkFnX_h = logScale ? expArray(cached_x_h) : cached_x_h;
		double[] linkFnX_l = logScale ? expArray(cached_x_l) : cached_x_l;
		lambdaFunc.getY(linkFnX_h, lambdas_h);
		lambdaFunc.getY(linkFnX_l, lambdas_l);
		muFunc.getY(linkFnX_h, mus_h);
		muFunc.getY(linkFnX_l, mus_l);

		transitionMatricesDirty = true;
	}
	
	/**
	 * set the taxon indices under all children of each node
	 */
	protected void setTaxonIndices(Node node) {
		if (node.isRoot()) {
			// reset the arrays
			Arrays.fill(taxaIndexUnderNode, -1);
		}
		if (node.isLeaf()) {
			taxaIndexUnderNode[node.getNr() * taxonCount] = data.getTaxonIndex(node.getID());
		} else {
			int k = node.getNr() * taxonCount;
			setTaxonIndices(node.getLeft());
			setTaxonIndices(node.getRight());
			// for left
			int k_left = node.getLeft().getNr() * taxonCount;
			while (taxaIndexUnderNode[k_left] != -1) {
				taxaIndexUnderNode[k] = taxaIndexUnderNode[k_left];
				k_left++;
				k++;
			}
			// for right
			int k_right = node.getRight().getNr() * taxonCount;
			while (taxaIndexUnderNode[k_right] != -1) {
				taxaIndexUnderNode[k] = taxaIndexUnderNode[k_right];
				k_right++;
				k++;
			}
		}
	}

	protected void setLeafPartials(Node node, int patterncount, int threadID) {
		// for leaf only

		int taxonIndex = data.getTaxonIndex(node.getID());
		// obtain the mapping: the pattern -> sub-pattern id
		// to map the state -> sub-pattern id
		HashMap<Integer, Integer> state2subpatnid = new HashMap<>();
		// collect the number of sub-patterns and the mapping between global index to
		// local partial index
		int subpatns = 0;
		int s = node.getNr() * patterncount;
		for (int patternIndex = 0; patternIndex < patterncount; patternIndex++) {
			int stateid = data.getPattern(taxonIndex, patternIndex);
			int subpatnid;
			if (!state2subpatnid.containsKey(stateid)) {
				state2subpatnid.put(stateid, subpatns);
				subpatnid = subpatns;
				subpatns++;
			} else {
				subpatnid = state2subpatnid.get(stateid);
			}
			// mapping: pattern -> sub-pattern id
			pattern2SubpatnPerNode[s + patternIndex] = subpatnid;
		}
		assert (subpatns > 0);

		// In "low" resolution mode the leaf itself is low-resolution; in "high" or
		// "mixed" modes the leaf always starts at high resolution (and is downsampled
		// later if its parent operates at low resolution).
		boolean leafIsLow = RESOLUTION_MODE_LOW.equals(resolutionMode);
		int numRateBins_leaf    = leafIsLow ? numRateBins_l : numRateBins_h;
		int numEntries_leaf     = leafIsLow ? treeModel.numEntries_l : treeModel.numEntries_h;
		int padLeft_leaf        = leafIsLow ? treeModel.padLeft_l    : treeModel.padLeft_h;
		double startSubsRate_leaf = leafIsLow ? startSubsRate_l : startSubsRate_h;
		double dx_leaf          = leafIsLow ? dx_l : dx_h;

		int singlePartialSizeLeaf = (stateCount + 1) * numRateBins_leaf;

		// the size of the partials stored for this leaf is based on its parent's resolution
		int numRateBins_parent = numRateBins_h;
		if (isLowResolution(node.getParent())) {
			// assume all leaves should have a parent
			numRateBins_parent = numRateBins_l;
		}
		int singlePartialSizeParent = (stateCount + 1) * numRateBins_parent;
		double[] partials = new double[subpatns * singlePartialSizeParent];
		double[] compensates = new double[subpatns];

		double[] traitValues = getTraits(node);
		// Java initialises boolean arrays to false, so no explicit fill needed.
		boolean[] updated = new boolean[subpatns];

		double[] tipLikelihoods = tipModel.getTipLikelihoods(traitValues, numEntries_leaf,
				startSubsRate_leaf + padLeft_leaf * dx_leaf, dx_leaf);

		// compute the partial likelihood for each sub-pattern.
		for (int patternIndex = 0; patternIndex < patterncount; patternIndex++) {
			int subpatnid = pattern2SubpatnPerNode[s + patternIndex];
			if (!updated[subpatnid]) {
				updated[subpatnid] = true;
				int stateid = data.getPattern(taxonIndex, patternIndex);
				boolean[] stateSet = data.getStateSet(stateid);

				// Allocate a fresh array for this subpattern so normalization()'s
				// in-place mutation of patnPartials does not bleed into the next
				// subpattern's initial condition.
				double[] patnPartials = new double[singlePartialSizeLeaf];
				// E initial values are zero (Java array initialisation).

				int k = numRateBins_leaf;
				// D initial values
				for (int state = 0; state < stateCount; state++) {
					if (stateSet[state]) {
						// set likelihoods for nucleotide in data
						if (numRateBins_leaf <= numEntries_leaf) {
							for (int i = 0; i < numRateBins_leaf; i++) {
								patnPartials[k++] = tipLikelihoods[i];
							}
						} else {
							// padLeft leading zeros before the tip likelihoods
							for (int i = 0; i < padLeft_leaf; i++) {
								patnPartials[k++] = 0.0;
							}
							for (int i = 0; i < numEntries_leaf; i++) {
								patnPartials[k++] = tipLikelihoods[i];
							}
							// set to zeros for the rest (padRight + 1)
							for (int i = padLeft_leaf + numEntries_leaf; i < numRateBins_leaf; i++) {
								patnPartials[k++] = 0.0;
							}
						}
					} else {
						// otherwise leave likelihoods to zero
						k += numRateBins_leaf;
					}
				}
				// propagate along the parent branch
				double[] logp_patn = new double[1];
				double[] patnPartialsResult = computeSingleBranchLikelihood(node.getParent(), node, patnPartials, logp_patn, 0, threadID);

				assert(patnPartialsResult.length == singlePartialSizeParent);
				int st = subpatnid * singlePartialSizeParent;
				System.arraycopy(patnPartialsResult, 0, partials, st, singlePartialSizeParent);
				compensates[subpatnid] = logp_patn[0];
			}
		}

		// store the partials and compensates
		mosseLikelihoodCore.setNodeMossePartials(node.getNr(), partials);
		mosseLikelihoodCore.setNodeMosseCompensates(node.getNr(), compensates);
	}
	
	/**
	 * set leaf partials using tip GLM likelihood model *
	 */
	@Override
	protected void setPartials(Node node, int patterncount) {
		// assume the array taxaIndexUpdateNode has been updated
		if (node.isRoot()) {
			List<Node> leaves = tree.getExternalNodes();
			
			if (pool == null) {
				// single thread
				for (int sid = 0; sid < leaves.size(); sid++) {
	            	int threadID = 0;
					setLeafPartials(leaves.get(sid), patterncount, threadID);
				}
			} else {
				// multiple threads
		        runInPool(() -> IntStream.range(0, leaves.size()).parallel().forEach(sid -> {
	            	int threadID = threadIndexInPool();
					setLeafPartials(leaves.get(sid), patterncount, threadID);
		        }));
			}
		}
	}

	/**
	 * set leaf states (not applicable for this class, use setPartials instead)
	 */
	@Override
	protected void setStates(Node node, int patternCount) {
		throw new UnsupportedOperationException();
	}

	@Override
	protected void initCore() {
		mosseLikelihoodCore.initialize(nodeCount, patterns, true, m_useAmbiguities.get());

		/*
		final int extNodeCount = nodeCount / 2 + 1;
		final int intNodeCount = nodeCount / 2;

		// Create partials
		for (int i = 0; i < intNodeCount; i++) {
			mosseLikelihoodCore.createNodePartials(extNodeCount + i);
		}*/
	}

	/**
	 * calculate log P without caching (for testing)
	 *
	 * @return log P
	 */
	public double calculateLogPFull() {
		traverseFull(tree.getRoot());
		return logP;
	}

	/**
	 * Compute the normalization (or log compensation)
	 *
	 * @param vars -- array to be normalized
	 * @return log of scaling (i.e. lq)
	 */
	public double normalization(boolean lowResolution, double[] vars) {
		// eliminating the if (lowResolution) branch in this very-hot method.
		if (lowResolution) {
			return normalizationL(vars);
		} else {
			return normalizationH(vars);
		}
	}

	/** High-resolution normalization: sums/divides only over valid numEntries_h bins per
	 *  D column, skipping the padding bins that the C native code may leave non-zero. */
	public double normalizationH(double[] vars) {
		int nx = numRateBins_h;
		int numEntries = treeModel.numEntries_h;
		double vsum = 0.0;
		for (int col = 1; col < numPlan; col++) {
			int colStart = col * nx;
			for (int i = 0; i < numEntries; i++)
				vsum += vars[colStart + i];
		}
		if (vsum <= 0.0) return Double.NEGATIVE_INFINITY;
		double inv = 1.0 / vsum;
		for (int col = 1; col < numPlan; col++) {
			int colStart = col * nx;
			for (int i = 0; i < numEntries; i++)
				vars[colStart + i] *= inv;
		}
		return Math.log(vsum);
	}

	/** Low-resolution normalization: sums/divides only over valid numEntries_l bins per
	 *  D column, skipping the padding bins that the C native code may leave non-zero. */
	public double normalizationL(double[] vars) {
		int nx = numRateBins_l;
		int numEntries = treeModel.numEntries_l;
		double vsum = 0.0;
		for (int col = 1; col < numPlan; col++) {
			int colStart = col * nx;
			for (int i = 0; i < numEntries; i++)
				vsum += vars[colStart + i];
		}
		if (vsum <= 0.0) return Double.NEGATIVE_INFINITY;
		double inv = 1.0 / vsum;
		for (int col = 1; col < numPlan; col++) {
			int colStart = col * nx;
			for (int i = 0; i < numEntries; i++)
				vars[colStart + i] *= inv;
		}
		return Math.log(vsum);
	}

	/*
	public double normalization(double[] vars, int nx, double dx) {
		// normalize the values of vars
		// ignore the first nx entries (i.e. first row)
		double vsum = 0.0;
		for (int i = nx; i < vars.length; i++) {
			vsum += vars[i];
		}
		vsum *= dx;
		for (int i = nx; i < vars.length; i++) {
			vars[i] /= vsum;
		}
		return Math.log(vsum);
	}*/

	protected double[] buildRateVector(int numEntries, double dx, int padLeft, double rmin) {
		double[] rates = new double[numEntries];
		if (logScale) {
			// log-scale: the bin axis is uniform in log r, so the
			// per-bin rate fed to the C kernel is exp(log_r). Always > 0,
			for (int i = 0; i < numEntries; i++) {
				double logr = rmin + (padLeft + i) * dx;
				rates[i] = Math.exp(logr);
			}
		} else {
			for (int i = 0; i < numEntries; i++) {
				double r = rmin + (padLeft + i) * dx;
				rates[i] = (r > 0.0) ? r : 0.0;
			}
		}
		return rates;
	}

	/**
	 * extract the underlying substitution-rate matrix Q from the
	 * BEAST2 SubstitutionModel and return it as a length-(stateCount^2)
	 * flat vector in row-major order.
	 *
	 * This uses a numerical approximation:
	 *
	 *   Q ≈ (P(τ) − I) / τ      for a small τ
	 *
	 * where P(τ) is obtained from substitutionModel.getTransitionProbabilities
	 * with dt = τ and rate = 1. This works for any SubstitutionModel
	 * implementation without requiring it to expose its rate matrix directly.
	 * The error is O(τ) so τ is chosen small enough that downstream matrix
	 * exponentials are accurate to machine precision.
	 */
	protected double[] buildQFlat(Node node) {
		// tau=1e-8 loses precision for unequal frequencies; use a larger value.
		final double tau = 1e-4;
		int sq = stateCount * stateCount;
		double[] Pt = new double[sq];
		substitutionModel.getTransitionProbabilities(node, tau, 0, 1.0, Pt);
		double[] Qf = new double[sq];
		for (int i = 0; i < stateCount; i++) {
			for (int j = 0; j < stateCount; j++) {
				double identity = (i == j) ? 1.0 : 0.0;
				Qf[i * stateCount + j] = (Pt[i * stateCount + j] - identity) / tau;
			}
		}
		return Qf;
	}

	/**
	 * extract the substitution model's eigendecomposition
	 */
	protected void buildEigenDecomp(Node node, double[] qFlatRef) {
		hasEigen = false;
		eVal = null; eVec = null; iEvec = null;
		// set -Dmosse.disableEigen=true to disable using eigendecomposition method
		if (Boolean.getBoolean("mosse.disableEigen")) {
			return;
		}
		if (substitutionModel.canReturnComplexDiagonalization()) {
			// whether it contains non-real (i.e. complex) eigenvalues and eigenvectors
			return;
		}
		EigenDecomposition ed;
		try {
			ed = substitutionModel.getEigenDecomposition(node);
		} catch (Exception e) {
			return;
		}
		if (ed == null) return;
		double[] eValTry  = ed.getEigenValues();
		double[] eVecTry  = ed.getEigenVectors();
		double[] iEvecTry = ed.getInverseEigenVectors();
		if (eValTry == null || eVecTry == null || iEvecTry == null) return;
		if (eValTry.length != stateCount
				|| eVecTry.length  != stateCount * stateCount
				|| iEvecTry.length != stateCount * stateCount) {
			return;
		}
		// Reconstruct Q exactly from U · diag(λ) · U^-1 and overwrite qFlatRef.
		// This is mathematically exact and avoids 2nd-order error in (P-I)/tau.
		// Validate the layout via rate-matrix invariants (row sums ~0,
		// off-diagonals >= 0); fall back to GSL if invariants fail.
		double[] qRecon = new double[stateCount * stateCount];
		for (int i = 0; i < stateCount; i++) {
			for (int j = 0; j < stateCount; j++) {
				double v = 0.0;
				for (int k = 0; k < stateCount; k++) {
					v += eVecTry[i * stateCount + k] * eValTry[k] * iEvecTry[k * stateCount + j];
				}
				qRecon[i * stateCount + j] = v;
			}
		}
		for (int i = 0; i < stateCount; i++) {
			double rowSum = 0.0;
			for (int j = 0; j < stateCount; j++) {
				rowSum += qRecon[i * stateCount + j];
				if (i != j && qRecon[i * stateCount + j] < -1e-9) return;
			}
			if (Math.abs(rowSum) > 1e-6) return;
		}
		System.arraycopy(qRecon, 0, qFlatRef, 0, qRecon.length);
		eVal = eValTry; eVec = eVecTry; iEvec = iEvecTry;
		hasEigen = true;
	}

	/**
	 * build the per-bin eQ = exp(Q * r[ix] * dt) cache.
	 * Returns null when hasEigen is false (caller should use the
	 * per-step path). When non-null, layout is row-major:
	 *   cache[ix * 16 + i * 4 + j] = (U · diag(exp(eVal·r[ix]·dt)) · U⁻¹)[i, j]
	 * The same cache is reused for every branch within a likelihood call
	 * because tmp1 = r[ix] · dt has no branch-dependent quantities.
	 */
	protected double[] buildEQCache(double[] rates, double dt) {
		if (!hasEigen) return null;
		int n = rates.length;
		double[] cache = new double[n * 16];
		double[] eL = new double[stateCount];
		for (int ix = 0; ix < n; ix++) {
			double tmp1 = rates[ix] * dt;
			for (int k = 0; k < stateCount; k++) {
				eL[k] = Math.exp(eVal[k] * tmp1);
			}
			for (int i = 0; i < stateCount; i++) {
				for (int j = 0; j < stateCount; j++) {
					double sum = 0.0;
					for (int k = 0; k < stateCount; k++) {
						sum += eVec[i * stateCount + k] * eL[k] * iEvec[k * stateCount + j];
					}
					cache[ix * 16 + i * 4 + j] = sum;
				}
			}
		}
		return cache;
	}

	/**
	 * create a flatTransitionMatrice
	 */
	protected double[] createFlatTransitionMatrice(Node node, boolean lowResolution) {
		double dx;
		int numEntries;
		int padLeft;
		if (lowResolution) {
			dx = dx_l;
			numEntries = treeModel.numEntries_l;
			padLeft = treeModel.padLeft_l;
		} else {
			dx = dx_h;
			numEntries = treeModel.numEntries_h;
			padLeft = treeModel.padLeft_h;
		}
		int sqStateCount = stateCount * stateCount;

		double[] identityMatrix = new double[sqStateCount];
		for (int i = 0; i < stateCount; i++)
			identityMatrix[i * stateCount + i] = 1.0;

		double[] transitionMatrix = new double[sqStateCount];
		substitutionModel.getTransitionProbabilities(node, deltaT, 0, dx, transitionMatrix);
		
		DoubleMatrix matrixTran = new DoubleMatrix(stateCount, stateCount, transitionMatrix);

		// Use exponentiation-by-squaring to precompute a table of
		// matrixTran^(2^k) for k = 0..LOG2_MAX
		int totalSteps = padLeft + numEntries;
		int LOG2_MAX = 32 - Integer.numberOfLeadingZeros(totalSteps); // ceil(log2(totalSteps))
		DoubleMatrix[] powersOfTran = new DoubleMatrix[LOG2_MAX + 1];
		powersOfTran[0] = matrixTran;
		for (int k = 1; k <= LOG2_MAX; k++) {
			powersOfTran[k] = powersOfTran[k - 1].mmul(powersOfTran[k - 1]);
		}

		// Compose the current matrix incrementally by step count.
		// For each bin we need matrixTran^(step), where step increments by 1
		// each time currX > delta. Composing a single extra factor per step is
		// unavoidable; the savings from the power table come from the initial
		// fast-forward through the padLeft skip region.
		DoubleMatrix matrixCurr = new DoubleMatrix(stateCount, stateCount, identityMatrix);
		double currX = rmin;
		double delta = dx / 100.0;

		// Fast-forward through padLeft skipped bins using exponentiation-by-squaring.
		// Count how many steps actually need multiplying (where currX > delta).
		int stepsInPad = 0;
		double currX_scan = rmin;
		for (int i = 0; i < padLeft; i++) {
			if (currX_scan > delta) stepsInPad++;
			currX_scan += dx;
		}
		// Apply matrixTran^stepsInPad via repeated squaring.
		int remaining = stepsInPad;
		for (int k = LOG2_MAX; k >= 0 && remaining > 0; k--) {
			if (remaining >= (1 << k)) {
				matrixCurr = matrixCurr.mmul(powersOfTran[k]);
				remaining -= (1 << k);
			}
		}
		currX = currX_scan; // currX is now at the first entry bin

		// For the numEntries bins, each bin needs exactly one more factor if currX > delta.
		double[][] transitionMatrices = new double[numEntries][transitionMatrix.length];
		for (int i = 0; i < numEntries; i++) {
			if (currX > delta) {
				matrixCurr = matrixCurr.mmul(matrixTran);
			}
			transitionMatrices[i] = matrixCurr.toArray();
			currX += dx;
		}

		return Arrays.stream(transitionMatrices).flatMapToDouble(Arrays::stream).toArray();
	}
	

	/**
	 * compute likelihoods for single branch return the log compensation
	 */
	protected double[] computeSingleBranchLikelihood(Node node, Node child, double[] partialsIn, double[] logCompen, int categoryID, int threadID) {
		boolean lowResolution;
		double[] partialsOut;

		logCompen[0] = 0.0;
		if (!isLowResolution(node)) {
			// if node has high resolution, then high resolution for the whole branch
			lowResolution = false;
			double branchTime = (node.getHeight() - child.getHeight());
			logCompen[0] += normalizationH(partialsIn);
			partialsOut = treeModel.calculateBranchLogP(branchTime, partialsIn, lambdas_h, mus_h,
					rates_h, qFlat, eVal, eVec, iEvec, hasEigen, eQCache_h, eigenGeneration, lowResolution, threadID);
		} else if (isLowResolution(child)) {
			// if child has low resolution, then low resolution for the whole branch
			lowResolution = true;
			double branchTime = (node.getHeight() - child.getHeight());
			logCompen[0] += normalizationL(partialsIn);
			partialsOut = treeModel.calculateBranchLogP(branchTime, partialsIn, lambdas_l, mus_l,
					rates_l, qFlat, eVal, eVec, iEvec, hasEigen, eQCache_l, eigenGeneration, lowResolution, threadID);
		} else {
			// combination of high and low resolutions along the branch
			// high resolutions between child.getHight() and t_mid
			double branchTime;
			double t_mid = tc;
			if (node.getHeight() < tc) {
				t_mid = node.getHeight();
			}
			branchTime = (t_mid - child.getHeight());
			lowResolution = false;
			logCompen[0] += normalizationH(partialsIn);
			partialsOut = treeModel.calculateBranchLogP(branchTime, partialsIn, lambdas_h, mus_h,
					rates_h, qFlat, eVal, eVec, iEvec, hasEigen, eQCache_h, eigenGeneration, lowResolution, threadID);
			// reduce the size of partials to "numPlan * numRateBins_l"
			partialsOut = reduceSize(partialsOut);
			// then low resolution between t_mid and node.getHeight()
			lowResolution = true;
			branchTime = (node.getHeight() - t_mid);
			if (branchTime > 0.0) {
				logCompen[0] += normalizationL(partialsOut);
				partialsOut = treeModel.calculateBranchLogP(branchTime, partialsOut, lambdas_l, mus_l,
						rates_l, qFlat, eVal, eVec, iEvec, hasEigen, eQCache_l, eigenGeneration, lowResolution, threadID);
			}
		}
		// normalization (log compensation) on the output partials
		logCompen[0] += normalization(lowResolution, partialsOut);

		return partialsOut;
	}

	/**
	 * traverse the subtree rooted at node
	 *
	 * @param node
	 */
	private void traverseFull(Node node) {

		if (node.isLeaf()) {
			// nothing to do
			return;
		}

		if (node.isRoot()) {
			// root node
			// compute lambdas_h, lambdas_l, mus_h, and mus_l
			computeLambdaMus();
			// compute taxon indices under all children of each node
			setTaxonIndices(node);
			// Punctuational support: instead of precomputing per-bin transition probability stacks,
			// the punc integrator takes a per-bin substitution-rate vector r and
			// the single underlying rate matrix Q. Build them when dirty.
			if (transitionMatricesDirty) {
				rates_h = buildRateVector(numRateBins_h, dx_h, treeModel.padLeft_h, rmin);
				rates_l = buildRateVector(numRateBins_l, dx_l, treeModel.padLeft_l, rmin);
				qFlat   = buildQFlat(node);
				buildEigenDecomp(node, qFlat);
				// build per-bin eQ cache once per likelihood call. Only
				// valid for the a == 0 path; for a > 0 the C kernel must rebuild
				// eQ per step because tmp1 depends on dd[ix]. Caches are null
				// when hasEigen is false (GSL path) — the JNI bridge accepts null.
				if (hasEigen && treeModel.a == 0.0) {
					eQCache_h = buildEQCache(rates_h, deltaT);
					eQCache_l = buildEQCache(rates_l, deltaT);
				} else {
					eQCache_h = null;
					eQCache_l = null;
				}
				// Bump eigendata generation so JNI plans know to re-copy on next call.
				eigenGeneration++;
				transitionMatricesDirty = false;
			}
			// compute the partials for all leaves
			setPartials(node, patterns); // all site patterns

			if (pool != null) {
				// Tree-level parallelism: descend into both subtrees in parallel.
				// Per-node parallelism inside computePartialLikelihood still works
				// because IntStream.parallel() picks up the surrounding pool.
				final Node rootNode = node;
				pool.invoke(new RecursiveAction() {
					private static final long serialVersionUID = 1L;
					@Override
					protected void compute() {
						TraverseTask right = new TraverseTask(rootNode.getRight());
						right.fork();
						new TraverseTask(rootNode.getLeft()).compute();
						right.join();
					}
				});
				computePartialLikelihood(node);
				return;
			}
		}

		traverseFull(node.getLeft());
		traverseFull(node.getRight());
		computePartialLikelihood(node);
	}

	/**
	 * compute the mapping between the global pattern array and local subpattern-id
	 * return the number of subpatterns
	 */
	 protected int computeMapGlobal2SubpatternID(Node node) {

		int subpatns = patterns;
		int startPos = node.getNr() * patterns; // starting pos in patternMapPerNode
		if (!node.isRoot()) {
			// collect the number of sub-patterns if it is not a root node
			// and compute the mapping between global pattern index and local partial index.
			// Encode each subpattern as a compact String key instead of an
			// ArrayList<Integer>. String hashing is O(k) like ArrayList but avoids
			// boxing overhead and is much cheaper in practice due to interning and
			// compact memory layout. For typical stateCount<=64, each state fits in
			// two hex chars, giving short, cache-friendly keys.
			HashMap<String, Integer> subpatn2subpatnid = new HashMap<>();
			subpatns = 0;
			StringBuilder sbKey = new StringBuilder();
			int s = node.getNr() * taxonCount; // starting pos in taxIndexUnderNode
			for (int patternIndex = 0; patternIndex < patterns; patternIndex++) {
				// build a compact string key for this subpattern
				sbKey.setLength(0);
				int subpatnid;
				for (int i = 0; i < data.getTaxonCount(); i++) {
					if (taxaIndexUnderNode[s + i] == -1) {
						break;
					}
					int taxonIndex = taxaIndexUnderNode[s + i];
					int patternState = data.getPattern(taxonIndex, patternIndex);
					// separate state values with ',' to avoid ambiguity (e.g. "1","2" vs "12")
					sbKey.append(patternState).append(',');
				}
				String key = sbKey.toString();
				if (!subpatn2subpatnid.containsKey(key)) {
					subpatn2subpatnid.put(key, subpatns);
					subpatnid = subpatns;
					subpatns++;
				} else {
					subpatnid = subpatn2subpatnid.get(key);
				}
				// save the mapping: pattern -> sub-pattern id
				pattern2SubpatnPerNode[startPos + patternIndex] = subpatnid;
			}
		} else {
			// for root
			for (int patternIndex = 0; patternIndex < patterns; patternIndex++) {
				pattern2SubpatnPerNode[startPos + patternIndex] = patternIndex;
			}
		}
		return subpatns;
	 }
	 
	 /**
	  * compute partial likelihood for a pattern,
	  * update numRateBinsPerNode
	  * And return the log-compensation
	  * 
	  * @param node
	  * @param threadID -1 if single-threaded, otherwise >=0
	  */
	 protected double computePartialLikelihoodPattern(int patternIndex, Node node, double[] patternPartialsLeft, double[] patternPartialsRight,
			 double[] partialsAllPatterns, double[] patternCompensatesLeft, double[] patternCompensatesRight, int categoryID, int threadID) {
		 
		// numRateBins, numEntries, lambdas
		int numRateBins_curr = numRateBins_h;
		int numEntries_curr = treeModel.numEntries_h;
		double[] lambdas_curr = lambdas_h;
		if (isLowResolution(node)) {
			numRateBins_curr = numRateBins_l;
			numEntries_curr = treeModel.numEntries_l;
			lambdas_curr = lambdas_l;
		}
		int partialSizeCurr = numPlan * numRateBins_curr;
		
		// pos in pattern2SubpatnPerNode
		int left_t = node.getLeft().getNr() * patterns;
		int right_t = node.getRight().getNr() * patterns;
		
		int leftSubpatn = pattern2SubpatnPerNode[left_t + patternIndex]; 
		int rightSubpatn = pattern2SubpatnPerNode[right_t + patternIndex]; 

		double leftCompensate = patternCompensatesLeft[leftSubpatn];
		double rightCompensate = patternCompensatesRight[rightSubpatn];
		if (Double.isNaN(leftCompensate) || Double.isNaN(rightCompensate))
			return Double.NaN;
		
		int leftPos = leftSubpatn * partialSizeCurr;
		int rightPos = rightSubpatn * partialSizeCurr;
		
		// Reuse thread-local scratch buffers instead of allocating new
		// arrays on every call. The scratch arrays are sized to numPlan*numRateBins_h
		// (maximum partial size); we simply use the first partialSizeCurr elements.
		double[][] scratch = threadLocalScratch.get();
		double[] partialsLeft  = scratch[0];
		double[] partialsRight = scratch[1];
		double[] patnPartials  = scratch[2];
		System.arraycopy(patternPartialsLeft,  leftPos,  partialsLeft,  0, partialSizeCurr);
		System.arraycopy(patternPartialsRight, rightPos, partialsRight, 0, partialSizeCurr);

		int k = 0;
		// E is topology independent
		System.arraycopy(partialsLeft, k, patnPartials, k, numRateBins_curr);
		k += numRateBins_curr;

		// D_parent[s][r] = λ(r) × (P·D_left)[s][r] × (P·D_right)[s][r]
		// where P = I + a·Q (punctuational substitution at speciation)
		int nEntries = Math.min(numRateBins_curr, numEntries_curr);
		double puncA = treeModel.a;
		for (int j = 0; j < nEntries; j++) {
			double[] dL = new double[stateCount];
			double[] dR = new double[stateCount];
			for (int s = 0; s < stateCount; s++) {
				dL[s] = partialsLeft[(s + 1) * numRateBins_curr + j];
				dR[s] = partialsRight[(s + 1) * numRateBins_curr + j];
			}
			applyPuncMatrix(dL, puncA);
			applyPuncMatrix(dR, puncA);
			double lambdaX = lambdas_curr[j];
			for (int i = 0; i < stateCount; i++)
				patnPartials[(i + 1) * numRateBins_curr + j] = dL[i] * dR[i] * lambdaX;
		}
		for (int j = nEntries; j < numRateBins_curr; j++)
			for (int i = 0; i < stateCount; i++)
				patnPartials[(i + 1) * numRateBins_curr + j] = 0.0;
		k = numPlan * numRateBins_curr;
		
		double[] logp_patn = new double[1];
		logp_patn[0] = 0.0; // log-compensate for this pattern
		double[] patnPartialsResult;
		int singlePartialSizeParent;

		final double[] patnPartialsForNative;
		if (patnPartials.length == partialSizeCurr) {
			patnPartialsForNative = patnPartials; // high-res: no copy needed
		} else {
			patnPartialsForNative = Arrays.copyOf(patnPartials, partialSizeCurr); // low-res: trim
		}

		if (node.isRoot()) {
			// Normalize at root so that sum(valid D entries) = 1, and accumulate
			// the log compensation exactly as computeSingleBranchLikelihood does
			boolean rootIsLow = isLowResolution(node);
			logp_patn[0] += normalization(rootIsLow, patnPartialsForNative);
			patnPartialsResult = patnPartialsForNative;
			singlePartialSizeParent = partialSizeCurr;
		} else {
		
			// propagate along the parent branch
			patnPartialsResult = computeSingleBranchLikelihood(node.getParent(), node, patnPartialsForNative, logp_patn, categoryID, threadID);

			// the size of the resulting partials is based on its parent
			int numRateBins_parent = numRateBins_h;
			if (isLowResolution(node.getParent())) {
				// assume all leaves should have a parent
				numRateBins_parent = numRateBins_l;
			}
			singlePartialSizeParent = numPlan * numRateBins_parent;
		
			// to make sure the array length is correct
			assert(patnPartialsResult.length == singlePartialSizeParent);
		}
		int curr_t = node.getNr() * patterns;
		int subpatnid = pattern2SubpatnPerNode[curr_t + patternIndex];
		int st = subpatnid * singlePartialSizeParent;
		System.arraycopy(patnPartialsResult, 0, partialsAllPatterns, st, singlePartialSizeParent);
		
		double compensate = logp_patn[0] + leftCompensate + rightCompensate;
		// System.out.println("node: " + node.getNr() + "; patternIndex = " + patternIndex + " has compensate = " + compensate);
		return compensate;
	 }
	
	/**
	 * compute the partial likelihood array
	 *
	 * @param node
	 */
	protected void computePartialLikelihood(Node node) {
        if (node.isLeaf()) return;

		// internal node or the root node
		double[] patternPartialsLeft = mosseLikelihoodCore.getNodeMossePartials(node.getLeft().getNr());
		double[] patternPartialsRight = mosseLikelihoodCore.getNodeMossePartials(node.getRight().getNr());
		double[] patternCompensatesLeft = mosseLikelihoodCore.getNodeMosseCompensates(node.getLeft().getNr());
		double[] patternCompensatesRight = mosseLikelihoodCore.getNodeMosseCompensates(node.getRight().getNr());

		// the resulting partial size of single pattern is based on its parent's
		// resolution (or the root's own resolution when there is no parent).
		// isLowResolution(node) covers the root correctly for all three modes.
		int numRateBins_curr = isLowResolution(node) ? numRateBins_l : numRateBins_h;
		int singlePartialSize = numPlan * numRateBins_curr;
		int subpatns = computeMapGlobal2SubpatternID(node);

		assert (subpatns > 0);

		double[] partialsAllPatterns = new double[subpatns * singlePartialSize];
		double[] compensatesAllPatterns = new double[subpatns];

		int[] rep = new int[subpatns]; // representative pattern for the sub-pattern
        Arrays.fill(rep, -1);

		int t = node.getNr() * patterns; // start pos in pattern2SubpatnPerNode
        if (node.isRoot()) {
            // root: subpattern == patternIndex, so representative is itself
            for (int p = 0; p < patterns; p++) {
            	rep[p] = p;
            }
        } else {
            for (int p = 0; p < patterns; p++) {
                int sid = pattern2SubpatnPerNode[t + p];
                if (rep[sid] < 0) rep[sid] = p;
            }
        }

		if (pool == null) {
			// single thread
			int threadID = 0;
			for (int sid = 0; sid < subpatns; sid++) {
	            int patternIndex = rep[sid];
	            if (patternIndex >= 0) {
	            	compensatesAllPatterns[sid] = computePartialLikelihoodPattern(patternIndex, node, patternPartialsLeft, patternPartialsRight, partialsAllPatterns, patternCompensatesLeft, patternCompensatesRight, 0, threadID);
	            }
			}
		} else {
			// multiple threads
	        runInPool(() -> IntStream.range(0, subpatns).parallel().forEach(sid -> {
	            int patternIndex = rep[sid];
	            if (patternIndex >= 0) {
	            	int threadID = threadIndexInPool();
	            	compensatesAllPatterns[sid] = computePartialLikelihoodPattern(patternIndex, node, patternPartialsLeft, patternPartialsRight, partialsAllPatterns, patternCompensatesLeft, patternCompensatesRight, 0, threadID);
	            }
	        }));
		}

        mosseLikelihoodCore.setNodeMossePartials(node.getNr(), partialsAllPatterns);
        mosseLikelihoodCore.setNodeMosseCompensates(node.getNr(), compensatesAllPatterns);

        if (node.isRoot()) {
            boolean conditionSurv = true;
            // Choose root-resolution parameters based on the active resolution mode.
            // In "high" mode the root operates at high resolution; in all other modes
            // (mixed and low) the root always operates at low resolution.
            boolean rootIsLow = !RESOLUTION_MODE_HIGH.equals(resolutionMode);
            int    nx_root = rootIsLow ? numRateBins_l : numRateBins_h;
            double dx_root = rootIsLow ? dx_l          : dx_h;
	        if (pool == null) {
	        	// single thread
	        	for (int p = 0; p < patterns; p++) {
	        		if (compensatesAllPatterns[p] == Double.NEGATIVE_INFINITY || Double.isNaN(compensatesAllPatterns[p])) {
	        			patternLogLikelihoods[p] = Double.NEGATIVE_INFINITY;
	        		} else {
		                int startPos = pattern2SubpatnPerNode[t + p] * singlePartialSize;
		                double[] partials = new double[singlePartialSize];
		                System.arraycopy(partialsAllPatterns, startPos, partials, 0, singlePartialSize);
		                patternLogLikelihoods[p] = makeRootFuncMosse(nx_root, dx_root, resolution, partials, conditionSurv)
		                		+ compensatesAllPatterns[p];
	        		}
	        	}
	        } else {
	        	// multi-threaded
	            runInPool(() -> IntStream.range(0, patterns).parallel().forEach(p -> {
	        		if (compensatesAllPatterns[p] == Double.NEGATIVE_INFINITY || Double.isNaN(compensatesAllPatterns[p])) {
	        			patternLogLikelihoods[p] = Double.NEGATIVE_INFINITY;
	        		} else {
		                int startPos = pattern2SubpatnPerNode[t + p] * singlePartialSize;
		                double[] partials = new double[singlePartialSize];
		                System.arraycopy(partialsAllPatterns, startPos, partials, 0, singlePartialSize);
		                patternLogLikelihoods[p] = makeRootFuncMosse(nx_root, dx_root, resolution, partials, conditionSurv)
		                		+ compensatesAllPatterns[p];
	        		}
	            }));
	        }
        }
	}

	/**
	 *
	 * @param nx            number of bins for substitution rate
	 * @param dx            distance between xs
	 * @param r             for resolution scale factor
	 * @param result        root node result matrix of D and E values (column-major:
	 *                      first nx entries = E, next nx entries = D[state0], etc.)
	 * @param conditionSurv whether to condition on survival
	 * @return log probability for root
	 */
	protected double makeRootFuncMosse(int nx, double dx, int r, double[] result, boolean conditionSurv) {

		// Operate directly on the flat column-major result[] without
		// transposing into a double[nx][numPlan] intermediate. The layout is:
		//   result[j * nx + i]  =>  column j (plan dimension), row i (bin)
		// Column 0 = E values; columns 1..stateCount = D values per state.

		// Select cached substitution-rate array and lambda array that match the
		// resolution at which the root partials were computed.
		boolean rootIsLow = !RESOLUTION_MODE_HIGH.equals(resolutionMode);
		double[] x       = rootIsLow ? cached_x_l : cached_x_h;
		double[] lambdas = rootIsLow ? lambdas_l   : lambdas_h;

		int numEntries = rootIsLow ? treeModel.numEntries_l : treeModel.numEntries_h;

		// Compute root.i: stationary frequencies of Q (left eigenvector for eigenvalue 0).
		// In R: root.i <- solve(t(cbind(c(1,1,1,1), pars$Q_orig[,-1])), c(1,0,0,0))
		// This is equivalent to the substitution model's equilibrium frequencies.
		double[] rootI = substitutionModel.getFrequencies();

		// Project d.root through root.i: d_proj[i] = sum_j( D[j][i] * rootI[j] )
		// In R: d.root <- d.root %*% root.i
		// result layout: result[(j+1)*nx + i] = D[state j][bin i]
		double[] dProj = new double[numEntries];
		for (int i = 0; i < numEntries; i++) {
			double val = 0.0;
			for (int j = 0; j < stateCount; j++) {
				val += result[(j + 1) * nx + i] * rootI[j];
			}
			dProj[i] = val;
		}

		// root.p is computed on the projected (scalar) d.root
		double[] rootP = (sharedRootP != null && !captureRootP) ? sharedRootP : getRootProbFlatProjected(dProj, x, numEntries);

		// Jacobian correction: rootP is density on λ, bins are in log(λ) space
		if (sharedRootP == null || captureRootP) {
			if (logScale)
				for (int i = 0; i < numEntries; i++)
					rootP[i] *= Math.exp(x[i]);
			if (captureRootP) sharedRootP = rootP;
		}

		// survival conditioning is identical for every site + flat: compute once, apply once in calcLogP
		if (conditionSurv && captureRootP) {
			double denom = 0.0;
			for (int i = 0; i < numEntries; i++) {
				double surv = 1.0 - result[i]; // E column
				if (surv < 1e-30) return Double.NEGATIVE_INFINITY;
				denom += rootP[i] * lambdas[i] * surv * surv;
			}
			if (denom <= 0.0) return Double.NEGATIVE_INFINITY;
			survDenom = denom;
		}

		// compute product sum: sum_i rootP[i] * dProj[i]
		double sum = 0.0;
		for (int i = 0; i < numEntries; i++) {
			sum += rootP[i] * dProj[i];
		}
		// double logProb = Math.log(sum);
		double logProb = Math.log(sum);
		return logProb;
	}

	/**
	 * Compute root probability weights for the projected (single-column) d.root vector.
	 * Mirrors root.p.mosse() in the updated R after d.root has been projected via root.i.
	 */
	private double[] getRootProbFlatProjected(double[] dProj, double[] x, int numEntries) {
		double[] p = new double[numEntries];
		if (rootOption == ROOT_OBS) {
			// p <- d.root / sum(d.root)
			double dsum = 0.0;
			for (int i = 0; i < numEntries; i++) dsum += dProj[i];
			double factor = (dsum == 0.0) ? 0.0 : 1.0 / dsum;
			for (int i = 0; i < numEntries; i++) p[i] = dProj[i] * factor;
		} else if (rootOption == ROOT_FLAT) {
			// p <- 1 / (pars$nx - 1)
			double val = 1.0 / (numEntries - 1);
			Arrays.fill(p, val);
		} else if (rootOption == ROOT_GIVEN && rootFunc != null) {
			// p <- root.f(x)
			// In log-scale mode, x[] holds log-rate values; LinkFns expect rate-space input.
			double[] xForLinkFn = logScale ? expArray(x) : x;
			double[] y = new double[xForLinkFn.length];
			y = rootFunc.getY(xForLinkFn, y);
			for (int i = 0; i < numEntries; i++) {
				p[i] = (i < y.length) ? y[i] : 0.0;
			}
		} else {
			throw new RuntimeException("Unsupported root option: " + rootOption);
		}
		return p;
	}

	// Punctuational support: elementwise exp, used at the LinkFn boundary in log-scale mode.
	private static double[] expArray(double[] x) {
		double[] out = new double[x.length];
		for (int i = 0; i < x.length; i++) out[i] = Math.exp(x[i]);
		return out;
	}

	/**
	 * Apply P = I + a*Q to a state vector d (in place).
	 * When a == 0 this is a no-op (P = I).
	 */
	private void applyPuncMatrix(double[] d, double a) {
		if (a == 0.0 || qFlat == null) return;
		double[] result = new double[stateCount];
		for (int i = 0; i < stateCount; i++) {
			double sum = 0.0;
			for (int s = 0; s < stateCount; s++)
				sum += qFlat[i * stateCount + s] * d[s];
			result[i] = d[i] + a * sum;
		}
		System.arraycopy(result, 0, d, 0, stateCount);
	}

	private double[] getSubstitutionRates(int numElements, double startRate, double dx, double padLeft) {
		double[] res = new double[numElements];
		res[0] = startRate + dx * padLeft;
		for (int i = 1; i < numElements; i++) {
			res[i] = res[i - 1] + dx;
		}
		return res;
	}

	private double[] getTraits(Node node) {
		String taxonName = node.getID();
		double[] traitValues = new double[traits.size()];
		for (int i = 0; i < traits.size(); i++) {
			double traitValue = traits.get(i).getValue(taxonName);
			traitValues[i] = traitValue;
		}
		return traitValues;
	}

	@Override
	public double calculateLogP() {
		// check whether beta is out of the range
		if (paramsOutOfRange())
			return Double.NEGATIVE_INFINITY;

		// reuse the flat-pass rootP for every per-site root (first pass also inits native plans)
		traverseFull(tree.getRoot());
		captureRootP = true;
		lastFlatLogP = computeFlatTreeLogLikelihood();
		captureRootP = false;
		traverseFull(tree.getRoot());
		calcLogP();
		sharedRootP = null;
		return logP;
	}
	
	// count out-of-range guard hits per reason; warnGuard logs only the first 10 of each
	private final java.util.Map<String, Integer> guardWarnCount = new java.util.HashMap<>();
	private boolean warnGuard(String key) {
		return guardWarnCount.merge(key, 1, Integer::sum) <= 10;
	}

	public boolean paramsOutOfRange() {
		// check whether beta is out of the range
		if (tipModel.betaOutOfRange(beta_lowbound, beta_upbound)) {
			if (warnGuard("beta")) System.err.println("paramsOutOfRange: beta outside [" + beta_lowbound + ", " + beta_upbound + "]");
			return true;
		}
		// epsilon must be >= grid spacing dx
		double dxTip = RESOLUTION_MODE_LOW.equals(resolutionMode) ? dx_l : dx_h;
		if (tipModel.epsilon.getValue().doubleValue() < dxTip) {
			if (warnGuard("epsilon")) System.err.println("paramsOutOfRange: epsilon (" + tipModel.epsilon.getValue()
				+ ") < grid spacing dx (" + dxTip + ")");
			return true;
		}
		// diffusion kernel width sd = sqrt(dt*diffusion) must be >= 0.2*dx (calibrated),
		// else the FFT convolution is sub-grid / ill-conditioned (non-reproducible logP)
		if (Math.sqrt(treeModel.diffusion * deltaT) < 0.2 * dxTip) {
			if (warnGuard("diffusion")) System.err.println("paramsOutOfRange: diffusion kernel sd=sqrt(dt*diffusion)="
				+ Math.sqrt(treeModel.diffusion * deltaT) + " < 0.2*dx=" + (0.2 * dxTip)
				+ " (diffusion=" + treeModel.diffusion + ")");
			return true;
		}

		// Check that tip substitution-rate distributions stay within the FFT grid.
		// If subst+epsilon shifts the distribution outside [rmin, rmax], the
		// tip likelihoods are truncated and the likelihood computation is inaccurate.
		double substV   = tipModel.meanSubstitution.getValue();
		double epsilonV = tipModel.epsilon.getValue();
		double maxEffect = 0.0;
		double minEffect = 0.0;
		for (int i = 0; i < tipModel.beta.getDimension(); i++) {
			double b = tipModel.beta.getValue(i);
			maxEffect += Math.max(b * traitmax_global, b * traitmin_global);
			minEffect += Math.min(b * traitmax_global, b * traitmin_global);
		}
		// require 3-sigma coverage within the grid
		if (substV + maxEffect + 3.0 * epsilonV > rmax) {
			if (warnGuard("gridHi")) System.err.println("paramsOutOfRange: tip rate distribution above grid: subst+maxEffect+3*epsilon="
				+ (substV + maxEffect + 3.0 * epsilonV) + " > rmax=" + rmax);
			return true;
		}
		if (substV + minEffect - 3.0 * epsilonV < rmin) {
			if (warnGuard("gridLo")) System.err.println("paramsOutOfRange: tip rate distribution below grid: subst+minEffect-3*epsilon="
				+ (substV + minEffect - 3.0 * epsilonV) + " < rmin=" + rmin);
			return true;
		}

		// Prevent LinearFunction degeneracy: when linearGrowthRate is so high that
		// the ramp saturates for all tip rate distributions, r becomes non-identifiable.
		// The ramp: y = y0 + r*x for 0 < x < (y1-y0)/r, then caps at y1.
		// Require the cap threshold (y1-y0)/r >= maxPositiveRate / 3.
		// Equivalently: r * maxPositiveRate <= 3 * (y1-y0).
//		if (lambdaFunc instanceof LinearFunction) {
//			LinearFunction lf = (LinearFunction) lambdaFunc;
//			double y0_lf = lf.curveYBaseValueInput.get().getValue();
//			double y1_lf = lf.curveMaxYInput.get().getValue();
//			double r_lf  = lf.linearGrowthRateInput.get().getValue();
//			double yRange = y1_lf - y0_lf;
//			double maxPosRate = substV + maxEffect;
//			if (yRange > 0 && maxPosRate > 0 && r_lf * maxPosRate > 3.0 * yRange) {
//				return true;
//			}
//		}

		// Punctuational matrix sanity: P = I + a*Q must have non-negative diagonals.
		//   P[i,i] = 1 + a*Q[i,i] = 1 - a*|Q[i,i]|
		// where |Q[i,i]| = sum_{j!=i} Q[i,j] (off-diagonals of a valid rate matrix
		// are non-negative). If a * max_i |Q[i,i]| > 1, the diagonal of P goes
		// negative, the substitution "probabilities" become invalid, and the
		// likelihood is undefined — reject the proposal with -inf.
		// Compute Q via (P(tau) - I) / tau for small tau (the same trick as
		// buildQFlat), so we don't assume a particular substitution model.
		if (treeModel.a > 0.0) {
			final double tau = 1e-4;
			int sq = stateCount * stateCount;
			double[] Pt = new double[sq];
			substitutionModel.getTransitionProbabilities(tree.getRoot(), tau, 0, 1.0, Pt);
			double maxDiagMag = 0.0;
			for (int i = 0; i < stateCount; i++) {
				double sumOffDiag = 0.0;
				for (int j = 0; j < stateCount; j++) {
					if (i == j) continue;
					double q_ij = Pt[i * stateCount + j] / tau;
					if (q_ij > 0) sumOffDiag += q_ij;
				}
				if (sumOffDiag > maxDiagMag) maxDiagMag = sumOffDiag;
			}
			if (treeModel.a * maxDiagMag > 1.0) {
				if (warnGuard("punc")) System.err.println("paramsOutOfRange: punctuation a*max|Q_ii|=" + (treeModel.a * maxDiagMag)
					+ " > 1 (P=I+a*Q has negative diagonal)");
				return true;
			}
		}

		return false;
	}
	
	protected void printParams() {
		String newickstr = toNewick(tree.getRoot()) + ";";
		System.out.println(newickstr);
		System.out.println("resolutionMode = " + resolutionMode);
		System.out.println("tc = " + tc);
		System.out.println("dx_h = " + dx_h);
		System.out.println("rmin = " + rmin);
		System.out.println("rmax = " + rmax);
		
		printSiteModelParameters();
		tipModel.printParams();
		treeModel.printParams();
		lambdaFunc.printParams();
		muFunc.printParams();
	}

	protected void printLogP() {
		printParams();
		// printSiteCatLikes(patternLogLikelihoods, 4);
		count++;
		System.out.println("#" + count + " logP = " + df.format(logP));
		System.out.println();
	}

	@Override
	protected void calcLogP() {
		logP = 0.0;
		if (useAscertainedSitePatterns) {
			final double ascertainmentCorrection = data.getAscertainmentCorrection(patternLogLikelihoods);
			for (int i = 0; i < data.getPatternCount(); i++) {
				logP += (patternLogLikelihoods[i] - ascertainmentCorrection) * data.getPatternWeight(i);
			}
		} else {
			// Each patternLogLikelihoods[i] = log p(tree AND site_i AND tip traits).
			// Correct for the tree-and-traits probability being counted N times:
			// logL = sum_i w_i * logL_i - (N-1) * logL_flat
			// where logL_flat = log p(tree AND tip traits) computed with flat tip likelihoods
			// (D_tip[s][bin] = tipLikelihoods[bin] for all nucleotide states s).

			int totalSites = 0;
			for (int i = 0; i < patterns; i++) {
				totalSites += data.getPatternWeight(i);
			}

			double logL_flat = lastFlatLogP;
			if (!Double.isFinite(logL_flat)) { logP = Double.NEGATIVE_INFINITY; return; }
			double logSurvDenom = Math.log(survDenom);

			for (int i = 0; i < patterns; i++) {
				if (!Double.isFinite(patternLogLikelihoods[i])) { logP = Double.NEGATIVE_INFINITY; return; }
				if (patternLogLikelihoods[i] > logL_flat + 1e-6 || patternLogLikelihoods[i] - logSurvDenom > 0.0) { logP = Double.NEGATIVE_INFINITY; return; }
				logP += patternLogLikelihoods[i] * data.getPatternWeight(i);
			}
			logP -= (totalSites - 1) * logL_flat;
			logP -= logSurvDenom;

			if (logP > 0.0) logP = Double.NEGATIVE_INFINITY;
		}
	}

	/**
	 * Compute the flat tree log-likelihood: log p(tree AND tip traits) using flat
	 * tip likelihoods where D_tip[s][bin] = tipLikelihoods[bin] for ALL nucleotide
	 * states s. Used in: logP = sum_p w_p * logL_p - (N-1) * computeFlatTreeLogLikelihood()
	 */
	protected double computeFlatTreeLogLikelihood() {
		int N = tree.getNodeCount();
		double[][] flatPartials = new double[N][];
		double[] compensates = new double[N];
		Node root = tree.getRoot();
		if (pool != null && !captureRootP) {
			// Empirically (jstack), the serial doFlatTraversal on the main thread was
			// the dominant Amdahl bottleneck: while ~200 pool workers slept,
			// the main thread did the same per-branch JNI work serially. Mirror the
			// tree-level parallelism used in traverseFull (TraverseTask) so this
			// second pass runs on the pool.
			pool.invoke(new FlatTraverseTask(root, flatPartials, compensates));
		} else {
			doFlatTraversal(root, flatPartials, compensates);
		}
		boolean rootIsLow = !RESOLUTION_MODE_HIGH.equals(resolutionMode);
		int nx_root = rootIsLow ? numRateBins_l : numRateBins_h;
		double dx_root = rootIsLow ? dx_l : dx_h;
		double logProb = makeRootFuncMosse(nx_root, dx_root, resolution, flatPartials[root.getNr()], true);
		return logProb + compensates[root.getNr()];
	}

	/**
	 * Post-order traversal for the flat computation.
	 * Stores propagated flat partials in flatPartials[] and accumulated log-compensations in compensates[].
	 */
	private void doFlatTraversal(Node node, double[][] flatPartials, double[] compensates) {
		if (node.isLeaf()) {
			flatPartials[node.getNr()] = computeFlatLeafAndPropagate(node, compensates);
			return;
		}
		doFlatTraversal(node.getLeft(),  flatPartials, compensates);
		doFlatTraversal(node.getRight(), flatPartials, compensates);
		combineFlatInternalNode(node, flatPartials, compensates);
	}

	/**
	 * Body of the post-order combine for a non-leaf node in the flat-likelihood
	 * traversal. Reads both children's flatPartials/compensates, writes this node's.
	 * Safe to call concurrently for different nodes — each task writes to its own
	 * index in flatPartials[] / compensates[], and reads only from already-joined
	 * children. Uses threadIndexInPool() so each pool worker gets its own per-thread
	 * native FFT plan (previously hard-coded to slot 0, which would corrupt under
	 * parallel execution).
	 */
	private void combineFlatInternalNode(Node node, double[][] flatPartials, double[] compensates) {
		boolean nodeIsLow = isLowResolution(node);
		int numRateBins_curr  = nodeIsLow ? numRateBins_l : numRateBins_h;
		int numEntries_curr   = nodeIsLow ? treeModel.numEntries_l : treeModel.numEntries_h;
		double[] lambdas_curr = nodeIsLow ? lambdas_l : lambdas_h;
		int partialSizeCurr   = numPlan * numRateBins_curr;

		double[] partialsLeft  = flatPartials[node.getLeft().getNr()];
		double[] partialsRight = flatPartials[node.getRight().getNr()];

		double[] patnPartials = new double[partialSizeCurr];
		System.arraycopy(partialsLeft, 0, patnPartials, 0, numRateBins_curr); // E from left

		int nEntries = Math.min(numRateBins_curr, numEntries_curr);
		double puncA = treeModel.a;
		for (int j = 0; j < nEntries; j++) {
			double[] dL = new double[stateCount];
			double[] dR = new double[stateCount];
			for (int s = 0; s < stateCount; s++) {
				dL[s] = partialsLeft[(s + 1) * numRateBins_curr + j];
				dR[s] = partialsRight[(s + 1) * numRateBins_curr + j];
			}
			applyPuncMatrix(dL, puncA);
			applyPuncMatrix(dR, puncA);
			double lambdaX = lambdas_curr[j];
			for (int i = 0; i < stateCount; i++)
				patnPartials[(i + 1) * numRateBins_curr + j] = dL[i] * dR[i] * lambdaX;
		}
		for (int j = nEntries; j < numRateBins_curr; j++)
			for (int i = 0; i < stateCount; i++)
				patnPartials[(i + 1) * numRateBins_curr + j] = 0.0;

		double logCompensate;
		if (node.isRoot()) {
			boolean rootIsLow = !RESOLUTION_MODE_HIGH.equals(resolutionMode);
			logCompensate = normalization(rootIsLow, patnPartials);
			flatPartials[node.getNr()] = patnPartials;
		} else {
			double[] logp = new double[1];
			double[] result = computeSingleBranchLikelihood(node.getParent(), node, patnPartials, logp, 0, threadIndexInPool());
			flatPartials[node.getNr()] = result;
			logCompensate = logp[0];
		}

		compensates[node.getNr()] = logCompensate
				+ compensates[node.getLeft().getNr()]
				+ compensates[node.getRight().getNr()];
	}

	/**
	 * RecursiveAction mirror of TraverseTask, but for the flat-likelihood post-order.
	 * Forks the right subtree, computes the left in the current worker, joins, then
	 * combines this internal node. Leaves call the same computeFlatLeafAndPropagate
	 * helper as the serial path.
	 */
	private final class FlatTraverseTask extends RecursiveAction {
		private static final long serialVersionUID = 1L;
		private final Node node;
		private final double[][] flatPartials;
		private final double[]   compensates;
		FlatTraverseTask(Node node, double[][] flatPartials, double[] compensates) {
			this.node = node;
			this.flatPartials = flatPartials;
			this.compensates  = compensates;
		}
		@Override
		protected void compute() {
			if (node.isLeaf()) {
				flatPartials[node.getNr()] = computeFlatLeafAndPropagate(node, compensates);
				return;
			}
			FlatTraverseTask right = new FlatTraverseTask(node.getRight(), flatPartials, compensates);
			right.fork();
			new FlatTraverseTask(node.getLeft(), flatPartials, compensates).compute();
			right.join();
			combineFlatInternalNode(node, flatPartials, compensates);
		}
	}

	/**
	 * Compute the flat leaf partial (D[s][bin] = tipLikelihoods[bin] for all states s)
	 * and propagate it along the leaf's parent branch.
	 * Stores the resulting log-compensation in compensates[leaf.getNr()].
	 */
	private double[] computeFlatLeafAndPropagate(Node leaf, double[] compensates) {
		boolean leafIsLow = RESOLUTION_MODE_LOW.equals(resolutionMode);
		int numRateBins_leaf    = leafIsLow ? numRateBins_l : numRateBins_h;
		int numEntries_leaf     = leafIsLow ? treeModel.numEntries_l : treeModel.numEntries_h;
		int padLeft_leaf        = leafIsLow ? treeModel.padLeft_l : treeModel.padLeft_h;
		double startSubsRate_leaf = leafIsLow ? startSubsRate_l : startSubsRate_h;
		double dx_leaf          = leafIsLow ? dx_l : dx_h;
		int singlePartialSizeLeaf = (stateCount + 1) * numRateBins_leaf;

		double[] traitValues = getTraits(leaf);
		double[] tipLikelihoods = tipModel.getTipLikelihoods(traitValues, numEntries_leaf,
				startSubsRate_leaf + padLeft_leaf * dx_leaf, dx_leaf);

		double[] patnPartials = new double[singlePartialSizeLeaf];
		// E values are zero at the tip (Java array initialization)

		int k = numRateBins_leaf;
		for (int state = 0; state < stateCount; state++) {
			// Flat: all nucleotide states receive tipLikelihoods (no observed-state indicator)
			if (numRateBins_leaf <= numEntries_leaf) {
				for (int i = 0; i < numRateBins_leaf; i++) {
					patnPartials[k++] = tipLikelihoods[i];
				}
			} else {
				for (int i = 0; i < padLeft_leaf; i++) { patnPartials[k++] = 0.0; }
				for (int i = 0; i < numEntries_leaf; i++) { patnPartials[k++] = tipLikelihoods[i]; }
				for (int i = padLeft_leaf + numEntries_leaf; i < numRateBins_leaf; i++) { patnPartials[k++] = 0.0; }
			}
		}

		double[] logp = new double[1];
		double[] result = computeSingleBranchLikelihood(leaf.getParent(), leaf, patnPartials, logp, 0, threadIndexInPool());
		compensates[leaf.getNr()] = logp[0];
		return result;
	}

	@Override
	protected boolean requiresRecalculation() {
		// Mark transition matrices dirty whenever the substitution model
		// (or RHAS rates) may have changed so they are recomputed on the next traversal.
		if (m_siteModel.isDirtyCalculation()) {
			transitionMatricesDirty = true;
		}
		// always recalculate the tree likelihood itself
		return true;
	}

	public static String toNewick(Node n) {
		StringBuilder sb = new StringBuilder();

		if (!n.isLeaf()) {
			sb.append("(");
			for (int i = 0; i < n.getChildCount(); i++) {
				if (i > 0) {
					sb.append(",");
				}
				sb.append(toNewick(n.getChild(i)));
			}
			sb.append(")");
		} else {
			// taxon label or fallback ID
			String label = n.getID() != null ? n.getID() : "node" + n.getNr();
			sb.append(label);
		}

		sb.append(":").append(n.getLength());
		return sb.toString();
	}

	/**
	 * Returns true if the given node should be computed at low resolution.
	 *
	 * Behaviour depends on {@code resolutionMode}:
	 * <ul>
	 *   <li>{@value #RESOLUTION_MODE_MIXED} (default) — the existing tc-based rule:
	 *       leaves are always high-resolution; the root and any internal node at or
	 *       above {@code tc} are low-resolution; nodes below {@code tc} are
	 *       high-resolution.</li>
	 *   <li>{@value #RESOLUTION_MODE_LOW} — every node (including leaves) is
	 *       treated as low-resolution.</li>
	 *   <li>{@value #RESOLUTION_MODE_HIGH} — every node (including the root) is
	 *       treated as high-resolution.</li>
	 * </ul>
	 */
	protected boolean isLowResolution(Node node) {
		switch (resolutionMode) {
			case RESOLUTION_MODE_LOW:
				return true;   // entire tree uses low resolution
			case RESOLUTION_MODE_HIGH:
				return false;  // entire tree uses high resolution
			default: // RESOLUTION_MODE_MIXED
				if (node.isLeaf()) {
					return false; // leaf always uses high resolution
				} else if (node.isRoot() || node.getHeight() >= tc) {
					return true; // low resolution for root and nodes at or above tc
				} else {
					return false; // high resolution for nodes below tc
				}
		}
	}

	private double[] reduceSize(double[] partials) {
		// reduce the size of partials to "numPlan * numRateBins_l"
		int partialNewSize = numPlan * numRateBins_l;
		double[] partialsNew = new double[partialNewSize];
		// selecting the corresponding entries in the partialMiddle as the input for low
		// resolution
		for (int i = 0; i < numPlan; i++) {
			int k = 0;
			int s_h = i * numRateBins_h;
			int s_l = i * numRateBins_l;
			for (int j = resolution - 1; j < numRateBins_h && k < numRateBins_l; j += resolution, k++) {
				partialsNew[s_l + k] = partials[s_h + j];
			}
			while (k < numRateBins_l) {
				partialsNew[s_l + k] = 0.0;
				k++;
			}
		}
		return partialsNew;
	}

	/**
	 * @return the corresponding thread ID
	 * always zero for this single-threaded class
	 */
    protected int threadIndexInPool() {
        return 0;
    }
	

    public void printSiteModelParameters() {

        SubstitutionModel subst =
                m_siteModel.substModelInput.get();

        System.out.println("Substitution model: "
                + subst.getClass().getSimpleName());

        printBEASTObjectParameters(subst, "  ");
    }

    private void printBEASTObjectParameters(Object obj, String indent) {

        if (!(obj instanceof beast.base.core.BEASTObject)) {
            return;
        }

        Map<String, Input<?>> inputs =
                ((beast.base.core.BEASTObject) obj).getInputs();

        for (Map.Entry<String, Input<?>> e : inputs.entrySet()) {
            Input<?> input = e.getValue();
            Object value = input.get();

            if (value == null) continue;

            // Case 1: RealParameter
            if (value instanceof RealParameter rp) {
                System.out.print(indent + e.getKey() + " = ");
                for (int i = 0; i < rp.getDimension(); i++) {
                    System.out.print(rp.getValue(i) + " ");
                }
                System.out.println();
            }

            // Case 2: nested BEASTObject (e.g. Frequencies)
            else if (value instanceof beast.base.core.BEASTObject) {
                // System.out.println(indent + e.getKey() + ":");
                // printBEASTObjectParameters(value, indent + "  ");
            	printBEASTObjectParameters(value, "  ");
            }
        }
    }
    
    protected double logSumExp(final double[] logs) {
        double max = Double.NEGATIVE_INFINITY;
        for (double x : logs) {
        	if (x > max) {
        		max = x;
        	}
        }
        double sum = 0.0;
        for (double x : logs) {
        	sum += Math.exp(x - max);
        }
        return max + Math.log(sum);
    }

	protected static void show1DArray(double[] array, String desc) {
		System.out.println(desc);
		boolean error_found = false;
		int max_num_per_line = 30;
		for (int i = 0; i < array.length; i++) {
			if (i > 0 && i % max_num_per_line == 0)
				System.out.println();
			System.out.print("," + array[i]);
			if (Double.isNaN(array[i]))
				error_found = true;
		}
		System.out.println();
		if (error_found)
			System.exit(1);
	}
	
	protected static void show1DArray(double[] array, String desc, int max_num_per_line) {
		System.out.println(desc);
		boolean error_found = false;
		for (int i = 0; i < array.length; i++) {
			if (i > 0 && i % max_num_per_line == 0)
				System.out.println();
			System.out.print("," + array[i]);
			if (Double.isNaN(array[i]))
				error_found = true;
		}
		System.out.println();
		if (error_found)
			System.exit(1);
	}

	protected static void show1DArrayTranspose(double[] array, String desc, int items_per_col) {
		System.out.println(desc);
		assert (array.length % items_per_col == 0);
		int ncols = array.length / items_per_col;
		for (int i = 0; i < items_per_col; i++) {
			for (int j = 0; j < ncols; j++) {
				if (j > 0)
					System.out.print(",");
				System.out.print(array[j*items_per_col + i]);
			}
			System.out.println();
		}
	}

	protected static void show2DArray(double[][] array, String desc) {
		System.out.println(desc);
		boolean error_found = false;
		for (int i = 0; i < array.length; i++) {
			for (int j = 0; j < array[0].length; j++) {
				System.out.print("," + array[i][j]);
				if (Double.isNaN(array[i][j]))
					error_found = true;
			}
			System.out.println();
		}
		if (error_found)
			System.exit(1);
	}
}
