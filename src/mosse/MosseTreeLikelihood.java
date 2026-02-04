package mosse;

import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

import org.jblas.DoubleMatrix;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Log;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.likelihood.TreeLikelihood;
import beast.base.evolution.sitemodel.SiteModel;
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

	protected LinkFn rootFunc;
	protected LinkFn lambdaFunc;
	protected LinkFn muFunc;
	protected List<TraitSet> traits;
	protected MosseTipLikelihood tipModel;
	protected MosseDistribution treeModel;
	protected double tc; // time < tc for high resolution, while time >= tc for low resolution

	// variables for high resolution
	protected int numRateBins_h;
	protected double dx_h;
	protected double startSubsRate_h;
	protected double[] lambdas_h;
	protected double[] mus_h;
	protected double[][] flatTransitionMatrices_h;

	// variables for low resolution
	protected int numRateBins_l;
	protected double dx_l;
	protected double startSubsRate_l;
	protected double[] lambdas_l;
	protected double[] mus_l;
	protected double[][] flatTransitionMatrices_l;

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
    
	// the format of the displayed log-likelihood value
	DecimalFormat df;
	
	// gamma distribution
	protected int numCategories; // number of categories of the gamma distribution
	protected double[] categoryRates; // rates for all categories
	protected double[] categoryProps; // proportions for all categories
	protected double single_rate = 1.0;

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

		// high resolution
		numRateBins_h = treeModel.numRateBins_h;
		dx_h = treeModel.dx;
		startSubsRate_h = dx_h;
		// low resolution
		numRateBins_l = treeModel.numRateBins_l;
		dx_l = treeModel.dx * resolution;
		startSubsRate_l = dx_l;
				
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

		// RHAS
		numCategories = m_siteModel.getCategoryCount();
		if (numCategories == 1) {
			categoryRates = null;
			categoryProps = null;
		} else {
			categoryRates = m_siteModel.getCategoryRates(null);
			categoryProps = m_siteModel.getCategoryProportions(null);
		}
		showRHASParams();

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
		
		// for this class, always single-threaded
		pool = null;
		
		// the format of the displayed log-likelihood value
		df = new DecimalFormat();
		df.setMaximumFractionDigits(7);
		df.setRoundingMode(RoundingMode.CEILING);
		
		count= 0;
	}

	protected void computeLambdaMus() {
		
		// update the values of pads and numEntries
		treeModel.computePadNumEntries();

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
		double[] x_h = getSubstitutionRates(numEntries_h, startSubsRate_h, dx_h, padLeft_h); // substitution rates
		double[] x_l = getSubstitutionRates(numEntries_l, startSubsRate_l, dx_l, padLeft_l); // substitution rates
		lambdaFunc.getY(x_h, lambdas_h);
		lambdaFunc.getY(x_l, lambdas_l);
		muFunc.getY(x_h, mus_h);
		muFunc.getY(x_l, mus_l);
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

		// for leaf, always use high resolution to begin with
		int singlePartialSizeLeaf = (stateCount + 1) * numRateBins_h;
		
		// the size of the partials is based on its parent
		int numRateBins_parent = numRateBins_h;
		if (isLowResolution(node.getParent())) {
			// assume all leaves should have a parent
			numRateBins_parent = numRateBins_l;
		}
		int singlePartialSizeParent = (stateCount + 1) * numRateBins_parent;
		double[][] partials = new double[numCategories][subpatns * singlePartialSizeParent];
		double[][] compensates = new double[numCategories][subpatns];
		
		double[] traitValues = getTraits(node);
		boolean[] updated = new boolean[subpatns];
		Arrays.fill(updated, false);
		
		double subsInterval = startSubsRate_h;
		double[] tipLikelihoods = tipModel.getTipLikelihoods(traitValues, treeModel.numEntries_h,
				startSubsRate_h + treeModel.padLeft_h * subsInterval, subsInterval);

		// compute the partial likelihood for single sub-pattern with categories
		double[][] patnPartials = new double[numCategories][singlePartialSizeLeaf];
		// the first numRateBins_h positions are zeros for E initial values
		int k = 0;
		for (int c = 0; c < numCategories; c++) {
			for (int i = 0; i < numRateBins_h; i++) {
				patnPartials[c][k++] = 0.0;
			}
		}
		for (int patternIndex = 0; patternIndex < patterncount; patternIndex++) {
			int subpatnid = pattern2SubpatnPerNode[s + patternIndex];
			if (!updated[subpatnid]) {
				updated[subpatnid] = true;
				int stateid = data.getPattern(taxonIndex, patternIndex);
				boolean[] stateSet = data.getStateSet(stateid);
				
				k = numRateBins_h;
				// D initial values
				for (int state = 0; state < stateCount; state++) {
					if (stateSet[state]) {
						// set likelihoods for nucleotide in data
						if (numRateBins_h <= treeModel.numEntries_h) {
							for (int i = 0; i < numRateBins_h; i++) {
								patnPartials[0][k++] = tipLikelihoods[i];
							}
						} else {
							for (int i = 0; i < treeModel.numEntries_h; i++) {
								patnPartials[0][k++] = tipLikelihoods[i];
							}
							// set to zeros for the rest
							for (int i = treeModel.numEntries_h; i < numRateBins_h; i++) {
								patnPartials[0][k++] = 0.0;
							}
						}
					} else {
						// otherwise leave likelihoods to zero
						for (int i = 0; i < numRateBins_h; i++) {
							patnPartials[0][k++] = 0.0;
						}
					}
				}
				// copy the same to the other categories
				int copy_size = stateCount * numRateBins_h;
				for (int c = 1; c < numCategories; c++) {
					System.arraycopy(patnPartials[0], numRateBins_h, patnPartials[c], numRateBins_h, copy_size);
				}
				for (int c = 0; c < numCategories; c++) {
					// propagate along the parent branch
					double[] logp_patn = new double[1];
					logp_patn[0] = 0.0; // log-compensate for this pattern
					double[] patnPartialsResult = computeSingleBranchLikelihood(node.getParent(), node, patnPartials[c], logp_patn, c, threadID);
					
					// to make sure the array length is correct
					assert(patnPartialsResult.length == singlePartialSizeParent);
					int st = subpatnid * singlePartialSizeParent;
					System.arraycopy(patnPartialsResult, 0, partials[c], st, singlePartialSizeParent);
					compensates[c][subpatnid] = logp_patn[0];
				}
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
		        Runnable job = () -> IntStream.range(0, leaves.size()).parallel().forEach(sid -> {
	            	int threadID = threadIndexInPool();
					setLeafPartials(leaves.get(sid), patterncount, threadID);
		        });
	            try {
	                pool.submit(job).get();
	            } catch (InterruptedException e) {
	                Thread.currentThread().interrupt();
	                throw new RuntimeException(e);
	            } catch (ExecutionException e) {
	                throw new RuntimeException(e.getCause());
	            }
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
		mosseLikelihoodCore.initialize(nodeCount, patterns, m_siteModel.getCategoryCount(), true,
				m_useAmbiguities.get());

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
		// normalize the values of vars

		int nx = numRateBins_h;
		double dx = dx_h;
		if (lowResolution) {
			nx = numRateBins_l;
			dx = dx_l;
		}

		int totSize = nx * numPlan;
		assert (vars.length >= totSize);

		double vsum = 0.0;
		// ignore the first nx entries (i.e. first row)
		for (int i = nx; i < totSize; i++) {
			vsum += vars[i];
		}
		vsum *= dx;
		for (int i = nx; i < totSize; i++) {
			vars[i] /= vsum;
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

	/**
	 * create a flatTransitionMatrice
	 */
	protected double[] createFlatTransitionMatrice(Node node, double rate, boolean lowResolution) {
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
		double[] transitionMatrix = new double[sqStateCount];
		substitutionModel.getTransitionProbabilities(node, deltaT, 0, dx * rate, transitionMatrix); // startTime is greater
																								// than endTime
		double[][] transitionMatrices = new double[numEntries][transitionMatrix.length];
		DoubleMatrix matrixOne = new DoubleMatrix(stateCount, stateCount, transitionMatrix);
		DoubleMatrix matrixTwo = new DoubleMatrix(stateCount, stateCount, transitionMatrix);

		// skip the first padLeft entries
		for (int i = 0; i < padLeft; i++) {
			// multiplication of matrix
			matrixTwo = matrixOne.mmul(matrixTwo);
		}
		// int l = 0;
		// update transitionMatrices
		transitionMatrices[0] = matrixTwo.toArray();
		// for (int i = padLeft; i < numEntries + padLeft - 1; i++) {
		for (int l = 1; l < numEntries; l++) {
			// multiplication of matrix
			matrixTwo = matrixOne.mmul(matrixTwo);
			// l++;
			transitionMatrices[l] = matrixTwo.toArray();
		}

		return Arrays.stream(transitionMatrices).flatMapToDouble(Arrays::stream).toArray();
	}
	
	/**
	 * create flatTransitionMatrice for all categories 
	 */
	protected double[][] createFlatTransitionMatrices(Node node, boolean lowResolution) {
		double[][] flatTransitionMatrices;
		
		flatTransitionMatrices = new double[numCategories][];
		for (int c = 0; c < numCategories; c++) {
			double rate = single_rate;
			if (categoryRates != null)
				rate = categoryRates[c];
			flatTransitionMatrices[c] = createFlatTransitionMatrice(node, rate, lowResolution);
		}
		
		return flatTransitionMatrices;
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
			logCompen[0] += normalization(lowResolution, partialsIn);
			partialsOut = treeModel.calculateBranchLogP(branchTime, partialsIn, lambdas_h, mus_h,
					flatTransitionMatrices_h[categoryID], lowResolution, threadID);
		} else if (isLowResolution(child)) {
			// if child has low resolution, then low resolution for the whole branch
			lowResolution = true;
			double branchTime = (node.getHeight() - child.getHeight());
			logCompen[0] += normalization(lowResolution, partialsIn);
			partialsOut = treeModel.calculateBranchLogP(branchTime, partialsIn, lambdas_l, mus_l,
					flatTransitionMatrices_l[categoryID], lowResolution, threadID);
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
			logCompen[0] += normalization(lowResolution, partialsIn);
			partialsOut = treeModel.calculateBranchLogP(branchTime, partialsIn, lambdas_h, mus_h,
					flatTransitionMatrices_h[categoryID], lowResolution, threadID);
			// reduce the size of partials to "numPlan * numRateBins_l"
			partialsOut = reduceSize(partialsOut);
			// then low resolution between t_mid and node.getHeight()
			lowResolution = true;
			branchTime = (node.getHeight() - t_mid);
			if (branchTime > 0.0) {
				logCompen[0] += normalization(lowResolution, partialsOut);
				partialsOut = treeModel.calculateBranchLogP(branchTime, partialsOut, lambdas_l, mus_l,
						flatTransitionMatrices_l[categoryID], lowResolution, threadID);
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
			// update RHAS information
			if (numCategories > 1) {
				categoryRates = m_siteModel.getCategoryRates(null);
				categoryProps = m_siteModel.getCategoryProportions(null);
			}
			// create the transition matrices
			boolean lowResolution = true;
			flatTransitionMatrices_l = createFlatTransitionMatrices(node, lowResolution);
			lowResolution = false;
			flatTransitionMatrices_h = createFlatTransitionMatrices(node, lowResolution);
			// compute the partials for all leaves
			setPartials(node, patterns); // all site patterns
		}

		traverseFull(node.getLeft()); // left child
		traverseFull(node.getRight()); // right child
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
			// and compute the mapping between global pattern index and local partial index
			HashMap<ArrayList<Integer>, Integer> subpatn2subpatnid = new HashMap<>();
			subpatns = 0;
			ArrayList<Integer> subpattern = new ArrayList<>();
			int s = node.getNr() * taxonCount; // starting pos in taxIndexUnderNode
			for (int patternIndex = 0; patternIndex < patterns; patternIndex++) {
				// get the subpattern
				subpattern.clear();
				int subpatnid;
				for (int i = 0; i < data.getTaxonCount(); i++) {
					if (taxaIndexUnderNode[s + i] == -1) {
						break;
					}
					int taxonIndex = taxaIndexUnderNode[s + i];
					int stateCount = data.getPattern(taxonIndex, patternIndex);
					subpattern.add(stateCount);
				}
				if (!subpatn2subpatnid.containsKey(subpattern)) {
					subpatn2subpatnid.put(subpattern, subpatns);
					subpatnid = subpatns;
					subpatns++;
				} else {
					subpatnid = subpatn2subpatnid.get(subpattern);
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
		
		double[] partialsLeft = new double[partialSizeCurr];
		System.arraycopy(patternPartialsLeft, leftPos, partialsLeft, 0, partialSizeCurr);
		double[] partialsRight = new double[partialSizeCurr];
		System.arraycopy(patternPartialsRight, rightPos, partialsRight, 0, partialSizeCurr);
		double[] patnPartials = new double[partialSizeCurr];

		int k = 0;
		// E is topology independent
		System.arraycopy(partialsLeft, k, patnPartials, k, numRateBins_curr);
		k += numRateBins_curr;
		
		if (numRateBins_curr <= numEntries_curr) {
			for (int i = 1; i < numPlan; i++) {
				for (int j = 0; j < numRateBins_curr; j++) {
					// non padded elements
					// D_left * D_right * lambda(x)
					double lambdaX = lambdas_curr[j]; // birth rate at substitution rate x
					patnPartials[k] = partialsLeft[k] * partialsRight[k] * lambdaX;
					k++;
				}
			}
		} else {
			for (int i = 1; i < numPlan; i++) {
				for (int j = 0; j < numEntries_curr; j++) {
					// non padded elements
					// D_left * D_right * lambda(x)
					double lambdaX = lambdas_curr[j]; // birth rate at substitution rate x
					patnPartials[k] = partialsLeft[k] * partialsRight[k] * lambdaX;
					k++;
				}
				// leave the rest to zeros
				for (int j = numEntries_curr; j <  numRateBins_curr; j++) {
					patnPartials[k++] = 0.0;
				}
			}
		}
		
		double[] logp_patn = new double[1];
		logp_patn[0] = 0.0; // log-compensate for this pattern
		double[] patnPartialsResult;
		int singlePartialSizeParent;
		
		if (node.isRoot()) {
			patnPartialsResult = patnPartials;
			singlePartialSizeParent = partialSizeCurr;
		} else {
		
			// propagate along the parent branch
			patnPartialsResult = computeSingleBranchLikelihood(node.getParent(), node, patnPartials, logp_patn, categoryID, threadID);

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
		
		return logp_patn[0] + leftCompensate + rightCompensate;
	 }
	
	/**
	 * compute the partial likelihood array
	 *
	 * @param node
	 */
	protected void computePartialLikelihood(Node node) {
        if (node.isLeaf()) return;

		// internal node or the root node
		double[][] patternPartialsLeft = mosseLikelihoodCore.getNodeMossePartials(node.getLeft().getNr());
		double[][] patternPartialsRight = mosseLikelihoodCore.getNodeMossePartials(node.getRight().getNr());
		double[][] patternCompensatesLeft = mosseLikelihoodCore.getNodeMosseCompensates(node.getLeft().getNr());
		double[][] patternCompensatesRight = mosseLikelihoodCore.getNodeMosseCompensates(node.getRight().getNr());
		 
		// the resulting partial size of single pattern is based on parents
		int numRateBins_curr = numRateBins_h;
		if (node.isRoot() || isLowResolution(node.getParent())) {
			numRateBins_curr = numRateBins_l;
		}
		int singlePartialSize = numPlan * numRateBins_curr;
		int subpatns = computeMapGlobal2SubpatternID(node);

		assert (subpatns > 0);
		
		double[][] partialsAllPatterns = new double[numCategories][subpatns * singlePartialSize];
		double[][] compensatesAllPatterns = new double[numCategories][subpatns];

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

        int totjobs = subpatns * numCategories;
		if (pool == null) {
			// single thread
			int threadID = 0;
			for (int jobid = 0; jobid < totjobs; jobid++) {
				int sid = jobid % subpatns;
				int categoryid = jobid / subpatns;
	            int patternIndex = rep[sid];
	            if (patternIndex >= 0) {
	            	compensatesAllPatterns[categoryid][sid] = computePartialLikelihoodPattern(patternIndex, node, patternPartialsLeft[categoryid], patternPartialsRight[categoryid], partialsAllPatterns[categoryid], patternCompensatesLeft[categoryid], patternCompensatesRight[categoryid], categoryid, threadID);
	            }
			}
		} else {
			// multiple threads
	        Runnable job = () -> IntStream.range(0, totjobs).parallel().forEach(jobid -> {
				int sid = jobid % subpatns;
				int categoryid = jobid / subpatns;
	            int patternIndex = rep[sid];
	            if (patternIndex >= 0) {
	            	int threadID = threadIndexInPool();
	            	compensatesAllPatterns[categoryid][sid] = computePartialLikelihoodPattern(patternIndex, node, patternPartialsLeft[categoryid], patternPartialsRight[categoryid], partialsAllPatterns[categoryid], patternCompensatesLeft[categoryid], patternCompensatesRight[categoryid], categoryid, threadID);
	            }
	        });
            try {
                pool.submit(job).get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            } catch (ExecutionException e) {
                throw new RuntimeException(e.getCause());
            }
		}
		
        mosseLikelihoodCore.setNodeMossePartials(node.getNr(), partialsAllPatterns);
        mosseLikelihoodCore.setNodeMosseCompensates(node.getNr(), compensatesAllPatterns);

        if (node.isRoot()) {
            int totaljobs = patterns * numCategories;
            double[] patternCatLogLikes;
            if (categoryProps == null) {
            	patternCatLogLikes = patternLogLikelihoods;
            } else {
	        	// gamma model is used
            	patternCatLogLikes = new double[totaljobs];
            }
            
	        if (pool == null) {
	        	// single thread
	        	for (int jobid = 0; jobid < totaljobs; jobid++) {
	        		int p = jobid / numCategories; // pattern-id
	        		int c = jobid % numCategories; // category-id
	        		if (compensatesAllPatterns[c][p] == Double.NEGATIVE_INFINITY || Double.isNaN(compensatesAllPatterns[c][p])) {
	        			patternCatLogLikes[jobid] = Double.NEGATIVE_INFINITY;
	        		} else {
		                int startPos = pattern2SubpatnPerNode[t + p] * singlePartialSize;
		                double[] partials = new double[singlePartialSize];
		                System.arraycopy(partialsAllPatterns[c], startPos, partials, 0, singlePartialSize);
		
		                boolean conditionSurv = true;
		                double patternLogLikelihood = makeRootFuncMosse(numRateBins_l, dx_l, resolution, partials, conditionSurv);
		                patternCatLogLikes[jobid] = patternLogLikelihood + compensatesAllPatterns[c][p];
	        		}
	        	}
	        } else {
	        	// multi-threaded
	            Runnable rootJob = () -> IntStream.range(0, totaljobs).parallel().forEach(jobid -> {
	        		int p = jobid / numCategories; // pattern-id
	        		int c = jobid % numCategories; // category-id
	        		if (compensatesAllPatterns[c][p] == Double.NEGATIVE_INFINITY || Double.isNaN(compensatesAllPatterns[c][p])) {
	        			patternCatLogLikes[jobid] = Double.NEGATIVE_INFINITY;
	        		} else {
		                int startPos = pattern2SubpatnPerNode[t + p] * singlePartialSize;
		                double[] partials = new double[singlePartialSize];
		                System.arraycopy(partialsAllPatterns[c], startPos, partials, 0, singlePartialSize);
		
		                boolean conditionSurv = true;
		                double patternLogLikelihood = makeRootFuncMosse(numRateBins_l, dx_l, resolution, partials, conditionSurv);
		                patternCatLogLikes[jobid] = patternLogLikelihood + compensatesAllPatterns[c][p];
	        		}
	            });
	            try { pool.submit(rootJob).get(); }
	            catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new RuntimeException(e); }
	            catch (ExecutionException e) { throw new RuntimeException(e.getCause()); }
	        }
	        
	        // compute patternLogLikelihoods
	        if (categoryProps != null) {
	        	// RHAS model is used
	        	double[] logProps = new double[numCategories];
	        	for (int c = 0; c < numCategories; c++)
	        		logProps[c] = Math.log(categoryProps[c]);
		        if (pool == null) {
		        	// single thread
		        	for (int p = 0; p < patterns; p++) {
		        		double[] catLogLikes = new double[numCategories];
		        		int s = p * numCategories;
		        		for (int c = 0; c < numCategories; c++) {
		        			catLogLikes[c] = patternCatLogLikes[s + c] + logProps[c];
		        			// System.out.print("," + catLogLikes[c]);
		        		}
		        		patternLogLikelihoods[p] = logSumExp(catLogLikes);
		        		// System.out.println("," + patternLogLikelihoods[p]);
		        	}
		        } else {
		        	// multi-threaded
		            Runnable rootJob = () -> IntStream.range(0, patterns).parallel().forEach(p -> {
		        		double[] catLogLikes = new double[numCategories];
		        		int s = p * numCategories;
		        		for (int c = 0; c < numCategories; c++) {
		        			catLogLikes[c] = patternCatLogLikes[s + c] + logProps[c];
		        		}
		        		patternLogLikelihoods[p] = logSumExp(catLogLikes);
		            });
		            try { pool.submit(rootJob).get(); }
		            catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new RuntimeException(e); }
		            catch (ExecutionException e) { throw new RuntimeException(e.getCause()); }
		        }
	        }
	        
	        /*
	        // show the patternCatLogLikes array
	        System.out.println("patternCatLogLikes:");
	        int k;
	        for (int i = 0; i < numCategories; i++) {
	        	System.out.print("Cat " + (i+1));
	        	k = i;
	        	for (int j = 0; j < patterns; j++) {
	        		System.out.print("," + patternCatLogLikes[k]);
	        		k += numCategories;
	        	}
	        	System.out.println();
	        }
	        
	        // show the overall log likelihood
        	System.out.print("Overall");
	        for (int j = 0; j < patterns; j++) {
	        	System.out.print("," + patternLogLikelihoods[j]);
	        }
	        System.out.println();

	        // show the frequencies of the patterns
        	System.out.print("Freq");
	        for (int j = 0; j < patterns; j++) {
	        	System.out.print("," + data.getPatternWeight(j));
	        }
	        System.out.println();
	        */
        }
	}

	/**
	 *
	 * @param nx            number of bins for substitution rate
	 * @param dx            distance between xs
	 * @param r             for resolution scale factor
	 * @param result        root node result matrix of D and E values
	 * @param conditionSurv whether to condition on survival
	 * @return log probability for root
	 */
	protected double makeRootFuncMosse(int nx, double dx, int r, double[] result, boolean conditionSurv) {

		double[][] vals = new double[nx][numPlan];
		int count = 0;
		for (int j = 0; j < numPlan; j++) { // nucleotide types columns
			for (int i = 0; i < nx; i++) { // nx rows
				vals[i][j] = result[count];
				count++;
			}
		}

		double[][] dRoot = getDValues(vals); // get root D values in last column

		double[] eRoot = null;

		double[] x = getSubstitutionRates(treeModel.numEntries_l, startSubsRate_l, dx_l, treeModel.padLeft_l);
		// root options
		double[][] rootP = getRootProb(dRoot, x, nx, rootOption, rootFunc);

		if (conditionSurv) {
			eRoot = getColumn(vals, 0); // get root E values as a column
			// apply function on dRoot (and root is always low resolution)
			for (int i = 0; i < lambdas_l.length; i++) {
				double lambdaX = lambdas_l[i];
				// sanity check
				final double min_value = 1e-30;
				if (1-eRoot[i] < min_value)
					return Double.NEGATIVE_INFINITY;
				// element-wise division of d column
				double factor = 1.0 / (lambdaX * (1 - eRoot[i]) * (1 - eRoot[i]));
				for (int j = 0; j < stateCount; j++)
					dRoot[i][j] = dRoot[i][j] * factor ;
			}
		}

		double[][] rootProduct = getProduct(rootP, dRoot);

		double logProb = Math.log(getSum(rootProduct) * dx);

		return logProb;
	}

	private double getColumnSum(double[][] dRoot, int column) {
		double sum = 0.0;
		int nrow = dRoot.length;
		for (int i = 0; i < nrow; i++) {
			sum = sum + dRoot[i][column];
		}
		return sum;
	}

	private double[][] getDValues(double[][] vals) {
		// all columns except first column
		int nrow = vals.length;
		int ncol = vals[0].length;
		assert (ncol > 1);
		double[][] dValues = new double[nrow][ncol - 1];
		for (int i = 0; i < nrow; i++) {
			for (int j = 0; j < ncol - 1; j++) {
				dValues[i][j] = vals[i][j + 1];
			}
		}
		return dValues;
	}

	private double[][] getRootProb(double[][] dRoot, double[] x, int nx, int rootOption, LinkFn rootFunc) {
		double dx = x[1] - x[0];
		int numSubstBins = dRoot.length;
		int ntypes = dRoot[0].length;
		double[][] p = new double[numSubstBins][ntypes];

		if (rootOption == ROOT_FLAT) {
			for (int i = 0; i < numSubstBins; i++) {
				for (int j = 0; j < ntypes; j++) {
					p[i][j] = 1 / ((nx - 1) * ntypes * dx);
				}
			}
		} else if (rootOption == ROOT_OBS) {
			double factor = 1.0 / (getSum(dRoot) * dx);
			for (int i = 0; i < numSubstBins; i++) {
				for (int j = 0; j < ntypes; j++) {
					p[i][j] = dRoot[i][j] * factor;
				}
			}
		} else {
			double[] rootI = substitutionModel.getFrequencies(); // equilibrium freqs

			if (rootOption == ROOT_EQUI) { // check this
				for (int i = 0; i < numSubstBins; i++) {
					for (int j = 0; j < ntypes; j++) {
						p[i][j] = rootI[j] * dRoot[i][j] / (getColumnSum(dRoot, j) * dx); // mapply
					}
				}
			} else if (rootOption == ROOT_GIVEN) { // test this with an appropriate function
				double[] y = new double[x.length];
				if (rootFunc != null) {
					y = rootFunc.getY(x, y);
					for (int i = 0; i < numSubstBins; i++) {
						for (int j = 0; j < ntypes; j++) {
							p[i][j] = rootI[j] * y[i]; // mapply
						}
					}
				}
			}
		}

		return p;
	}

	private double[][] getProduct(double[][] array1, double[][] array2) {
		assert (array1.length == array2.length);
		assert (array1[0].length == array2[0].length);
		double[][] res = new double[array1.length][array1[0].length];
		for (int i = 0; i < res.length; i++) {
			for (int j = 0; j < res[0].length; j++) {
				res[i][j] = array1[i][j] * array2[i][j];
			}
		}
		return res;
	}

	private double getSum(double[][] array) {
		double res = 0.0;
		for (double[] element : array) {
			for (int j = 0; j < array[0].length; j++) {
				res = res + element[j];
			}
		}
		return res;
	}

	private double[] getColumn(double[][] values, int colIndex) {
		int rows = values.length;
		double[] column = new double[rows];
		if (colIndex == -1) {
			colIndex = values[0].length - 1;
		}
		for (int i = 0; i < rows; i++) {
			column[i] = values[i][colIndex];
		}
		return column;
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
		traverseFull(tree.getRoot());
		calcLogP();
		printLogP();
		return logP;
	}
	
	protected void printParams() {
		String newickstr = toNewick(tree.getRoot()) + ";";
		System.out.println(newickstr);
		System.out.println("tc = " + tc);
		printSiteModelParameters();
		tipModel.printParams();
		treeModel.printParams();
		lambdaFunc.printParams();
		muFunc.printParams();
		showRHASParams();
	}

	protected void printSiteCatLikes(double[] arr, int num_cats) {
		System.out.println("Site log-likelihoods for each category:");
		for (int cat = 0; cat < num_cats; cat++) {
			int k = cat;
			for (int p = 0; p < patterns; p++) {
				if (p > 0)
					System.out.print(",");
				System.out.print(arr[k]);
				k += num_cats;
			}
			System.out.println();
		}
	}
		
	protected void printLogP() {
		printParams();
		// printSiteCatLikes(patternLogLikelihoods, 1);
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
			for (int i = 0; i < patterns; i++) {
				logP += patternLogLikelihoods[i] * data.getPatternWeight(i);
			}
		}
	}

	@Override
	protected boolean requiresRecalculation() {
		// always recalculate
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

	protected boolean isLowResolution(Node node) {
		if (node.isLeaf()) {
			return false; // leaf always uses high resolution
		} else if (node.isRoot() || node.getHeight() >= tc) {
			return true; // low resolution for root and the nodes on or above tc
		} else {
			return false; // high resolution for nodes lower than tc
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
	
	/**
	 * show the parameters associated with gamma distribution
	 */
	protected void showRHASParams() {
		System.out.print("Number of categories: " + numCategories);
		
		if (m_siteModel instanceof SiteModel) {
		    SiteModel siteModel = (SiteModel) m_siteModel;
		    RealParameter shapeParam = siteModel.shapeParameterInput.get();
		    if (shapeParam != null) {
		        double alpha = shapeParam.getValue();
		        System.out.print("; Gamma shape (alpha) = " + alpha);
		    }
		}
		
		if (categoryRates != null) {
			System.out.print("; Rates: ");
			for (int i = 0; i < categoryRates.length; i++) {
				if (i > 0)
					System.out.print(",");
				System.out.print(categoryRates[i]);
			}
		}
		if (categoryProps != null) {
			System.out.print("; Proportions: ");
			for (int i = 0; i < categoryProps.length; i++) {
				if (i > 0)
					System.out.print(",");
				System.out.print(categoryProps[i]);
			}
		}
		
		System.out.println();
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

	protected void show1DArray(double[] array, String desc) {
		System.out.println(desc);
		boolean error_found = false;
		int max_num_per_line = 30;
		for (int i = 0; i < array.length; i++) {
			if (i % max_num_per_line == 0)
				System.out.println();
			System.out.print("," + array[i]);
			if (Double.isNaN(array[i]))
				error_found = true;
		}
		System.out.println();
		if (error_found)
			System.exit(1);
	}
	
	protected void show2DArray(double[][] array, String desc) {
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
