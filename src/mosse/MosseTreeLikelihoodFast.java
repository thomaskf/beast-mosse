package mosse;

import beast.base.core.Description;
import beast.base.inference.Distribution;
import beast.base.core.Input;
import beast.base.inference.parameter.IntegerParameter;
import beast.base.inference.parameter.RealParameter;
import beast.base.core.Log;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.likelihood.TreeLikelihood;
import beast.base.evolution.sitemodel.SiteModel;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TraitSet;
import beast.base.evolution.tree.Tree;
import beast.base.evolution.tree.TreeInterface;
import org.jblas.DoubleMatrix;

import java.lang.UnsupportedOperationException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * @author Kylie Chen
 * @author Thomas Wong
 */

@Description("Mosse likelihood class calculates the probability of sequence and trait data on a tree")
public class MosseTreeLikelihoodFast extends TreeLikelihood {

    protected MosseLikelihoodCore mosseLikelihoodCore;
    // trait data
    final public Input<List<TraitSet>> traitListInput = new Input<>("traits", "list of traits", new ArrayList<>());
    // tip model, species diversification model and trait model
    final public Input<MosseTipLikelihood> tipModelInput = new Input<>("tipModel", "model of tip probabilities", Input.Validate.REQUIRED);
    final public Input<Distribution> treeModelInput = new Input<>("treeModel", "species diversification model", Input.Validate.REQUIRED);

    // substitution rate parameters
    final public Input<RealParameter> startSubsRateInput = new Input<>("startSubsRate", "lower range for substitution rate", Input.Validate.REQUIRED);
    final public Input<IntegerParameter> numRateBinsInput = new Input<>("numRateBins", "number of bins for substitution rate", Input.Validate.REQUIRED);

    // lambda and mu functions
    final public Input<LinkFn> lambdaFuncInput = new Input<>("lambdaFunc", "function for birth rate lambda", Input.Validate.REQUIRED);
    final public Input<LinkFn> muFuncInput = new Input<>("muFunc", "function for death rate mu", Input.Validate.REQUIRED);

    // root options
    final public Input<LinkFn> rootFuncInput = new Input<>("rootFunc", "function for root", Input.Validate.OPTIONAL);

    final public Input<IntegerParameter> rootOptionInput = new Input<>("rootOption", "option for root calculation", Input.Validate.OPTIONAL);

    final public Input<IntegerParameter> resolutionOptionInput = new Input<>("resolution", "resolution scale factor", new IntegerParameter("1"), Input.Validate.OPTIONAL);

    // root options
    final public int ROOT_FLAT = 1;
    final public int ROOT_OBS = 2;
    final public int ROOT_EQUI = 3; // check test case for NAN
    final public int ROOT_GIVEN = 4; // check test case for NAN

    protected int resolution;

    protected int rootOption = 3; // default to ROOT_EQUI

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
    protected int numEntries_h; // number of non-zero elements in lambdas
    protected int padLeft_h;
    protected int padRight_h;
    protected double[] lambdas_h;
    protected double[] mus_h;
    protected double[] flatTransitionMatrices_h;
    
    // variables for low resolution
    protected int numRateBins_l;
    protected double dx_l;
    protected double startSubsRate_l;
    protected int numEntries_l; // number of non-zero elements in lambdas
    protected int padLeft_l;
    protected int padRight_l;
    protected double[] lambdas_l;
    protected double[] mus_l;
    protected double[] flatTransitionMatrices_l;
    
    protected int numRateBins_max; // max{numRateBins_h,numRateBins_l}

    private boolean updateTips = true;

    private boolean updateSiteModel = true;
    
    // store the log compensations for each nodes
    private double[] log_compensates;

    // Force full recomputation on the first call to calculateLogP()
    private boolean firstEval = true;
    
    @Override
    public void initAndValidate() {
        traits = traitListInput.get();
        tipModel = tipModelInput.get();
        alignment = dataInput.get();
        treeModel = (MosseDistribution) treeModelInput.get();

        if (resolutionOptionInput.get() != null) {
            resolution = resolutionOptionInput.get().getValue();
        } else {
            resolution = resolutionOptionInput.defaultValue.getValue();
        }
        
        // high resolution
        numRateBins_h = numRateBinsInput.get().getValue() * resolution;
        dx_h = treeModel.dxInput.get().getValue();
        startSubsRate_h = startSubsRateInput.get().getValue();
        // low resolution
        numRateBins_l = numRateBinsInput.get().getValue();
        dx_l = treeModel.dxInput.get().getValue() * resolution;
        startSubsRate_l = startSubsRateInput.get().getValue() + treeModel.dxInput.get().getValue() * (resolution - 1);

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
        if (alignment.getTaxonCount() != treeInput.get().getLeafNodeCount()) {
            String message = String.format(
                    "The number of leaves in tree (%d) does not match the number of sequences (%d).",
                    treeInput.get().getLeafNodeCount(),
                    alignment.getTaxonCount());
            throw new IllegalArgumentException(message);
        } else if (numRateBinsInput.get().getValue() <= 0) {
            throw new IllegalArgumentException("numRateBins input must be a positive integer");
        } else if (!(siteModelInput.get() instanceof SiteModel.Base)) {
            throw new IllegalArgumentException("siteModel input should be of type SiteModel.Base");
        } else if (branchRateModelInput.get() != null) {
            System.err.println("Ignoring clock model " + branchRateModelInput.get().getID());
        }

        beagle = null;

        tc = treeInput.get().getRoot().getHeight() / 10.0;
        int nodeCount = treeInput.get().getNodeCount();
        m_siteModel = (SiteModel.Base) siteModelInput.get();
        m_siteModel.setDataType(alignment.getDataType());
        substitutionModel = m_siteModel.substModelInput.get();

        // remove requirement for clock model
        branchRateModelInput.setRule(Input.Validate.OPTIONAL);
        branchRateModel = null;
        m_branchLengths = new double[nodeCount];
        storedBranchLengths = new double[nodeCount];

        int stateCount = alignment.getMaxStateCount();
        int patterns = alignment.getPatternCount();

        // set likelihood core number of states and number of rate bins
        mosseLikelihoodCore = new MosseLikelihoodCore(stateCount, numRateBins_max);
        
        // compute the padLeft, padRight, and numEntries for low resolution
        boolean lowResolution = true;
        padLeft_l = treeModel.getPadLeft(lowResolution);
        padRight_l = treeModel.getPadRight(lowResolution);
        numEntries_l = numRateBins_l - padLeft_l - padRight_l - 1;

        // compute the padLeft, padRight, and numEntries for high resolution
        lowResolution = false;
        padLeft_h = treeModel.getPadLeft(lowResolution);
        padRight_h = treeModel.getPadRight(lowResolution);
        numEntries_h = numRateBins_h - padLeft_h - padRight_h - 1;

        // get lambdas and mus
        lambdas_h = new double[numEntries_h];
        lambdas_l = new double[numEntries_l];
        mus_h = new double[numEntries_h];
        mus_l = new double[numEntries_l];
        double[] x_h = getSubstitutionRates(numEntries_h, startSubsRate_h, dx_h, padLeft_h); // substitution rates
        double[] x_l = getSubstitutionRates(numEntries_l, startSubsRate_l, dx_l, padLeft_l); // substitution rates
        lambdaFunc.getY(x_h, lambdas_h, true);
        lambdaFunc.getY(x_l, lambdas_l, true);
        muFunc.getY(x_h, mus_h, true);
        muFunc.getY(x_l, mus_l, true);
        
        String className = getClass().getSimpleName();
        Alignment alignment = dataInput.get();

        // logging likelihood class
        Log.info.println(className + "(" + getID() + ") uses " + mosseLikelihoodCore.getClass().getSimpleName());
        Log.info.println("  " + alignment.toString(true));

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

        if (alignment.isAscertained) {
            useAscertainedSitePatterns = true;
        }
        
        // root partial array always use low resolution
        m_fRootPartials = new double[patterns * stateCount * numRateBins_l];
        
        // initialize the log-compensation array
        System.out.println("Number of nodes: " + treeInput.get().getNodeCount());
        log_compensates = new double[treeInput.get().getNodeCount()];
        Arrays.fill(log_compensates, 0.0);
        
        // optional but consistent: force a full recompute on first call
        hasDirt = Tree.IS_DIRTY;
        updateSiteModel = true;
        updateTips = true;    }

    /**
     * set leaf partials using tip GLM likelihood model *
     */
    @Override
    protected void setPartials(Node node, int patternCount) {
        if (node.isLeaf()) {
        	// for leaf, always use high resolution
            Alignment data = dataInput.get();
            int states = data.getDataType().getStateCount();
            double[] traitValues = getTraits(node);
            double[] partials = new double[patternCount * (states + 1) * numRateBins_max];
            int k = 0;
            int taxonIndex = data.getTaxonIndex(node.getID());
            for (int patternIndex = 0; patternIndex < patternCount; patternIndex++) {
            	k = patternIndex * (states + 1) * numRateBins_max;
            	double subsInterval = startSubsRate_h;
            	// question: should the following line be outside the loop?
                double[] tipLikelihoods = tipModel.getTipLikelihoods(traitValues, numEntries_h, startSubsRate_h + padLeft_h * subsInterval, subsInterval);
                int stateCount = data.getPattern(taxonIndex, patternIndex);
                boolean[] stateSet = data.getStateSet(stateCount);
                // E initial values are zero
                for (int i = 0; i < numRateBins_h; i++) {
                    partials[k++] = 0.0;
                }
                // D initial values
                for (int state = 0; state < states; state++) {
                    if (stateSet[state]) {
                        // set likelihoods for nucleotide in data
                        for (int i = 0; i < numRateBins_h; i++) {
                            if (i < numEntries_h) {
                                partials[k++] = tipLikelihoods[i];
                            } else {
                                partials[k++] = 0.0; // within padding
                            }
                        }
                    } else {
                        // otherwise set likelihoods to zero
                        for (int i = 0; i < numRateBins_h; i++) {
                            partials[k++] = 0.0;
                        }
                    }
                }
            }
            mosseLikelihoodCore.setNodePartials(node.getNr(), partials);

        } else {
            setPartials(node.getLeft(), patternCount);
            setPartials(node.getRight(), patternCount);
        }
    }

    /**
     * set leaf states (not applicable for this class, use setPartials instead)
     */
    @Override
    protected void setStates(Node node, int patternCount) {
        throw new UnsupportedOperationException();
    }

    protected void initCore() {
        final int nodeCount = treeInput.get().getNodeCount();
        mosseLikelihoodCore.initialize(
                nodeCount,
                dataInput.get().getPatternCount(),
                m_siteModel.getCategoryCount(),
                true,
                m_useAmbiguities.get()
        );

        // number of internal nodes and external nodes for a rooted tree
        final int extNodeCount = nodeCount / 2 + 1;
        final int intNodeCount = nodeCount / 2;

        // set up tip partials
        setPartials(treeInput.get().getRoot(), dataInput.get().getPatternCount());

        hasDirt = Tree.IS_FILTHY;
        for (int i = 0; i < intNodeCount; i++) {
            mosseLikelihoodCore.createNodePartials(extNodeCount + i);
        }
    }

    /**
     * Compute the normalization (or log compensation)
     * @param vars -- array to be normalized
     * @param nx -- number of entries for each row
     * @param dx
     * @return log of scaling (i.e. lq)
     */
    public double normalization(double[] vars, int nx, double dx) {
        // normalize the values of vars
        // ignore the first nx entries (i.e. first row)
        double vsum = 0.0;
        for (int i = nx; i < vars.length; i++)
        	vsum += vars[i];
        vsum *= dx;
        for (int i = nx; i < vars.length; i++)
        	vars[i] /= vsum;
        return Math.log(vsum);
    }
    
    /**
     * create a flatTransitionMatrice
     */
    private double[] createFlatTransitionMatrice(Node node, boolean lowResolution) {
        double deltaT = treeModel.dtInput.get().getValue(); // 0.001; // dt
        int numStates = dataInput.get().getDataType().getStateCount();
        double rate; // dx
        int numEntries;
        int padLeft;
        if (lowResolution) {
        	rate = treeModel.dxInput.get().getValue() * resolution;
        	numEntries = numEntries_l;
        	padLeft = padLeft_l;
        } else {
        	rate = treeModel.dxInput.get().getValue();
            numEntries = numEntries_h;
            padLeft = padLeft_h;
        }
        double[] transitionMatrix = new double[numStates * numStates];
        substitutionModel.getTransitionProbabilities(node, deltaT, 0, rate, transitionMatrix); // startTime (i.e. deltaT) is greater than endTime (i.e. 0)
        double[][] transitionMatrices = new double[numEntries][transitionMatrix.length];
        double[] prevMatrix = new double[numStates * numStates];
        prevMatrix = transitionMatrix;

        // skip the first padLeft entries
        for (int i = 0; i < padLeft; i++) {
            // multiplication of matrix
            DoubleMatrix matrixOne = new DoubleMatrix(numStates, numStates, transitionMatrix);
            DoubleMatrix matrixTwo = new DoubleMatrix(numStates, numStates, prevMatrix);
            DoubleMatrix result = matrixOne.mmul(matrixTwo);
            prevMatrix = result.toArray();
        }
        int l = 0;
        // update transitionMatrices
        transitionMatrices[l] = prevMatrix;
        for (int i = padLeft; i < numEntries + padLeft - 1; i++) {
            // multiplication of matrix
            DoubleMatrix matrixOne = new DoubleMatrix(numStates, numStates, transitionMatrix);
            DoubleMatrix matrixTwo = new DoubleMatrix(numStates, numStates, transitionMatrices[l]);
            DoubleMatrix result = matrixOne.mmul(matrixTwo);
            l++;
            transitionMatrices[l] = result.toArray();
        }

        return Arrays.stream(transitionMatrices)
                .flatMapToDouble(Arrays::stream)
                .toArray();
    }
    
    /**
     * compute likelihoods for single branch
     * return the log compensation
     */
    private double computeSingleBranchLikelihood(Node node, Node child, double[] partials) {
        int numPlan = 5; // dimensions
        double logP = 0.0; // for log-compensation

        // normalization (log compensation) on the input partials
        if (child.getHeight() < tc || child.isLeaf()) {
        	// high resolution
        	logP += normalization(partials, numRateBins_h, dx_h);
        } else {
        	// low resolution
        	logP += normalization(partials, numRateBins_l, dx_l);
        }
        
        if (node.getHeight() < tc && !node.isRoot()) {
    		// high resolution for the whole branch
    		boolean lowResolution = false;
    		double branchTime = node.getHeight() - child.getHeight();
    		if (flatTransitionMatrices_h == null)
    			flatTransitionMatrices_h = createFlatTransitionMatrice(node, lowResolution);
    		treeModel.calculateBranchLogP(branchTime, partials, lambdas_h, mus_h, flatTransitionMatrices_h, partials, lowResolution);
    	} else if (child.getHeight() >= tc && !child.isLeaf()) {
    		// low resolution for the whole branch
    		boolean lowResolution = true;
    		double branchTime = node.getHeight() - child.getHeight();
    		if (flatTransitionMatrices_l == null)
    			flatTransitionMatrices_l = createFlatTransitionMatrice(node, lowResolution);
    		treeModel.calculateBranchLogP(branchTime, partials, lambdas_l, mus_l, flatTransitionMatrices_l, partials, lowResolution);
    	} else {
    		// combination of high and low resolutions along the branch
    		// high resolutions between child.getHight() and t_mid
    		double branchTime;
    		double t_mid = tc;
    		if (node.getHeight() < tc)
    			t_mid = node.getHeight();
    		branchTime = t_mid - child.getHeight();
    		boolean lowResolution = false;
    		if (flatTransitionMatrices_h == null)
    			flatTransitionMatrices_h = createFlatTransitionMatrice(node, lowResolution);
    		treeModel.calculateBranchLogP(branchTime, partials, lambdas_h, mus_h, flatTransitionMatrices_h, partials, lowResolution);
    		// selecting the corresponding entries in the partialMiddle as the input for low resolution
    		for (int i = 0; i < numPlan; i++) {
        		int k = 0;
        		int s_h = i * numRateBins_h;
        		int s_l = i * numRateBins_l;
    			for (int j = resolution-1; j < numRateBins_h; j+= resolution) {
    				if (k < numRateBins_l) {
	    				partials[s_l + k] = partials[s_h + j];
	    				k++;
    				} else {
    					break;
    				}
    			}
    			while (k < numRateBins_l) {
    				partials[s_l + k] = 0.0;
    				k++;
    			}
    		}
    		// reduce the size of partials to "numPlan * numRateBins_l"
    		int partial2_size = numPlan * numRateBins_l;
    		double[] partials2 = Arrays.copyOf(partials, partial2_size);

        	// normalization (log compensation) on the input partials (low resolution)
        	logP += normalization(partials2, numRateBins_l, dx_l);
    		
    		// then low resolution between t_mid and node.getHeight()
    		lowResolution = true;
    		branchTime = node.getHeight() - t_mid;
    		if (branchTime > 0.0) {
	    		if (flatTransitionMatrices_l == null)
	    			flatTransitionMatrices_l = createFlatTransitionMatrice(node, lowResolution);
	    		treeModel.calculateBranchLogP(branchTime, partials2, lambdas_l, mus_l, flatTransitionMatrices_l, partials2, lowResolution);
    		}

            System.arraycopy(partials2, 0, partials, 0, partial2_size);
    		
    	}
        
        // normalization (log compensation) on the output partials
        if (node.getHeight() >= tc || node.isRoot()) {
        	// low resolution
        	logP += normalization(partials, numRateBins_l, dx_l);
        } else {
        	// high resolution
        	logP += normalization(partials, numRateBins_h, dx_h);
        }
        
    	return logP;
    }

    /**
     * traverse tree with optimized caching
     * @param node tree node
     * @return update flag
     */
    @Override
    protected int traverse(final Node node) {
        int numPlan = 5; // dimensions
        int numPattern = alignment.getPatternCount();
        
        int update = (node.isDirty() | hasDirt);
        final int nodeIndex = node.getNr();
        
        // If nothing changed at or below this node and no global dirt,
        // we can stop immediately – cached partials are still valid.
        if (update == Tree.IS_CLEAN) {
            return Tree.IS_CLEAN;
        }

        // ---- leaf case ----
        if (node.isLeaf()) {
        	if (updateTips) {
        		// update tips from GLM if node is a leaf
        		setPartials(node, numPattern);
        		updateTips = false;
        	}
        	
        	if (updateSiteModel) {
        		// update site transition matrices
        		flatTransitionMatrices_h = null;
        		flatTransitionMatrices_l = null;
                updateSiteModel = false;
        	}
        	
        	// Noting else to do for a leaf
        	return update;
        }

        // ---- internal node case ----
        final Node child1 = node.getLeft();
        final int update1 = traverse(child1);

        final Node child2 = node.getRight();
        final int update2 = traverse(child2);

        // if either child was updated, we must recompute this node's partials
        if (update1 != Tree.IS_CLEAN || update2 != Tree.IS_CLEAN) {
        	
        	// System.out.println("[F] Compute " + nodeIndex + "'s partial likelihood value");
            final int childNum1 = child1.getNr();
            final int childNum2 = child2.getNr();
            
            mosseLikelihoodCore.setNodePartialsForUpdate(nodeIndex);

            flatTransitionMatrices_l = null;
            flatTransitionMatrices_h = null;

            double[] patternPartialsLeft = new double[numPattern * numPlan * numRateBins_max];
            double[] patternPartialsRight = new double[numPattern * numPlan * numRateBins_max];
            double[] partialsAllPatterns = new double[numPattern * numPlan * numRateBins_max];
            
            // get the get compensation values from left and right children
            double logPNode = 0.0;
            logPNode += log_compensates[childNum1];
            logPNode += log_compensates[childNum2];

            // get child node partials all patterns
            mosseLikelihoodCore.getNodePartials(childNum1, patternPartialsLeft);
            mosseLikelihoodCore.getNodePartials(childNum2, patternPartialsRight);
            
            // numRateBins, numEntries, lambdas
            int numRateBins_left = numRateBins_h;
            int numRateBins_right = numRateBins_h;
            int numRateBins_curr = numRateBins_h;
            int numEntries_curr = numEntries_h;
            double[] lambdas_curr = lambdas_h;
            if (child1.getHeight() >= tc && !child1.isLeaf()) {
            	numRateBins_left = numRateBins_l;
            }
            if (child2.getHeight() >= tc && !child2.isLeaf()) {
            	numRateBins_right = numRateBins_l;
            }
            if (node.getHeight() >= tc || node.isRoot()) {
            	numRateBins_curr = numRateBins_l;
            	numEntries_curr = numEntries_l;
            	lambdas_curr = lambdas_l;
            }
            
            for (int pattern = 0; pattern < numPattern; pattern++) {
                // partial for single pattern
                int startPos = pattern * numPlan * numRateBins_max;
                int partialSizeLeft = numPlan * numRateBins_left;
                int partialSizeRight = numPlan * numRateBins_right;
                double[] partialsLeft = new double[partialSizeLeft];
                System.arraycopy(patternPartialsLeft, startPos, partialsLeft, 0, partialSizeLeft);
                double[] partialsRight = new double[partialSizeRight];
                System.arraycopy(patternPartialsRight, startPos, partialsRight, 0, partialSizeRight);
                
                // propagate each child branch
                logPNode += computeSingleBranchLikelihood(node, child1, partialsLeft);
                logPNode += computeSingleBranchLikelihood(node, child2, partialsRight);

                int k = 0;
                for (int i = 0; i < numPlan; i++) {
                    for (int j = 0; j < numRateBins_curr; j++) {
                        if (i == 0) {
                            // E is topology independent
                            partialsAllPatterns[startPos + k] = partialsLeft[k];
                        } else {
                            if (j < numEntries_curr) {
                                // non padded elements
                                // D_left * D_right * lambda(x)
                                double lambdaX = lambdas_curr[j]; // birth rate at substitution rate x
                                partialsAllPatterns[startPos + k] = partialsLeft[k] * partialsRight[k] * lambdaX;
                            } else {
                                // padded elements
                                partialsAllPatterns[startPos +  k] = 0.0;
                            }
                        }
                        k++;
                    }
                }
            }
            
	        // set node partials
	        mosseLikelihoodCore.setNodePartials(nodeIndex, partialsAllPatterns);
	        
	        // set the log compensation value
	        log_compensates[nodeIndex] = logPNode;
	        
            if (node.isRoot()) {
	        	// root is always low resolution
	            for (int pattern = 0; pattern < numPattern; pattern++) {
	                int startPos = pattern * numPlan * numRateBins_max;
	                int partialSizeRoot = numPlan * numRateBins_l;
	                double[] partials =  new double[partialSizeRoot];
	                System.arraycopy(partialsAllPatterns, startPos, partials, 0, partialSizeRoot);
	
	                boolean conditionSurv = true;
	                double patternLogLikelihood = makeRootFuncMosse(numRateBins_l, dx_l, resolution, partials, conditionSurv);
	                patternLogLikelihoods[pattern] = patternLogLikelihood;
	            }
	            
	            update |= (update1 | update2);
            }
        }

        return update;
    }
    
    /**
     *
     * @param nx number of bins for substitution rate
     * @param dx distance between xs
     * @param r for resolution scale factor
     * @param result root node result matrix of D and E values
     * @param conditionSurv whether to condition on survival
     * @return log probability for root
     */
    private double makeRootFuncMosse(int nx, double dx, int r, double[] result, boolean conditionSurv) {
        int ntypes = 4;

        double[][] vals = new double[nx][ntypes+1];
        int count = 0;
        for (int j = 0; j < ntypes + 1; j++) { // nucleotide types columns
            for (int i = 0; i < nx; i++) { // nx rows
                vals[i][j] = result[count];
                count++;
            }
        }

        double[][] dRoot = getDValues(vals); // get root D values in last column
        
        double[] eRoot = null;

        double[] x = getSubstitutionRates(numEntries_l, startSubsRate_l, dx_l, padLeft_l);
        // root options
        double[][] rootP = getRootProb(dRoot, x, nx, rootOption, rootFunc);

        if (conditionSurv) {
            eRoot = getColumn(vals, 0); // get root E values as a column
            // apply function on dRoot (and root is always low resolution)
            for (int j = 0; j < ntypes; j++) {
            	for (int i = 0; i < lambdas_l.length; i++) {
                    double lambdaX = lambdas_l[i];
                    // element-wise division of d column
                    dRoot[i][j] = dRoot[i][j] / (lambdaX * (1 - eRoot[i]) * (1 - eRoot[i]));
                }
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
        assert(ncol > 1);
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
        } else if (rootOption == ROOT_OBS)  {
            for (int i = 0; i < numSubstBins; i++) {
                for (int j = 0; j < ntypes; j++) {
                    p[i][j] = dRoot[i][j] / (getSum(dRoot) * dx);
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
            } else if (rootOption == ROOT_GIVEN){ // test this with an appropriate function
                double[] y = new double[x.length];
                if (rootFunc != null) {
                    y = rootFunc.getY(x, y, true);
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
        assert(array1.length == array2.length);
        assert(array1[0].length == array2[0].length);
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
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[0].length; j++) {
                res = res + array[i][j];
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
        for(int i = 1; i < numElements; i++) {
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
        final TreeInterface tree = treeInput.get();
        final int rootIndex = tree.getRoot().getNr();
        if (requiresRecalculation()) {
	    	if (traverse(tree.getRoot()) != Tree.IS_CLEAN) {
	    		calcLogP();
	    	}
        }
        double ans = logP + log_compensates[rootIndex];
        System.out.println("logP = " + ans);
        return ans;
    }

    protected void calcLogP() {
        logP = 0.0;
        if (useAscertainedSitePatterns) {
            final double ascertainmentCorrection = alignment.getAscertainmentCorrection(patternLogLikelihoods);
            for (int i = 0; i < alignment.getPatternCount(); i++) {
                logP += (patternLogLikelihoods[i] - ascertainmentCorrection) * alignment.getPatternWeight(i);
            }
        } else {
            for (int i = 0; i < alignment.getPatternCount(); i++) {
                logP += patternLogLikelihoods[i] * alignment.getPatternWeight(i);
            }
        }
    }
    
    @Override
    protected boolean requiresRecalculation() {
        // Reset global flags
        hasDirt = Tree.IS_CLEAN;
        updateTips = false;
        updateSiteModel = false;

        // first evaluation: force a full recomputation
        if (firstEval) {
        	firstEval = false;
        	hasDirt = Tree.IS_DIRTY;
        	updateSiteModel = true;
        	updateTips = true;
        	return true;
        }

        boolean recalc = false;
        
        // If site model changed, we must recompute all partials
        if (m_siteModel.isDirtyCalculation()) {
            hasDirt = Tree.IS_DIRTY;
            updateSiteModel = true;
            recalc = true;
        }

        // If tip model changed, we must recompute all leaf partials
        if (tipModel.isDirtyCalculation()) {
            hasDirt = Tree.IS_DIRTY;
            updateTips = true;
            recalc = true;
        }

        // If nothing global changed, check whether the tree itself has dirty nodes
        if (!recalc && treeInput.get().somethingIsDirty()) {
            recalc = true;
        }

        return recalc;
    }    
}
