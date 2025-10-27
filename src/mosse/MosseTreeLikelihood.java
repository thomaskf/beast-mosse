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
import org.jblas.MatrixFunctions;

import java.lang.UnsupportedOperationException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * @author Kylie Chen
 */

@Description("Mosse likelihood class calculates the probability of sequence and trait data on a tree")
public class MosseTreeLikelihood extends TreeLikelihood {

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


//    final public int SUBST_NUM_STATES = 4; // for testing

    // root options
    final public int ROOT_FLAT = 1;
    final public int ROOT_OBS = 2;
    final public int ROOT_EQUI = 3; // check test case for NAN
    final public int ROOT_GIVEN = 4; // check test case for NAN

    protected int resolution;

    protected int rootOption = 3; // default to ROOT_EQUI
    // protected int rootOption = 2; // default to ROOT_OBS

    protected LinkFn rootFunc;
    protected LinkFn lambdaFunc;
    protected LinkFn muFunc;
    protected List<TraitSet> traits;
    protected MosseTipLikelihood tipModel;
    protected MosseDistribution treeModel;
    protected double tc; // time < tc for high resolution, while time >= tc for low resolution  

    // variables for high resolution
    protected double startSubsRate_h;
    protected int numRateBins_h;
    protected int numEntries_h; // number of non-zero elements in lambdas
    protected int padLeft_h;
    protected int padRight_h;
    protected double[] lambdas_h;
    protected double[] mus_h;
    double[] flatTransitionMatrices_h;
    
    // variables for low resolution
    protected double startSubsRate_l;
    protected int numRateBins_l;
    protected int numEntries_l; // number of non-zero elements in lambdas
    protected int padLeft_l;
    protected int padRight_l;
    protected double[] lambdas_l;
    protected double[] mus_l;
    double[] flatTransitionMatrices_l;
    
    protected int numRateBins_max; // max{numRateBins_h,numRateBins_l}

    @Override
    public void initAndValidate() {
        traits = traitListInput.get();
        tipModel = tipModelInput.get();
        treeModel = (MosseDistribution) treeModelInput.get();

        if (resolutionOptionInput.get() != null) {
            resolution = resolutionOptionInput.get().getValue();
        } else {
            resolution = resolutionOptionInput.defaultValue.getValue();
        }

        // high resolution
        startSubsRate_h = startSubsRateInput.get().getValue();
        numRateBins_h = numRateBinsInput.get().getValue() * resolution;

        // low resolution
        startSubsRate_l = startSubsRateInput.get().getValue() * resolution;
        numRateBins_l = numRateBinsInput.get().getValue();

        // compute the maximum value of numRateBins
        if (numRateBins_h > numRateBins_l)
            numRateBins_max = numRateBins_h;
        else
            numRateBins_max = numRateBins_l;
        
        lambdaFunc = lambdaFuncInput.get();
        muFunc = muFuncInput.get();

        if (rootOptionInput.get() != null) {
            rootOption = rootOptionInput.get().getValue();
        }
        if (rootFuncInput.get() != null) {
            rootFunc = rootFuncInput.get();
        }

        // input checking
        if (dataInput.get().getTaxonCount() != treeInput.get().getLeafNodeCount()) {
            String message = String.format(
                    "The number of leaves in tree (%d) does not match the number of sequences (%d).",
                    treeInput.get().getLeafNodeCount(),
                    dataInput.get().getTaxonCount());
            throw new IllegalArgumentException(message);
        } else if (numRateBinsInput.get().getValue() <= 0) {
            throw new IllegalArgumentException("numRateBins input must be a positive integer");
        } else if (!(siteModelInput.get() instanceof SiteModel.Base)) {
            throw new IllegalArgumentException("siteModel input should be of type SiteModel.Base");
        } else if (branchRateModelInput.get() != null) {
            System.err.println("Ignoring clock model " + branchRateModelInput.get().getID());
        }

        beagle = null;

        tc = treeInput.get().getRoot().getHeight() / 2.0; // getTreeLength(treeInput.get());
        int nodeCount = treeInput.get().getNodeCount();
        m_siteModel = (SiteModel.Base) siteModelInput.get();
        m_siteModel.setDataType(dataInput.get().getDataType());
        substitutionModel = m_siteModel.substModelInput.get();

        // remove requirement for clock model
        branchRateModelInput.setRule(Input.Validate.OPTIONAL);
        branchRateModel = null;
        m_branchLengths = new double[nodeCount];
        storedBranchLengths = new double[nodeCount];

        int stateCount = dataInput.get().getMaxStateCount();
        int patterns = dataInput.get().getPatternCount();

        // set likelihood core number of states and number of rate bins
        mosseLikelihoodCore = new MosseLikelihoodCore(stateCount, numRateBins_max);
        
        boolean lowResolution = true;
        padLeft_l = treeModel.getPadLeft(lowResolution);
        padRight_l = treeModel.getPadRight(lowResolution);
        numEntries_l = numRateBins_l - padLeft_l - padRight_l - 1;
        lowResolution = false;
        padLeft_h = treeModel.getPadLeft(lowResolution);
        padRight_h = treeModel.getPadRight(lowResolution);
        numEntries_h = numRateBins_h - padLeft_h - padRight_h - 1;

        // show some parameter values
        System.out.println("startSubsRate_l = " + startSubsRate_l);
        System.out.println("numRateBins_l = " + numRateBins_l);
        System.out.println("padLeft_l = " + padLeft_l);
        System.out.println("padRight_l = " + padRight_l);
        System.out.println("numEntries_l = " + numEntries_l);
        
        System.out.println("startSubsRate_h = " + startSubsRate_h);
        System.out.println("numRateBins_h = " + numRateBins_h);
        System.out.println("padLeft_h = " + padLeft_h);
        System.out.println("padRight_h = " + padRight_h);
        System.out.println("numEntries_h = " + numEntries_h);

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

        if (dataInput.get().isAscertained) {
            useAscertainedSitePatterns = true;
        }
        
        // root partial array always use low resolution
        m_fRootPartials = new double[patterns * stateCount * numRateBins_l];
    }

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

//    protected double getTreeLength(TreeInterface tree) {
//        double total = 0.0;
//        int nNodes = tree.getNodeCount();
//
//        for (int i = 0; i < nNodes; i++) {
//            Node node = tree.getNode(i);
//            if (!node.isRoot()) {
//                double parentHeight = node.getParent().getHeight();
//                double nodeHeight = node.getHeight();
//                total += parentHeight - nodeHeight;
//            }
//        }
//        return total;
//    }
    
    protected void initCore() {
        final int nodeCount = treeInput.get().getNodeCount();
        mosseLikelihoodCore.initialize(
                nodeCount,
                dataInput.get().getPatternCount(),
                m_siteModel.getCategoryCount(),
                true,
                m_useAmbiguities.get()
        );

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
     * calculate log P without caching (for testing)
     * @return log P
     */
    public double calculateLogPFull() {
        final TreeInterface tree = treeInput.get();
        traverseFull(tree.getRoot());
        return logP;
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
        double rate = treeModel.dxInput.get().getValue(); // dx
        int numEntries = numEntries_h;
        int padLeft = padLeft_h;
        if (lowResolution) {
        	rate = rate * resolution;
        	numEntries = numEntries_l;
        	padLeft = padLeft_l;
        }
        double[] transitionMatrix = new double[numStates * numStates];
        substitutionModel.getTransitionProbabilities(node, deltaT, 0, rate, transitionMatrix); // startTime is greater than endTime
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
     */
    private void computeSingleBranchLikelihood(Node node, Node child, double[] partials) {
        int numPlan = 5; // dimensions
    	if (node.getHeight() <= tc) {
    		// high resolution for the whole branch
    		boolean lowResolution = false;
    		double branchTime = node.getHeight() - child.getHeight();
    		if (flatTransitionMatrices_h == null)
    			flatTransitionMatrices_h = createFlatTransitionMatrice(node, lowResolution);
    		treeModel.calculateBranchLogP(branchTime, partials, lambdas_h, mus_h, flatTransitionMatrices_h, partials, lowResolution);
    	} else if (child.getHeight() >= tc) {
    		// low resolution for the whole branch
    		boolean lowResolution = true;
    		double branchTime = node.getHeight() - child.getHeight();
    		if (flatTransitionMatrices_l == null)
    			flatTransitionMatrices_l = createFlatTransitionMatrice(node, lowResolution);
    		treeModel.calculateBranchLogP(branchTime, partials, lambdas_l, mus_l, flatTransitionMatrices_l, partials, lowResolution);
    	} else {
    		// high resolutions between child.getHight() and tc
    		double branchTime = tc - child.getHeight();
    		boolean lowResolution = false;
    		double[] partialsMiddle = new double[numPlan * numRateBins_max];
    		if (flatTransitionMatrices_h == null)
    			flatTransitionMatrices_h = createFlatTransitionMatrice(node, lowResolution);
    		treeModel.calculateBranchLogP(branchTime, partials, lambdas_h, mus_h, flatTransitionMatrices_h, partialsMiddle, lowResolution);
    		// selecting the corresponding entries in the partialMiddle as the input for low resolution
    		for (int i = 0; i < numPlan; i++) {
        		int k = 0;
        		int s_h = i * numRateBins_h;
        		int s_l = i * numRateBins_l;
    			for (int j = resolution-1; j < numRateBins_h && k < numRateBins_l; j+= resolution) {
    				partialsMiddle[s_l + k] = partialsMiddle[s_h + j];
    				k++;
    			}
    		}
    		// then low resolution between tc and node.getHeight()
    		lowResolution = true;
    		branchTime = node.getHeight() - tc;
    		if (flatTransitionMatrices_l == null)
    			flatTransitionMatrices_l = createFlatTransitionMatrice(node, lowResolution);
    		treeModel.calculateBranchLogP(branchTime, partialsMiddle, lambdas_l, mus_l, flatTransitionMatrices_l, partials, lowResolution);
    	}
    }

    /**
     * traverse the subtree rooted at node
     * @param node
     * @return log P
     */
    private double traverseFull(Node node) {
        int numPlan = 5; // dimensions
        double deltaT = treeModel.dtInput.get().getValue(); // 0.001; // dt
        // double rate = treeModel.dxInput.get().getValue(); // dx
        int numStates = dataInput.get().getDataType().getStateCount();
        int numPattern = dataInput.get().getPatternCount();
        double logPNode = 0.0;

//        double[] transitionMatrix = new double[numStates * numStates];
//        substitutionModel.getTransitionProbabilities(node, deltaT, 0, rate, transitionMatrix); // startTime is greater than endTime
//
//        double[][] transitionMatrices = new double[numEntries][transitionMatrix.length];
//        double[] prevMatrix = new double[numStates * numStates];
//        prevMatrix = transitionMatrix;
//
//        // skip the first padLeft entries
//        for (int i = 0; i < padLeft; i++) {
//            // multiplication of matrix
//            DoubleMatrix matrixOne = new DoubleMatrix(numStates, numStates, transitionMatrix);
//            DoubleMatrix matrixTwo = new DoubleMatrix(numStates, numStates, prevMatrix);
//            DoubleMatrix result = matrixOne.mmul(matrixTwo);
//            prevMatrix = result.toArray();
//        }
//        int l = 0;
//        // update transitionMatrices
//        transitionMatrices[l] = prevMatrix;
//        for (int i = padLeft; i < numEntries + padLeft - 1; i++) {
//            // multiplication of matrix
//            DoubleMatrix matrixOne = new DoubleMatrix(numStates, numStates, transitionMatrix);
//            DoubleMatrix matrixTwo = new DoubleMatrix(numStates, numStates, transitionMatrices[l]);
//            DoubleMatrix result = matrixOne.mmul(matrixTwo);
//            l++;
//            transitionMatrices[l] = result.toArray();
//        }


        // get lambdas and mus
        double[] x = getSubstitutionRates(numEntries); // substitution rates
        
        lambdas = new double[numEntries];
        mus = new double[numEntries];
        lambdas = lambdaFunc.getY(x, lambdas, true);
        
        mus = muFunc.getY(x, mus, true);

        if (node.isLeaf()) {
            // leaf node
            // leaf partials size = nrPatterns * (nrStates + 1) * numBins
            // columns = (4x D's for each nucleotide, 1x E), rows = bins for substitution rate
            setPartials(node, numPattern); // all site patterns

        } else {
            // internal node or the root node
        	double logPChild0, logPChild1;
            logPChild0 = traverseFull(node.getChild(0)); // left child
            logPChild1 = traverseFull(node.getChild(1)); // right child
            logPNode += logPChild0;
            logPNode += logPChild1;
            
            flatTransitionMatrices_l = null;
            flatTransitionMatrices_h = null;

            double[] patternPartialsLeft = new double[numPattern * numPlan * numRateBins_max];
            double[] patternPartialsRight = new double[numPattern * numPlan * numRateBins_max];
            double[] partialsAllPatterns = new double[numPattern * numPlan * numRateBins_max];

            // get child node partials all patterns
            mosseLikelihoodCore.getNodePartials(node.getLeft().getNr(), patternPartialsLeft);
            mosseLikelihoodCore.getNodePartials(node.getRight().getNr(), patternPartialsRight);
            int k = 0;
            for (int pattern = 0; pattern < numPattern; pattern++) {
                // partial for single pattern
                int partialSize = numPlan * numRateBins_max;
                int startPos = pattern * partialSize;
                double[] partialsLeft = new double[numPlan * numRateBins_max];
                System.arraycopy(patternPartialsLeft, startPos, partialsLeft, 0, partialSize);
                double[] partialsRight = new double[numPlan * numRateBins_max];
                System.arraycopy(patternPartialsRight, startPos, partialsRight, 0, partialSize);
                double[] partialsCombined = new double[partialsLeft.length];
                
                // normalization (log compensation) on the input partials
                double dx = treeModel.dxInput.get().getValue();
                double lc0_left = normalization(partialsLeft, numRateBins, dx);
                double lc0_right = normalization(partialsRight, numRateBins, dx);

                // propagate each child branch
                treeModel.calculateBranchLogP(branchTimeLeft, partialsLeft, lambdas, mus, flatTransitionMatrices, partialsLeft);
                treeModel.calculateBranchLogP(branchTimeRight, partialsRight, lambdas, mus, flatTransitionMatrices, partialsRight);

                // log compensation
                double lc_left = normalization(partialsLeft, numRateBins, dx);
                double lc_right = normalization(partialsRight, numRateBins, dx); 
                logPNode += lc_left + lc0_left;
                logPNode += lc_right + lc0_right;

                // assumes t less than tc threshold
                for (int i = 0; i < numPlan; i++) {
                    for (int j = 0; j < numRateBins; j++) {
                        int index = i * numRateBins + j;
                        if (i == 0) {
                            // E is topology independent
                            partialsCombined[index] = partialsLeft[index]; // for testing
                            partialsAllPatterns[k] = partialsLeft[index];
                        } else {
                            if (j < numEntries) {
                                // non padded elements
                                // D_left * D_right * lambda(x)
                                double lambdaX = lambdas[j]; // birth rate at substitution rate x
                                partialsCombined[index] = partialsLeft[index] * partialsRight[index] * lambdaX; // for testing
                                partialsAllPatterns[k] = partialsLeft[index] * partialsRight[index] * lambdaX;
                            } else {
                                // padded elements
                                partialsCombined[index] = 0.0; // set to zero
                                partialsAllPatterns[k] = 0.0;
                            }
                        }
                        k++;
                    }
                }
            }
            
	        // set node partials
	        mosseLikelihoodCore.setNodePartials(node.getNr(), partialsAllPatterns);
	
	        if (node.isRoot()) {
	            for (int pattern = 0; pattern < numPattern; pattern++) {
	                int partialSize = numPlan * numRateBins;
	                int startPos = pattern * partialSize;
	                double[] partials =  new double[numPlan * numRateBins];
	                System.arraycopy(partialsAllPatterns, startPos, partials, 0, partialSize);
	
	                boolean conditionSurv = true;
	                // root calc for a single pattern
	                // boolean conditionSurv = false;
	                
	                double patternLogLikelihood = makeRootFuncMosse(numRateBins, rate, resolution, partials, conditionSurv);
	                logPNode += patternLogLikelihood * dataInput.get().getPatternWeight(pattern);
	            }
            }
        }
        logP = logPNode;

        return logPNode;
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
        double dxScaled = dx * r;
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

        double[] x = getSubstitutionRates(nx);
        // root options
        double[][] rootP = getRootProb(dRoot, x, nx, rootOption, rootFunc);

        if (conditionSurv) {
            eRoot = getColumn(vals, 0); // get root E values as a column
            // apply function on dRoot
            for (int j = 0; j < ntypes; j++) {
            	for (int i = 0; i < lambdas.length; i++) {
                    double lambdaX = lambdas[i];
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

    private double getSum(double[] array) {
        double res = 0.0;
        for (double i: array) {
            res = res + i;
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

    public double[] getFlatTransitionMatrices() {
        return flatTransitionMatrices;
    }

    private double[] getSubstitutionRates(int numElements) {
        double[] res = new double[numElements];
        double start = treeModel.dxInput.get().getValue(); 
        double interval = start; // use start rate as interval
        res[0] = start + interval * padLeft;
        for(int i = 1; i < numElements; i++) {
            res[i] = res[i - 1] + interval;
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
        traverseFull(tree.getRoot());
        return logP;
    }

    @Override
    protected boolean requiresRecalculation() {
        // always recalculate
        return true;
    }
}
