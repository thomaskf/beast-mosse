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

    protected int rootOption = 2; // default to ROOT_OBS

    protected LinkFn rootFunc;
    protected LinkFn lambdaFunc;
    protected LinkFn muFunc;
    protected List<TraitSet> traits;
    protected MosseTipLikelihood tipModel;
    protected MosseDistribution treeModel;
    protected double startSubsRate;
    protected int numRateBins;
    protected int numEntries; // number of non-zero elements in lambdas
    protected int padLeft;
    protected int padRight;
    protected boolean lowResolution;

    protected double[] lambdas;

    protected double[] mus;

    double[] flatTransitionMatrices;

    @Override
    public void initAndValidate() {
        traits = traitListInput.get();
        tipModel = tipModelInput.get();
        treeModel = (MosseDistribution) treeModelInput.get();

        startSubsRate = startSubsRateInput.get().getValue();
        numRateBins = numRateBinsInput.get().getValue();

        lambdaFunc = lambdaFuncInput.get();
        muFunc = muFuncInput.get();

        if (rootOptionInput.get() != null) {
            rootOption = rootOptionInput.get().getValue();
        }
        if (rootFuncInput.get() != null) {
            rootFunc = rootFuncInput.get();
        }
        if (resolutionOptionInput.get() != null) {
            resolution = resolutionOptionInput.get().getValue();
        } else {
            resolution = resolutionOptionInput.defaultValue.getValue();
        }

        // input checking
        if (dataInput.get().getTaxonCount() != treeInput.get().getLeafNodeCount()) {
            String message = String.format(
                    "The number of leaves in tree (%d) does not match the number of sequences (%d).",
                    treeInput.get().getLeafNodeCount(),
                    dataInput.get().getTaxonCount());
            throw new IllegalArgumentException(message);
        } else if (numRateBins <= 0) {
            throw new IllegalArgumentException("numRateBins input must be a positive integer");
        } else if (!(siteModelInput.get() instanceof SiteModel.Base)) {
            throw new IllegalArgumentException("siteModel input should be of type SiteModel.Base");
        } else if (branchRateModelInput.get() != null) {
            System.err.println("Ignoring clock model " + branchRateModelInput.get().getID());
        }

        beagle = null;

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
        lowResolution = false; 
        padLeft = treeModel.getPadLeft(lowResolution); // using low resolution
        padRight = treeModel.getPadRight(lowResolution);
        mosseLikelihoodCore = new MosseLikelihoodCore(stateCount, numRateBins, padLeft, padRight);

        // num non-zero entries (length of lambda and mu)
        this.numEntries = numRateBins - padLeft - padRight - 1;

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
        m_fRootPartials = new double[patterns * stateCount * numRateBins];
        matrixSize = (stateCount + 1) * (stateCount + 1);
        probabilities = new double[(stateCount + 1) * (stateCount + 1)];
        Arrays.fill(probabilities, 1.0);

        if (dataInput.get().isAscertained) {
            useAscertainedSitePatterns = true;
        }
    }

    /**
     * set leaf partials using tip GLM likelihood model *
     */
    @Override
    protected void setPartials(Node node, int patternCount) {
        if (node.isLeaf()) {
            Alignment data = dataInput.get();
            int states = data.getDataType().getStateCount();
            double[] traitValues = getTraits(node);
            double[] partials = new double[patternCount * (states + 1) * numRateBins];
            int k = 0;
            int taxonIndex = data.getTaxonIndex(node.getID());
            for (int patternIndex = 0; patternIndex < patternCount; patternIndex++) {
                double[] tipLikelihoods = tipModel.getTipLikelihoods(traitValues, numEntries, startSubsRate);
                int stateCount = data.getPattern(taxonIndex, patternIndex);
                    boolean[] stateSet = data.getStateSet(stateCount);
                    // E initial values are zero
                    for (int i = 0; i < numRateBins; i++) {
                        partials[k++] = 0.0;
                    }
                    // D initial values
                    for (int state = 0; state < states; state++) {
                        if (stateSet[state]) {
                            // set likelihoods for nucleotide in data
                            for (int i = 0; i < numRateBins; i++) {
                                if (i < numEntries) {
                                    partials[k++] = tipLikelihoods[i];
                                } else {
                                    partials[k++] = 0.0; // within padding
                                }
                            }
                        } else {
                            // otherwise set likelihoods to zero
                            for (int i = 0; i < numRateBins; i++) {
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

    private void traverseFull(Node node) {
        int numPlan = 5; // dimensions
        double deltaT = treeModel.dtInput.get().getValue(); // 0.001; // dt
        double rate = treeModel.dxInput.get().getValue(); // dx
        int numStates = dataInput.get().getDataType().getStateCount();
        int numPattern = dataInput.get().getPatternCount();

        double[] transitionMatrix = new double[numStates * numStates];
        double[][] transitionMatrices = new double[numEntries][transitionMatrix.length];
        // P(0) = exp(dx * Q * dt)
        
        System.out.println("deltaT=" + deltaT + "; rate=" + rate + "; numEntries=" + numEntries);
    
        
        substitutionModel.getTransitionProbabilities(node, deltaT, 0, rate, transitionMatrix); // startTime is greater than endTime
        
        System.out.print("**transitionMatrix:");
        for (int i = 0; i < transitionMatrix.length; i++)
        	System.out.print(" " + transitionMatrix[i]);
        System.out.println();
        
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
        
        // show the first entry of transitionMatrices
        System.out.println("First entry of transitionMatrices:");
        for (int i = 0; i < numStates; i++) {
        	for (int j = 0; j < numStates; j++) {
        		System.out.printf(" " + transitionMatrices[0][i * numStates + j]);
        	}
        	System.out.println();
        }
        
        // show the last entry of transitionMatrices
        System.out.println("Last entry of transitionMatrices -- " + (numEntries - 1) + "'s entry");
        for (int i = 0; i < numStates; i++) {
        	for (int j = 0; j < numStates; j++) {
        		System.out.printf(" " + transitionMatrices[(numEntries - 1)][i * numStates + j]);
        	}
        	System.out.println();
        }
        
        flatTransitionMatrices = Arrays.stream(transitionMatrices)
                .flatMapToDouble(Arrays::stream)
                .toArray();

        // get lambdas and mus
        double[] x = getSubstitutionRates(numEntries); // substitution rates
        lambdas = new double[numEntries];
        mus = new double[numEntries];
        // System.out.println("numEntries: " + numEntries);
        lambdas = lambdaFunc.getY(x, lambdas, true);
        mus = muFunc.getY(x, mus, true);

        if (node.isLeaf()) {
            // leaf node
            // leaf partials size = nrPatterns * (nrStates + 1) * numBins
            // columns = (4x D's for each nucleotide, 1x E), rows = bins for substitution rate
            setPartials(node, numPattern); // all site patterns

        } else if (!node.isRoot()) {
            // internal node
            traverseFull(node.getChild(0)); // left child
            traverseFull(node.getChild(1)); // right child

            // propagate child branches
            double branchTimeLeft = node.getHeight() - node.getLeft().getHeight();
            double branchTimeRight = node.getHeight() - node.getRight().getHeight();

            double[] patternPartialsLeft = new double[numPattern * numPlan * numRateBins];
            double[] patternPartialsRight = new double[numPattern * numPlan * numRateBins];
            double[] partialsAllPatterns = new double[numPattern * numPlan * numRateBins];

            // get child node partials all patterns
            mosseLikelihoodCore.getNodePartials(node.getLeft().getNr(), patternPartialsLeft);
            mosseLikelihoodCore.getNodePartials(node.getRight().getNr(), patternPartialsRight);
            int k = 0;
            for (int pattern = 0; pattern < numPattern; pattern++) {
                // partial for single pattern
                int partialSize = numPlan * numRateBins;
                int startPos = pattern * partialSize;
                double[] partialsLeft =  new double[numPlan * numRateBins];
                System.arraycopy(patternPartialsLeft, startPos, partialsLeft, 0, partialSize);
                double[] partialsRight = new double[numPlan * numRateBins];
                System.arraycopy(patternPartialsRight, startPos, partialsRight, 0, partialSize);
                double[] partialsCombined = new double[partialsLeft.length];

                // propagate each child branch
                double leftP = treeModel.calculateBranchLogP(branchTimeLeft, partialsLeft, lambdas, mus, flatTransitionMatrices, partialsLeft);
                double rightP = treeModel.calculateBranchLogP(branchTimeRight, partialsRight, lambdas, mus, flatTransitionMatrices, partialsRight);

//                System.out.println("leftP " + leftP);
//                System.out.println("rightP " + rightP);

                int leftBound = mosseLikelihoodCore.padLeft + 1;
                int rightBound = numEntries + leftBound;
                // assumes t less than tc threshold
                for (int i = 0; i < numPlan; i++) {
                    for (int j = 0; j < numRateBins; j++) {
                        int index = i * numRateBins + j;
                        if (i == 0) {
                            // E is topology independent
                            partialsCombined[index] = partialsLeft[index]; // for testing
                            partialsAllPatterns[k] = partialsLeft[index];
                        } else {
                            if (j >= leftBound && j < rightBound) {
                                // non padded elements
                                // D_left * D_right * lambda(x)
                                double lambdaX = lambdas[j - leftBound]; // birth rate at substitution rate x
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
            // set internal node partials for all patterns
            mosseLikelihoodCore.setNodePartials(node.getNr(), partialsAllPatterns);

        } else {
            // root node
            traverseFull(node.getLeft());
            traverseFull(node.getRight());

            // propagate child branches of root
            double branchTimeLeft = node.getHeight() - node.getLeft().getHeight();
            double branchTimeRight = node.getHeight() - node.getRight().getHeight();

            double[] patternPartialsLeft = new double[numPattern * numPlan * numRateBins];
            double[] patternPartialsRight = new double[numPattern * numPlan * numRateBins];
            double[] partialsAllPatterns = new double[numPattern * numPlan * numRateBins];

            // get child node partials all patterns
            mosseLikelihoodCore.getNodePartials(node.getLeft().getNr(), patternPartialsLeft);
            mosseLikelihoodCore.getNodePartials(node.getRight().getNr(), patternPartialsRight);
            int k = 0;
            for (int pattern = 0; pattern < numPattern; pattern++) {
                // partial for single pattern
                int partialSize = numPlan * numRateBins;
                int startPos = pattern * partialSize;
                double[] partialsLeft = new double[numPlan * numRateBins];
                System.arraycopy(patternPartialsLeft, startPos, partialsLeft, 0, partialSize);
                double[] partialsRight = new double[numPlan * numRateBins];
                System.arraycopy(patternPartialsRight, startPos, partialsRight, 0, partialSize);
                double[] partialsCombined = new double[partialsLeft.length];

                // propagate each child branch
                treeModel.calculateBranchLogP(branchTimeLeft, partialsLeft, lambdas, mus, flatTransitionMatrices, partialsLeft);
                treeModel.calculateBranchLogP(branchTimeRight, partialsRight, lambdas, mus, flatTransitionMatrices, partialsRight);

                // assumes t less than tc threshold
                int leftBound = mosseLikelihoodCore.padLeft + 1;
                int rightBound = numEntries + leftBound;
                // assumes t less than tc threshold
                for (int i = 0; i < numPlan; i++) {
                    for (int j = 0; j < numRateBins; j++) {
                        int index = i * numRateBins + j;
                        if (i == 0) {
                            // E is topology independent
                            partialsCombined[index] = partialsLeft[index]; // for testing
                            partialsAllPatterns[k] = partialsLeft[index];
                        } else {
                            if (j >= leftBound && j < rightBound) {
                                // non padded elements
                                // D_left * D_right * lambda(x)
                                double lambdaX = lambdas[j - leftBound]; // birth rate at substitution rate x
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
            // set root node partials
            mosseLikelihoodCore.setNodePartials(node.getNr(), partialsAllPatterns);

            logP = 0.0;
            for (int pattern = 0; pattern < numPattern; pattern++) {
                int partialSize = numPlan * numRateBins;
                int startPos = pattern * partialSize;
                double[] partials =  new double[numPlan * numRateBins];
                System.arraycopy(partialsAllPatterns, startPos, partials, 0, partialSize);
                // root calc for a single pattern
                boolean conditionSurv = false;
                
                double patternLogLikelihood = makeRootFuncMosse(numRateBins, rate, resolution, partials, conditionSurv);
                logP += patternLogLikelihood + dataInput.get().getPatternWeight(pattern);
            }
        }
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
        // show the values of vals
        System.out.println("vals:");
        for (int j = 0; j < ntypes + 1; j++) { // nucleotide types columns
            for (int i = 0; i < nx; i++) { // nx rows
            	System.out.print(" " + vals[i][j]);
            }
            System.out.println();
        }

        double[][] dRoot = getDValues(vals); // get root D values in last column
        double[] eRoot = null;

        double[] x = getSubstitutionRates(nx);
        // root options
        double[][] rootP = getRootProb(dRoot, x, nx, rootOption, rootFunc);

        if (conditionSurv) {
            eRoot = getColumn(vals, 1); // get root E values as a column
            // apply function on dRoot
            for (int i = 0; i < ntypes; i++) {
                for (int j = 0; j < numEntries; j++) {
                    double lambdaX = lambdas[j];
                    // element-wise division of d column
                    dRoot[i][j] = dRoot[i][j] / (lambdaX * (1 - eRoot[j]) * (1 - eRoot[j]));
                }
            }
        }

        double[][] rootProduct = getProduct(rootP, dRoot);
        double lq = getSum(dRoot); // lq value is sum(D vector) for numerical underflow

        double logProb = Math.log(getSum(rootProduct) * dxScaled) + lq; // log for numerical underflow
        System.out.println("p = " + getSum(rootProduct) * dxScaled);

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
        res[0] = startSubsRate;
        double interval = startSubsRate; // use start rate as interval
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
