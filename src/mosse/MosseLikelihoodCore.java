package mosse;

import beast.base.core.Description;
import beast.base.evolution.likelihood.BeerLikelihoodCore;

/**
 * @author Kylie Chen
 * @author Thomas Wong
 */

@Description("Mosse likelihood core calculation class")
public class MosseLikelihoodCore extends BeerLikelihoodCore {

	protected int numRateBins;

	protected int lambdaSize;
	
	protected double[][][][] mossePartials;

	public MosseLikelihoodCore(int nrOfStates, int numRateBins) {
		super(nrOfStates);
		this.numRateBins = numRateBins;
	}

	/**
	 * initializes partial likelihood arrays.
	 *
	 * @param nodeCount           the number of nodes in the tree
	 * @param patternCount        the number of patterns
	 * @param matrixCount         the number of matrices (i.e., number of categories
	 *                            for gamma rate heterogeneity)
	 * @param integrateCategories whether sites are being integrated over all
	 *                            matrices
	 * @param useAmbiguities      whether to use ambiguious characters
	 */
	@Override
	public void initialize(int nodeCount, int patternCount, int matrixCount, boolean integrateCategories,
			boolean useAmbiguities) {
		this.nrOfNodes = nodeCount;
		this.nrOfPatterns = patternCount;
		this.nrOfMatrices = matrixCount; // matrix count should be 1
		this.integrateCategories = integrateCategories;

		mossePartials = new double[2][nodeCount][matrixCount][];

		currentMatrixIndex = new int[nodeCount];
		storedMatrixIndex = new int[nodeCount];

		currentPartialsIndex = new int[nodeCount];
		storedPartialsIndex = new int[nodeCount];

		states = new int[nodeCount][];

		for (int i = 0; i < nodeCount; i++) {
			for (int c = 0; c < matrixCount; c++) {
				mossePartials[0][i][c] = null;
				mossePartials[1][i][c] = null;
			}
			states[i] = null;
		}
	}

	public void setNodeMossePartials(int nodeIndex, double[][] partialsIn) {
		this.mossePartials[currentPartialsIndex[nodeIndex]][nodeIndex] = partialsIn;
	}

	public double[][] getNodeMossePartials(int nodeIndex) {
		return mossePartials[currentPartialsIndex[nodeIndex]][nodeIndex];
	}

	public void setNodeMossePartials(int nodeIndex, int categoryIndex, double[] partialsIn) {
		this.mossePartials[currentPartialsIndex[nodeIndex]][nodeIndex][categoryIndex] = partialsIn;
	}

	public double[] getNodeMossePartials(int nodeIndex, int categoryIndex) {
		return mossePartials[currentPartialsIndex[nodeIndex]][nodeIndex][categoryIndex];
	}
}
