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

	protected double[][][] mossePartials;

	protected double[][][] mosseCompensates;

	public MosseLikelihoodCore(int nrOfStates, int numRateBins) {
		super(nrOfStates);
		this.numRateBins = numRateBins;
	}

	/**
	 * initializes partial likelihood arrays.
	 *
	 * @param nodeCount           the number of nodes in the tree
	 * @param patternCount        the number of patterns
	 * @param integrateCategories whether sites are being integrated over all matrices
	 * @param useAmbiguities      whether to use ambiguous characters
	 */
	public void initialize(int nodeCount, int patternCount, boolean integrateCategories, boolean useAmbiguities) {
		this.nrOfNodes = nodeCount;
		this.nrOfPatterns = patternCount;
		this.nrOfMatrices = 1;
		this.integrateCategories = integrateCategories;

		mossePartials = new double[2][nodeCount][];
		mosseCompensates = new double[2][nodeCount][];

		currentMatrixIndex = new int[nodeCount];
		storedMatrixIndex = new int[nodeCount];

		currentPartialsIndex = new int[nodeCount];
		storedPartialsIndex = new int[nodeCount];

		states = new int[nodeCount][];
	}

	public void setNodeMossePartials(int nodeIndex, double[] partialsIn) {
		this.mossePartials[currentPartialsIndex[nodeIndex]][nodeIndex] = partialsIn;
	}

	public double[] getNodeMossePartials(int nodeIndex) {
		return mossePartials[currentPartialsIndex[nodeIndex]][nodeIndex];
	}

	public void setNodeMosseCompensates(int nodeIndex, double[] compensatesIn) {
		this.mosseCompensates[currentPartialsIndex[nodeIndex]][nodeIndex] = compensatesIn;
	}

	public double[] getNodeMosseCompensates(int nodeIndex) {
		return mosseCompensates[currentPartialsIndex[nodeIndex]][nodeIndex];
	}
}
