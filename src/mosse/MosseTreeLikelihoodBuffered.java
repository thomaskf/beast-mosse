package mosse;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.CalculationNode;

/**
 * @author Kylie Chen
 * @author Thomas Wong
 */

@Description("Mosse likelihood class calculates the probability of sequence and trait data on a tree")
public class MosseTreeLikelihoodBuffered extends MosseTreeLikelihood {

	private boolean updateTips;
	private boolean updateSiteModel;
	private boolean updateTreeModel;
	private boolean updateFunc;

	// array for storing during MCMC
	protected int[] storedTaxaIndexUnderNode;
	protected int[] storedPattern2SubpatnPerNode;
	protected int[] storedNumRateBinsPerNode;
	protected double[] storedFlatTransitionMatrices_h;
	protected double[] storedFlatTransitionMatrices_l;
	protected double[] storedPatternLogLikelihoods;

	@Override
	public void initAndValidate() {

		super.initAndValidate();

		// alignment = dataInput.get();
		// alignment -> data

		// initialize the flags
		updateTips = true;
		updateSiteModel = true;
		updateTreeModel = true;
		updateFunc = true;

		storedTaxaIndexUnderNode = new int[nodeCount * taxonCount];
		storedPattern2SubpatnPerNode = new int[nodeCount * patterns];
		storedNumRateBinsPerNode = new int[nodeCount];
		storedPatternLogLikelihoods = new double[patterns];
	}

	/**
	 * traverse tree with optimized caching
	 *
	 * @param node tree node
	 * @return update flag
	 */
	@Override
	protected int traverse(final Node node) {

		int update = node.isDirty();
		
		if (node.isRoot()) {
			// recalculate taxon indices under all children of each node
			setTaxonIndices(node);
			// compute lambdas_h, lambdas_l, mus_h, and mus_l
			computeLambdaMus();
			
			if (updateSiteModel || updateTreeModel || updateFunc) {
				// recompute all the transition matrices
				boolean lowResolution = true;
				flatTransitionMatrices_l = createFlatTransitionMatrice(node, lowResolution);
				lowResolution = false;
				flatTransitionMatrices_h = createFlatTransitionMatrice(node, lowResolution);
			}
			
			if (updateTips || updateTreeModel || updateSiteModel || updateFunc) {
				// update the partial of all leaves
				setPartials(node, patterns);
			}
			
		}
		

		// -------- leaf --------
		if (node.isLeaf()) {
			// nothing else to do for a leaf
			return update;
		}

		// --------- internal node ----------
		final int update1 = traverse(node.getLeft());
		final int update2 = traverse(node.getRight());

		// if either child was updated, we must recompute this node's partials
		if (update1 != Tree.IS_CLEAN || update2 != Tree.IS_CLEAN || updateSiteModel || updateTips || updateTreeModel || updateFunc) {

			mosseLikelihoodCore.setNodePartialsForUpdate(node.getNr());
			// System.out.println("Invoke computePartialLikelihood for node " + node.getNr());
			computePartialLikelihood(node);

		}
		update |= (update1 | update2);

		if (node.isRoot()) {
			// compute the logP
			calcLogP();
			// reset the flag
			updateTips = false;
			updateSiteModel = false;
			updateTreeModel = false;
			updateFunc = false;
		}

		return update;
	}

	@Override
	public double calculateLogP() {
		traverse(tree.getRoot());
		printLogP();
		return logP;
	}

	@Override
	protected boolean requiresRecalculation() {
		
		// printParams();

		// for debugging
		// checkNodeStatus(tree.getRoot());

		boolean recalc = false;
		
		if (treeModel.isDirtyCalculation()) {
			updateTreeModel = true;
		}

		if (m_siteModel.isDirtyCalculation()) {
			// update RHAS information
			if (numCategories > 1) {
				categoryRates = m_siteModel.getCategoryRates(null);
				categoryProps = m_siteModel.getCategoryProportions(null);
			}
			updateSiteModel = true;
		}

		if (tipModel.isDirtyCalculation()) {
			updateTips = true;
		}
		
		if (((CalculationNode) lambdaFunc).isDirtyCalculation() || ((CalculationNode) muFunc).isDirtyCalculation()) {
			updateFunc = true;
		}

		if (treeInput.get().somethingIsDirty() || updateTreeModel || updateSiteModel || updateTips || updateFunc) {
			recalc = true;
		}
		
		return recalc;
	}

	@Override
	public void store() {
		System.out.println("store");
		mosseLikelihoodCore.store();

		super.store(); // important: let the parent class store its state

		System.arraycopy(taxaIndexUnderNode, 0, storedTaxaIndexUnderNode, 0, taxaIndexUnderNode.length);
		System.arraycopy(pattern2SubpatnPerNode, 0, storedPattern2SubpatnPerNode, 0, pattern2SubpatnPerNode.length);
		System.arraycopy(numRateBinsPerNode, 0, storedNumRateBinsPerNode, 0, numRateBinsPerNode.length);
		System.arraycopy(patternLogLikelihoods, 0, storedPatternLogLikelihoods, 0, patternLogLikelihoods.length);
		storedFlatTransitionMatrices_h = flatTransitionMatrices_h;
		storedFlatTransitionMatrices_l = flatTransitionMatrices_l;
	}

	@Override
	public void restore() {
		System.out.println("restore");
		mosseLikelihoodCore.restore();

		super.restore(); // restore parent state (tree, partials, etc.)

		int[] tmp;
		double[] tmp2;

		tmp = taxaIndexUnderNode;
		taxaIndexUnderNode = storedTaxaIndexUnderNode;
		storedTaxaIndexUnderNode = tmp;

		tmp = pattern2SubpatnPerNode;
		pattern2SubpatnPerNode = storedPattern2SubpatnPerNode;
		storedPattern2SubpatnPerNode = tmp;

		tmp = numRateBinsPerNode;
		numRateBinsPerNode = storedNumRateBinsPerNode;
		storedNumRateBinsPerNode = tmp;
	
		tmp2 = patternLogLikelihoods;
		patternLogLikelihoods = storedPatternLogLikelihoods;
		storedPatternLogLikelihoods = tmp2;
		
		tmp2 = flatTransitionMatrices_h;
		flatTransitionMatrices_h = storedFlatTransitionMatrices_h;
		storedFlatTransitionMatrices_h = tmp2;
		
		tmp2 = flatTransitionMatrices_l;
		flatTransitionMatrices_l = storedFlatTransitionMatrices_l;
		storedFlatTransitionMatrices_l = tmp2;
	}

	private void checkNodeStatus(final Node node) {
		if (node.isDirty() == Tree.IS_DIRTY) {
			System.out.println("node " + node.getNr() + " is dirty");
		}
		if (node.isDirty() == Tree.IS_FILTHY) {
			System.out.println("node " + node.getNr() + " is filthy");
		}
		if (!node.isLeaf()) {
			checkNodeStatus(node.getLeft());
			checkNodeStatus(node.getRight());
		}
	}

}
