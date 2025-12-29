package mosse;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;

/**
 * @author Kylie Chen
 * @author Thomas Wong
 */

@Description("Mosse likelihood class calculates the probability of sequence and trait data on a tree")
public class MosseTreeLikelihoodBuffered extends MosseTreeLikelihood {

	private boolean updateTips;
	private boolean updateSiteModel;
	private boolean updateTreeModel;

	// array for storing during MCMC
	protected int[] storedTaxaIndexUnderNode;
	protected int[] storedPatternMapPerNode;
	protected int[] storedNumRateBinsPerNode;
	protected double[] storedLogCompensatesPerNode;
	protected double[] storedFlatTransitionMatrices_h;
	protected double[] storedFlatTransitionMatrices_l;

	@Override
	public void initAndValidate() {

		super.initAndValidate();

		// alignment = dataInput.get();
		// alignment -> data

		// initialize the flags
		updateTips = true;
		updateSiteModel = true;
		updateTreeModel = true;

		storedTaxaIndexUnderNode = new int[taxonCount * nodeCount];
		storedPatternMapPerNode = new int[nodeCount * patterns];
		storedNumRateBinsPerNode = new int[nodeCount];
		storedLogCompensatesPerNode = new double[nodeCount];
	}

	/**
	 * traverse tree with optimized caching
	 *
	 * @param node tree node
	 * @return update flag
	 */
	@Override
	protected int traverse(final Node node) {

		if (node.isRoot()) {
			if (updateTreeModel) {
				// update the values of pads and numEntries
				treeModel.computePadNumEntries();
				// compute lambda_h, lambda_l, mus_h, and mus_l
				computeLambdaMus();
			}
			// taxon indices under all children of each node
			setTaxonIndices(node);
			if (updateSiteModel || updateTreeModel) {
				boolean lowResolution = true;
				flatTransitionMatrices_l = createFlatTransitionMatrice(node, lowResolution);
				lowResolution = false;
				flatTransitionMatrices_h = createFlatTransitionMatrice(node, lowResolution);
			}
			if (updateTips || updateTreeModel) {
				// update all the partial for all leaves
				// System.out.println("update the partials for all leaves");
				setPartials(node, patterns);
			}
		}

		int update = node.isDirty();

		// -------- leaf --------
		if (node.isLeaf()) {
			// nothing else to do for a leaf
			return update;
		}

		// --------- internal node ----------
		final int update1 = traverse(node.getLeft());
		final int update2 = traverse(node.getRight());

		// if either child was updated, we must recompute this node's partials
		if (update1 != Tree.IS_CLEAN || update2 != Tree.IS_CLEAN || updateSiteModel || updateTips) {

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
		}

		return update;
	}

	@Override
	public double calculateLogP() {
		printParams();
		if (requiresRecalculation()) {
			traverse(tree.getRoot());
		}
		printLogP();
		return logP;
	}

	@Override
	protected boolean requiresRecalculation() {

		// for debugging
		// checkNodeStatus(tree.getRoot());

		boolean recalc = false;
		
		if (treeModel.isDirtyCalculation()) {
			updateTreeModel = true;
		}
		

		if (m_siteModel.isDirtyCalculation()) {
			updateSiteModel = true;
		}

		if (tipModel.isDirtyCalculation()) {
			updateTips = true;
		}

		if (treeInput.get().somethingIsDirty() || updateTreeModel || updateSiteModel || updateTips) {
			recalc = true;
		}
		
		return recalc;
	}

	@Override
	public void store() {
		// System.out.println("Store!");

		mosseLikelihoodCore.store();
		treeModel.store();

		super.store(); // important: let the parent class store its state

		System.arraycopy(taxaIndexUnderNode, 0, storedTaxaIndexUnderNode, 0, taxaIndexUnderNode.length);
		System.arraycopy(patternMapPerNode, 0, storedPatternMapPerNode, 0, patternMapPerNode.length);
		System.arraycopy(numRateBinsPerNode, 0, storedNumRateBinsPerNode, 0, numRateBinsPerNode.length);
		System.arraycopy(logCompensatesPerNode, 0, storedLogCompensatesPerNode, 0, logCompensatesPerNode.length);
		storedFlatTransitionMatrices_h = flatTransitionMatrices_h;
		storedFlatTransitionMatrices_l = flatTransitionMatrices_l;
	}

	@Override
	public void restore() {
		// System.out.println("Restore!");

		mosseLikelihoodCore.restore();
		treeModel.restore();

		super.restore(); // restore parent state (tree, partials, etc.)

		int[] tmp;
		double[] tmp2;

		tmp = taxaIndexUnderNode;
		taxaIndexUnderNode = storedTaxaIndexUnderNode;
		storedTaxaIndexUnderNode = tmp;

		tmp = patternMapPerNode;
		patternMapPerNode = storedPatternMapPerNode;
		storedPatternMapPerNode = tmp;

		tmp = numRateBinsPerNode;
		numRateBinsPerNode = storedNumRateBinsPerNode;
		storedNumRateBinsPerNode = tmp;

		tmp2 = logCompensatesPerNode;
		logCompensatesPerNode = storedLogCompensatesPerNode;
		storedLogCompensatesPerNode = tmp2;
		
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
