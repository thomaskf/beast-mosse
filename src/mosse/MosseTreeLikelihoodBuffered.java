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

	private boolean updateTips = true;
	private boolean updateSiteModel = true;

	// array for storing during MCMC
	protected int[] storedTaxaIndexUnderNode;
	protected int[] storedPatternMapPerNode;
	protected int[] storedNumRateBinsPerNode;
	protected double[] storedLogCompensatesPerNode;

	@Override
	public void initAndValidate() {

		super.initAndValidate();

		// alignment = dataInput.get();
		// alignment -> data

		// initialize the flags
		updateTips = true;
		updateSiteModel = true;

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
			// taxon indices under all children of each node
			setTaxonIndices(node);
			if (updateSiteModel) {
				flatTransitionMatrices_l = null;
				flatTransitionMatrices_h = null;
			}
			if (updateTips) {
				// update all the partial for all leaves
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
		if (update1 != Tree.IS_CLEAN || update2 != Tree.IS_CLEAN || updateSiteModel) {

			mosseLikelihoodCore.setNodePartialsForUpdate(node.getNr());
			// System.out.println("Invoke computePartialLikelihood for node " + node.getNr());
			computePartialLikelihood(node);

		}
		update |= (update1 | update2);

		// reset the flag
		if (node.isRoot()) {
			updateTips = false;
			updateSiteModel = false;
		}

		return update;
	}

	@Override
	public double calculateLogP() {
		String newickstr = toNewick(tree.getRoot()) + ";";
		System.out.println(newickstr);
		System.out.println("tc = " + tc);
		// System.out.println("Before calculation");
		// showPatternMapPerNodeArray();
		if (requiresRecalculation()) {
			if (traverse(tree.getRoot()) != Tree.IS_CLEAN) {
				calcLogP();
			}
		}
		System.out.println("logP = " + logP);
		// System.out.println("After calculation");
		// showPatternMapPerNodeArray();
		System.out.println();
		return logP;
	}

	@Override
	protected boolean requiresRecalculation() {

		// for debugging
		// checkNodeStatus(tree.getRoot());

		boolean recalc = false;

		// If site model changed, we must recompute all partials
		if (m_siteModel.isDirtyCalculation() || updateSiteModel) {
			updateSiteModel = true;
			recalc = true;
		}

		// If tip model changed, we must recompute all leaf partials
		if (tipModel.isDirtyCalculation() || updateTips) {
			updateTips = true;
			recalc = true;
		}

		// If nothing global changed, check whether the tree itself has dirty nodes
		if (!recalc && treeInput.get().somethingIsDirty()) {
			recalc = true;
		}

		return recalc;
	}

	@Override
	public void store() {
		// System.out.println("Store!");

		mosseLikelihoodCore.store();

		super.store(); // important: let the parent class store its state

		System.arraycopy(taxaIndexUnderNode, 0, storedTaxaIndexUnderNode, 0, taxaIndexUnderNode.length);
		System.arraycopy(patternMapPerNode, 0, storedPatternMapPerNode, 0, patternMapPerNode.length);
		System.arraycopy(numRateBinsPerNode, 0, storedNumRateBinsPerNode, 0, numRateBinsPerNode.length);
		System.arraycopy(logCompensatesPerNode, 0, storedLogCompensatesPerNode, 0, logCompensatesPerNode.length);
	}

	@Override
	public void restore() {
		// System.out.println("Restore!");

		mosseLikelihoodCore.restore();

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
