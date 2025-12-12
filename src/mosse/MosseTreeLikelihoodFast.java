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
import java.util.HashMap;
import java.util.List;


/**
 * @author Kylie Chen
 * @author Thomas Wong
 */

@Description("Mosse likelihood class calculates the probability of sequence and trait data on a tree")
public class MosseTreeLikelihoodFast extends MosseTreeLikelihood {

    private boolean updateTips = true;
    private boolean updateSiteModel = true;
    

    // array for storing during MCMC
    protected int[] storedTaxaIndexUnderNode;
    protected int[] storedPatternMapPerNode;
    protected int[] storedSubpatternPerNode;
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
        storedSubpatternPerNode = new int[nodeCount];
        storedNumRateBinsPerNode = new int[nodeCount];
        storedLogCompensatesPerNode = new double[nodeCount];
    }
    
    /**
     * traverse tree with optimized caching
     * @param node tree node
     * @return update flag
     */
    @Override
    protected int traverse(final Node node) {
    	
    	if (node.isRoot()) {
        	// taxon indices under all children of each node
            setTaxonIndices(node);
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
        System.out.println(tree.toString());
        if (requiresRecalculation()) {
	    	if (traverse(tree.getRoot()) != Tree.IS_CLEAN) {
	    		calcLogP();
	    	}
        }
        System.out.println("logP = " + logP);
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
    
    private void checkNodeStatus(final Node node) {
    	if (node.isDirty() == Tree.IS_DIRTY) {
    		System.out.println ("node " + node.getNr() + " is dirty");
    	}
    	if (node.isDirty() == Tree.IS_FILTHY) {
    		System.out.println ("node " + node.getNr() + " is filthy");
    	}
    	if (!node.isLeaf()) {
    		checkNodeStatus(node.getLeft());
    		checkNodeStatus(node.getRight());
    	}
    }
    

    @Override
    public void store() {
        super.store();  // important: let the parent class store its state

        if (taxaIndexUnderNode != null) {
        	if (storedTaxaIndexUnderNode == null || storedTaxaIndexUnderNode.length != taxaIndexUnderNode.length) {
        		storedTaxaIndexUnderNode = new int[taxaIndexUnderNode.length];
        	}
        	System.arraycopy(taxaIndexUnderNode, 0, storedTaxaIndexUnderNode, 0, taxaIndexUnderNode.length);
        }
        if (patternMapPerNode != null) {
            if (storedPatternMapPerNode == null || storedPatternMapPerNode.length != patternMapPerNode.length) {
                storedPatternMapPerNode = new int[patternMapPerNode.length];
            }
            System.arraycopy(patternMapPerNode, 0, storedPatternMapPerNode, 0, patternMapPerNode.length);
        }
        if (subpatternPerNode != null ) {
        	if (storedSubpatternPerNode == null || storedSubpatternPerNode.length != subpatternPerNode.length) {
        		storedSubpatternPerNode = new int[subpatternPerNode.length];
        	}
        	System.arraycopy(subpatternPerNode, 0, storedSubpatternPerNode, 0, subpatternPerNode.length);
        }
        if (numRateBinsPerNode != null) {
        	if (storedNumRateBinsPerNode == null || storedNumRateBinsPerNode.length != numRateBinsPerNode.length) {
        		storedNumRateBinsPerNode = new int[numRateBinsPerNode.length];
        	}
        	System.arraycopy(numRateBinsPerNode, 0, storedNumRateBinsPerNode, 0, numRateBinsPerNode.length);
        }
        if (logCompensatesPerNode != null) {
        	if (storedLogCompensatesPerNode == null || storedLogCompensatesPerNode.length != logCompensatesPerNode.length) {
        		storedLogCompensatesPerNode = new double[logCompensatesPerNode.length];
        	}
        	System.arraycopy(logCompensatesPerNode, 0, storedLogCompensatesPerNode, 0, logCompensatesPerNode.length);
        }
    }

    @Override
    public void restore() {
        super.restore();  // restore parent state (tree, partials, etc.)

        // swap or copy back
        int[] tmp = taxaIndexUnderNode;
        taxaIndexUnderNode = storedTaxaIndexUnderNode;
        storedTaxaIndexUnderNode = tmp;
        
        tmp = patternMapPerNode;
        patternMapPerNode = storedPatternMapPerNode;
        storedPatternMapPerNode = tmp;
        
        tmp = subpatternPerNode;
        subpatternPerNode = storedSubpatternPerNode;
        storedSubpatternPerNode = tmp;
        
        tmp = numRateBinsPerNode;
        numRateBinsPerNode = storedNumRateBinsPerNode;
        storedNumRateBinsPerNode = tmp;
        
        double[] tmp2 = logCompensatesPerNode;
        logCompensatesPerNode = storedLogCompensatesPerNode;
        storedLogCompensatesPerNode = tmp2;
    }

}

