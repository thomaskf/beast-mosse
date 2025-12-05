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
    
    // store the log compensations for each nodes
    private double[] log_compensates;

    // Force full recomputation on the first call to calculateLogP()
    private boolean firstEval = true;
    
    @Override
    public void initAndValidate() {
    	
    	super.initAndValidate();

    	alignment = dataInput.get();

    	// initialize the log-compensation array
        System.out.println("Number of nodes: " + treeInput.get().getNodeCount());
        log_compensates = new double[treeInput.get().getNodeCount()];
        Arrays.fill(log_compensates, 0.0);
        
        // optional but consistent: force a full recompute on first call
        hasDirt = Tree.IS_DIRTY;
        updateSiteModel = true;
        updateTips = true;
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
            
            HashMap<ArrayList<Integer>, Integer> subpattern2pos = new HashMap<ArrayList<Integer>, Integer>();
            HashMap<ArrayList<Integer>, Double> subpattern2logp = new HashMap<ArrayList<Integer>, Double>();
            Alignment data = dataInput.get();
        	int s = node.getNr() * data.getTaxonCount();

        	for (int pattern = 0; pattern < numPattern; pattern++) {
                // partial for single pattern
                int startPos = pattern * numPlan * numRateBins_max;
                ArrayList<Integer> subpattern = null;

                if (!node.isRoot()) {
	                // get the subpattern
	            	// first check whether the subpattern has appeared before
	            	subpattern = new ArrayList<Integer>();
	            	for (int i = 0; i < data.getTaxonCount(); i++) {
	            		if (taxaIndexUnderNode[s+i] == -1)
	            			break;
	            		int taxonIndex = taxaIndexUnderNode[s+i];
	            		int stateCount = data.getPattern(taxonIndex, pattern);
	            		subpattern.add(stateCount);
	            	}
                }
                double logp = 0.0;
            	if (subpattern != null && !subpattern.isEmpty() && subpattern2pos.containsKey(subpattern)) {
            		int pos = subpattern2pos.get(subpattern);
            		logp = subpattern2logp.get(subpattern);
            		System.arraycopy(partialsAllPatterns, pos, partialsAllPatterns, startPos, numPlan * numRateBins_curr);
            	} else {
	                int partialSizeLeft = numPlan * numRateBins_left;
	                int partialSizeRight = numPlan * numRateBins_right;
	                double[] partialsLeft = new double[partialSizeLeft];
	                System.arraycopy(patternPartialsLeft, startPos, partialsLeft, 0, partialSizeLeft);
	                double[] partialsRight = new double[partialSizeRight];
	                System.arraycopy(patternPartialsRight, startPos, partialsRight, 0, partialSizeRight);
	                
	                // propagate each child branch
	                logp += computeSingleBranchLikelihood(node, child1, partialsLeft);
	                logp += computeSingleBranchLikelihood(node, child2, partialsRight);
	
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
	                if (subpattern != null) {
	                	subpattern2pos.put(subpattern, startPos);
	                	subpattern2logp.put(subpattern, logp);
	                }
            	}
            	logPNode += logp;
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
    
    @Override
    public double calculateLogP() {
        final TreeInterface tree = treeInput.get();
        final int rootIndex = tree.getRoot().getNr();
        System.out.println(tree.toString());
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
