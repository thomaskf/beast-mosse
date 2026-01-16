package mosse;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;

import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ExecutionException;
import java.util.stream.IntStream;
import java.util.concurrent.ForkJoinWorkerThread;

/**
 * @author Thomas Wong
 */

@Description("MosseTreeLikelihoodBuffered with multi-threaded per-node computation")
public class MosseTreeLikelihoodBufferedMT extends MosseTreeLikelihoodBuffered implements AutoCloseable {

    private ForkJoinPool pool;
    
    static int threadIndexInPool() {
        Thread t = Thread.currentThread();
        if (t instanceof ForkJoinWorkerThread) {
            return ((ForkJoinWorkerThread) t).getPoolIndex();
        }
        return 0;
    }
    
    @Override
    public void initAndValidate() {
        super.initAndValidate();
        pool = (treeModel.numThreads == 1) ? null : new ForkJoinPool(treeModel.numThreads);
    }

    @Override
    public void close() {
        if (pool != null) {
            pool.shutdownNow();
            pool = null;
        }
    }    

    @Override
    protected void finalize() throws Throwable {
        close();
    }
    
    @Override
    protected void computePartialLikelihood(Node node) {
        if (node.isLeaf()) return;

		// to store the pattern -> sub-pattern id
		int[] patternMapSubpatternID = new int[patterns];

		// internal node or the root node
		double logPNode = 0.0;
		logPNode += logCompensatesPerNode[node.getLeft().getNr()];
		logPNode += logCompensatesPerNode[node.getRight().getNr()];

		// get child node partials all patterns
		final double[] patternPartialsLeft = mosseLikelihoodCore.getNodePartials(node.getLeft().getNr());
		final double[] patternPartialsRight = mosseLikelihoodCore.getNodePartials(node.getRight().getNr());
		 
		// numRateBins, numEntries, lambdas
		int numRateBins_curr = numRateBins_h;
		if (isLowResolution(node)) {
			numRateBins_curr = numRateBins_l;
		}
		numRateBinsPerNode[node.getNr()] = numRateBins_curr;
		int singlePartialSize = numPlan * numRateBins_curr;
		int t = node.getNr() * patterns; // starting pos in patternMapPerNode
		int subpatns = computeMapGlobal2Subpattern(patternMapSubpatternID, node, singlePartialSize, t);

		assert (subpatns > 0);
		
		double[] partialsAllPatterns = new double[subpatns * singlePartialSize];
		int[] freq = new int[subpatns];
		Arrays.fill(freq, 0);
        int[] rep = new int[subpatns];
        Arrays.fill(rep, -1);
		double[] logCompensates = new double[subpatns];

        if (node.isRoot()) {
            // root: subpattern == patternIndex, so representative is itself
            for (int p = 0; p < patterns; p++) {
            	rep[p] = p;
            	freq[p] = data.getPatternWeight(p);
            }
        } else {
            for (int p = 0; p < patterns; p++) {
                int sid = patternMapSubpatternID[p];
                if (rep[sid] < 0) rep[sid] = p;
                freq[sid] += data.getPatternWeight(p);
            }
        }

		if (pool == null) {
			// single thread
			int threadID = 0;
			for (int sid = 0; sid < subpatns; sid++) {
	            int patternIndex = rep[sid];
	            if (patternIndex >= 0) {
	            	logCompensates[sid] = computePartialLikelihoodPattern(patternIndex, node, patternPartialsLeft, patternPartialsRight, partialsAllPatterns, threadID);
	            }
			}
		} else {
			// multiple threads
	        Runnable job = () -> IntStream.range(0, subpatns).parallel().forEach(sid -> {
	            int patternIndex = rep[sid];
	            if (patternIndex >= 0) {
	            	int threadID = threadIndexInPool();
	            	logCompensates[sid] = computePartialLikelihoodPattern(patternIndex, node, patternPartialsLeft, patternPartialsRight, partialsAllPatterns, threadID);
	            }
	        });
            try {
                pool.submit(job).get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(e);
            } catch (ExecutionException e) {
                throw new RuntimeException(e.getCause());
            }
		}
		
        // accumulate node log compensate across patterns
        for (int p = 0; p < subpatns; p++) logPNode += logCompensates[p] * freq[p];
        
        mosseLikelihoodCore.setNodePartials(node.getNr(), partialsAllPatterns);

        // Root pattern likelihoods
        if (node.isRoot()) {
	        if (pool == null) {
	        	// single thread
	        	for (int p = 0; p < patterns; p++) {
	                int startPos = patternMapPerNode[t + p];
	                double[] partials = new double[singlePartialSize];
	                System.arraycopy(partialsAllPatterns, startPos, partials, 0, singlePartialSize);
	
	                boolean conditionSurv = false;
	                double pll = makeRootFuncMosse(numRateBins_l, dx_l, resolution, partials, conditionSurv);
	                patternLogLikelihoods[p] = pll;
	        	}
	        } else {
	        	// multi-threaded
	            Runnable rootJob = () -> IntStream.range(0, patterns).parallel().forEach(p -> {
	                int startPos = patternMapPerNode[t + p];
	                double[] partials = new double[singlePartialSize];
	                System.arraycopy(partialsAllPatterns, startPos, partials, 0, singlePartialSize);
	
	                boolean conditionSurv = false;
	                double pll = makeRootFuncMosse(numRateBins_l, dx_l, resolution, partials, conditionSurv);
	                patternLogLikelihoods[p] = pll;
	            });
	            try { pool.submit(rootJob).get(); }
	            catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new RuntimeException(e); }
	            catch (ExecutionException e) { throw new RuntimeException(e.getCause()); }
	        }
        }
        
        logCompensatesPerNode[node.getNr()] = logPNode;
    }
}
