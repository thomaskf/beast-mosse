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

@Description("MosseTreeLikelihood with multi-threaded per-node computation")
public class MosseTreeLikelihoodMT extends MosseTreeLikelihood implements AutoCloseable {

	@Override
    protected int threadIndexInPool() {
        Thread t = Thread.currentThread();
        if (t instanceof ForkJoinWorkerThread) {
        	int localThreadID = ((ForkJoinWorkerThread) t).getPoolIndex();
            return localThreadID;
        }
        return 0;
    }
    
    @Override
    public void initAndValidate() {
        super.initAndValidate();
        pool = (treeModel.numThreads == 1) ? null : new ForkJoinPool(treeModel.numThreads);

		// initialize the fft pointers
		treeModel.initFFTPtrs(dx_h);
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
}
