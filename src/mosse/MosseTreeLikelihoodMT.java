package mosse;

import beast.base.core.Description;

import java.lang.ref.Cleaner;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinWorkerThread;

/**
 * @author Thomas Wong
 */

@Description("MosseTreeLikelihood with multi-threaded per-node computation")
public class MosseTreeLikelihoodMT extends MosseTreeLikelihood implements AutoCloseable {

    private static final Cleaner CLEANER = Cleaner.create();
    private Cleaner.Cleanable cleanable;

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
        // super.initAndValidate() reads treeModel.numThreads and calls
        // treeModel.initFFTPtrs(dx_h) with the correct number of slots already,
        // because numThreads is set from treeModel before initFFTPtrs is invoked.
        // We must NOT call close() + initFFTPtrs() again here — doing so would
        // set closed=true and leave the re-created native pointers unfreeable
        // (bug 1.2 fix: eliminate the double-init memory leak).
        super.initAndValidate();

        // Create the ForkJoinPool for multi-threaded computation if needed.
        // The pool is null (single-threaded) when numThreads == 1.
        if (treeModel.numThreads > 1) {
            pool = new ForkJoinPool(treeModel.numThreads);
        }
        // else: pool remains null (single-threaded path), already set by super.

        // Register a Cleaner so the pool is shut down even if close() is never
        // called explicitly. The lambda captures only 'pool' (a local copy), not
        // 'this', so the Cleaner can fire when this object becomes unreachable.
        ForkJoinPool poolRef = pool;
        cleanable = CLEANER.register(this, () -> {
            if (poolRef != null) poolRef.shutdownNow();
        });
    }

    @Override
    public void close() {
        if (cleanable != null) {
            cleanable.clean(); // runs the Cleaner action (shuts down pool) exactly once
            cleanable = null;
        }
        pool = null;
    }
}
