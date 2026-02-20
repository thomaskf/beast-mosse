package mosse;

import beast.base.core.Description;
import beast.base.evolution.tree.Node;

import java.lang.ref.Cleaner;
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
        // super.initAndValidate() already calls treeModel.initFFTPtrs(dx_h) for a
        // single-threaded pool (numThreads == 1). We close those pointers and
        // re-initialize with the correct thread-count before registering the Cleaner,
        // so that every code path ends up with exactly one correctly-sized pool.
        super.initAndValidate();

        // Replace the single-thread pool created by the parent with the real MT pool.
        // initFFTPtrs is idempotent with respect to the FFT plans because the parent
        // already set dx_h; we just need the right number of pointer slots.
        if (treeModel.numThreads > 1) {
            try {
                treeModel.close(); // release the single-thread FFT ptrs from super
            } catch (Exception e) {
                throw new RuntimeException("Failed to release FFT pointers before MT re-init", e);
            }
            treeModel.initFFTPtrs(dx_h); // re-init with numThreads slots
            pool = new ForkJoinPool(treeModel.numThreads);
        }
        // else: pool remains null (single-threaded), FFT ptrs already correct.

        // Register a Cleaner action so the pool is shut down even if close() is
        // never called explicitly — without relying on the deprecated finalize().
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
