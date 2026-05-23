# Running BEAST2/MOSSE MCMC

This is a practical reference for running this MCMC efficiently on a multi-core
Linux server. Tested on rona (2× AMD EPYC 7742, 128 physical cores / 256
hardware threads, 8 NUMA nodes, Ubuntu 22.04, OpenJDK 17, FFTW3, GSL).

The settings here came from a tuning pass that took the original baseline from
~3.46 sec/MCMC-step down to ~0.93 sec/step (~3.7× wall-time speedup). The wins
are mostly source-code changes already in the tree; only a few choices are
*per-machine* and listed below.

---

## TL;DR — the launcher command

After everything is built (see "One-time setup"), every MCMC run is:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate mosse-build
cd /path/to/your/run_dir

CP="$HOME/mosse/beast-libs/beast/lib/launcher.jar:\
$HOME/mosse/beast-libs/BEAST.base/lib/BEAST.base.jar:\
$HOME/mosse/beast-libs/BEAST.app/lib/BEAST.app.jar:\
$HOME/mosse/beast-libs/BEASTLabs/lib/BEASTlabs.v2.0.3.jar:\
$HOME/mosse/beast-mosse/lib/jblas-1.2.5.jar:\
$HOME/mosse/beast-mosse/build/mosse.jar"

LIBPATH=$HOME/mosse/beast-mosse/build/dist

taskset -c 0-127 java \
  -Djava.util.concurrent.ForkJoinPool.common.parallelism=1 \
  -Djava.library.path=$LIBPATH \
  -cp "$CP" \
  beastfx.app.beast.BeastMCMC -seed 42 -overwrite your.xml
```

Three flags on the command line matter — don't drop them:

| Flag | Why |
|--|--|
| `taskset -c 0-127` | Pin to one logical CPU per physical core. SMT siblings on AMD EPYC contend for the same vector units; pinning recovers ~7% on FP-heavy MOSSE work. |
| `-Djava.util.concurrent.ForkJoinPool.common.parallelism=1` | Caps the JVM common ForkJoinPool so any stray `parallelStream` elsewhere in BEAST can't spawn 256 threads behind your back. |
| `-Djava.library.path=...` | Tells the JVM where `libtest.so` lives. |

For long runs, prepend `nohup` and append `> mcmc.out 2>&1 < /dev/null &; disown`
so the process survives ssh disconnects.

To **resume** instead of fresh start, swap `-overwrite` for `-resume`.

---

## XML requirements

In your MCMC XML, the `MosseDistribution` element must set `threads` to the
**number of physical cores** (NOT logical threads).

```xml
<MosseDistribution id="treemodel"
    spec="mosse.MosseDistribution" tree="@tree"
    nx="1024" dt="0.01" width="5" resolution="4" threads="128">
    ...
</MosseDistribution>
```

Rule: `threads = physical-core-count`. Going higher hurts — SMT siblings
contend for the same FPU. On rona that's `128`.

The tree-likelihood element must use `mosse.MosseTreeLikelihoodMT` (the MT
suffix, not the plain `MosseTreeLikelihood` — only MT creates the
ForkJoinPool):

```xml
<distribution id="mossetreelikelihood"
    spec="mosse.MosseTreeLikelihoodMT"
    ...>
```

---

## One-time setup (per machine)

### 1. Build environment via conda

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda create -y -n mosse-build -c conda-forge openjdk=17 fftw gsl ant make
conda activate mosse-build
```

Verify:
```bash
javac -version       # should print 17.x
gsl-config --version
ls $CONDA_PREFIX/include/fftw3.h
```

System `gcc` is fine (11.4 on rona). No need for the conda compilers.

### 2. BEAST jars

```bash
mkdir -p ~/mosse/beast-libs && cd ~/mosse/beast-libs
wget https://github.com/CompEvol/beast2/releases/download/v2.7.7/BEAST.v2.7.7.Linux.x86.tgz
tar xzf BEAST.v2.7.7.Linux.x86.tgz                                # provides launcher.jar
wget https://github.com/CompEvol/beast2/releases/download/v2.7.8/BEAST.base.package.v2.7.8.zip
wget https://github.com/CompEvol/beast2/releases/download/v2.7.8/BEAST.app.package.v2.7.8.zip
wget https://github.com/BEAST2-Dev/BEASTLabs/releases/download/v2.0.0/BEASTLabs.package.v2.0.3.zip
mkdir BEAST.base BEAST.app BEASTLabs
unzip -q -d BEAST.base BEAST.base.package.v2.7.8.zip
unzip -q -d BEAST.app  BEAST.app.package.v2.7.8.zip
unzip -q -d BEASTLabs  BEASTLabs.package.v2.0.3.zip
```

This populates `~/mosse/beast-libs/{beast/lib/launcher.jar, BEAST.base/lib/BEAST.base.jar,
BEAST.app/lib/BEAST.app.jar, BEASTLabs/lib/BEASTlabs.v2.0.3.jar}`.

### 3. Build the JNI shared library (`libtest.so`)

**This is per-machine** — the flags target a specific CPU architecture.

For rona (Zen2):
```bash
cd ~/mosse/beast-mosse/jni
gcc -shared -fPIC -O3 -march=znver2 -funroll-loops -Wno-unused-function \
    -I. \
    -I$CONDA_PREFIX/lib/jvm/include \
    -I$CONDA_PREFIX/lib/jvm/include/linux \
    -I$CONDA_PREFIX/include \
    -L$CONDA_PREFIX/lib \
    -o libtest.so main.c \
    -lm -lfftw3 -lgsl -lgslcblas \
    -Wl,-rpath,$CONDA_PREFIX/lib
mkdir -p ~/mosse/beast-mosse/build/dist
cp libtest.so ~/mosse/beast-mosse/build/dist/libtest.so
```

For **other machines**, replace `-march=znver2` with the right target:

| Machine | `-march=` |
|--|--|
| AMD EPYC 7xx2 series (Rome) | `znver2` |
| AMD EPYC 7xx3 series (Milan) | `znver3` |
| AMD EPYC 9xx4 series (Genoa) | `znver4` |
| Intel Xeon Skylake/Cascade Lake | `skylake-avx512` |
| Intel Xeon Ice Lake | `icelake-server` |
| Intel Xeon Sapphire Rapids | `sapphirerapids` |
| Anywhere (portable AVX2 baseline) | `x86-64-v3` |
| If unsure & you're building on the run host | `native` |

`-march=native` always works when you build on the same machine you run on.
Skipping `-march` (i.e. plain `-O3`) loses ~22% of the speed — generic codegen
won't issue AVX2/FMA. Don't drop it.

`-O3 -funroll-loops` accounts for most of the FP-loop speed; `-march=znver2`
adds the Zen2-specific AVX2+FMA codegen on top.

### 4. Build the Java jar

```bash
cd ~/mosse/beast-mosse
CP="$HOME/mosse/beast-libs/BEAST.base/lib/BEAST.base.jar:\
$HOME/mosse/beast-libs/BEAST.app/lib/BEAST.app.jar:\
$HOME/mosse/beast-libs/BEASTLabs/lib/BEASTlabs.v2.0.3.jar:\
$HOME/mosse/beast-mosse/lib/jblas-1.2.5.jar"
rm -rf build/classes && mkdir -p build/classes
find src -name "*.java" > /tmp/mosse-sources.list
javac -d build/classes -cp "$CP" --release 17 @/tmp/mosse-sources.list
jar cf build/mosse.jar -C build/classes mosse
```

The jar is portable across machines (same JVM version), so this only needs
re-doing after a `.java` change.

---

## When you move to a different machine

Update three things:

1. **Rebuild `libtest.so`** with the right `-march=<arch>` flag (see table above).
2. **Update `threads="N"` in the XML** to the new machine's physical-core count.
3. **Update `taskset -c 0-(N-1)`** in the launcher to match.

The Java jar does not need rebuilding (unless the JVM version differs).

Find the physical-core count with:
```bash
lscpu | grep -E "^Core\(s\) per socket|^Socket\(s\)"
# physical cores = core-per-socket × sockets
```

Find which CPU IDs are "first SMT sibling per core" with:
```bash
for c in $(ls -d /sys/devices/system/cpu/cpu[0-9]*); do
  cat $c/topology/thread_siblings_list
done | awk -F, '{print $1}' | sort -un | paste -sd,
```
The output is the `taskset -c <list>` argument.

---

## Verifying the run is using all your cores

While the MCMC is running, in another shell:

```bash
ps -o pid,etime,pcpu,nlwp -C java
```

`%CPU` should be roughly `100 × physical-core-count` (≈12000–12500% on
rona with threads=128). If you see only ~3000–5000%, something is wrong:

- Did you forget `taskset`?
- Did the XML still have `threads="52"` from an old example?
- Did `taskset -c 0-127` get a typo (e.g. `0–127` with an en-dash)?

You can confirm parallelism on a 10-step chain (~30 s) by running with
`chainLength="10"` and `time -v` instead of the full chain, then checking the
"Percent of CPU this job got" line in the output.

---

## Resume after stopping

BEAST2 writes the current state every 5000 steps (or whatever your log
interval) to `<filebase>.xml.state`. To pick up where you left off, run the
exact same XML with `-resume` instead of `-overwrite`:

```bash
taskset -c 0-127 java ... \
  beastfx.app.beast.BeastMCMC -seed 42 -resume your.xml
```

**Note:** if you change operator step sizes in the XML between runs and want
the new values used, you may also need to edit the `"p":...` value for that
operator in the JSON block at the bottom of `<filebase>.xml.state`. BEAST
restores per-operator tuning parameters from the state file, so the XML's new
value is otherwise ignored.

---

## A short menu of further speedups

If you ever need to squeeze more out, in rough ROI order:

1. **Subpattern batching** — biggest remaining architectural win. Currently
   each (branch × subpattern) is one JNI call; batching K subpatterns per call
   would amortise the JNI overhead K× and let FFTW reuse twiddle factors.
   Expected 2–5× on the JNI-dominated portion. High effort (changes JNI
   signature and the C-side integrator structure).
2. **FFTW_MEASURE + wisdom file** — currently FFTW plans with the cheapest
   `FFTW_DESTROY_INPUT` (estimate) strategy. Switching to `FFTW_MEASURE` and
   saving wisdom across runs is one-time-cost-for-permanent-gain. Expected
   5–30% on FFT-bound work.
3. **Coupled MCMC (MC³)** — run multiple chains with state-swaps. Same total
   CPU, but much better mixing for multi-modal posteriors. Use BEAST2's
   `CoupledMCMC` package.
4. **Reduce `nx` from 1024 → 512** — halves FFT size → roughly halves wall
   time. Pure trade-off vs numerical bin error; needs a short A/B chain to
   confirm acceptable accuracy.
5. **Partial-likelihood caching (BEAGLE-style)** — recompute only branches
   touched by each operator. Largest payoff for trees with topology-changing
   operators, smaller for the parameter-scaling operators this XML uses
   (~1.5×). Very high effort.

---

## Optimisations already applied (for reference)

All of these are in the source tree and need no per-run action:

| # | What | Where |
|--|--|--|
| Tree-level parallelism | Sibling subtrees compute in parallel | `MosseTreeLikelihood.TraverseTask`, `traverseFull` |
| Per-node subpattern parallelism | `IntStream.parallel` over subpatterns | `MosseTreeLikelihood.computePartialLikelihood` |
| Flat-traversal parallelism (#14) | `doFlatTraversal` also runs on the pool | `MosseTreeLikelihood.FlatTraverseTask`, `computeFlatTreeLogLikelihood` |
| JNI reusable buffers | No per-call malloc/copy for lambda/mu/r/vars | `jni/main.c` `lambda_buf`/`mu_buf`/`r_buf`/`vars_buf` |
| Kernel-FFT cache | Skip `qf_setup_kern_mosse` when inputs unchanged | `jni/main.c` `kern_valid` |
| Drop redundant copy (#15) | Return JNI array directly | `MosseDistribution.calculateBranchLogP` |
| Eigendata cache (#16) | Skip per-call eigendata copy via generation counter | `jni/main.c` `eigen_generation`, Java `eigenGeneration` |
| Per-thread native plans | No FFTW contention | `MosseDistribution.ptr_l_pool`, `ptr_h_pool` |
| Common-pool cap | `-Djava.util.concurrent.ForkJoinPool.common.parallelism=1` | Launcher (above) |

Per-run choices that *do* need to be set in the launcher / XML:

| # | Setting | Where |
|--|--|--|
| Pinning + thread count (#17) | `taskset -c 0-127` + XML `threads="128"` | Launcher + XML |
| AVX2/FMA codegen (#21) | `gcc -O3 -march=znver2 -funroll-loops` | JNI build step |
