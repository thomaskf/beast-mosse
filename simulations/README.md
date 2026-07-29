# MoSSE MCMC recovery experiment — reproducible pipeline

This folder reproduces the MoSSE parameter-recovery experiment end-to-end: simulate
data under a known MoSSE model, then fit it with `beast-mosse` and check the true
parameters are recovered.

The generative model (no punctuation, no across-site rate variation) is a Brownian
motion **on the log substitution rate**, with trait-dependent speciation and a
tip GLM. Ten independent datasets are drawn from one shared HKY model.

## Quick start (stages 1–6, local)
```bash
cd simulations
BEAST=/path/to/beast ./run_experiment.sh
```
This produces `xml_generated/sim8_seq1_0.2_<seed>.xml` (10 files), ready to run.
It is **idempotent** — finished stages are skipped; delete their outputs or set a
`FORCE_*` env var (`FORCE_SIM`, `FORCE_STRICT`, `FORCE_BD`) to redo.

A ready-made single example is in `../examples/sim8_seq1_0.2_43.xml` (the seed-43 run).

## Layout
```
simulations/
  run_experiment.sh      one-command driver (stages 1-6)
  submit_mcmc.pbs        PBS array template for a cluster (stage 7)
  sim/                   R simulation (diversitree-based)
    run_sim.R  main.nopunc.R  tree.mosse.withpunc.R  simSeq.mosse.withpunc.R  prune.mosse.R
  scripts/               pipeline (python + R)
    generate_strict_clock_xml.py  extract_strict_clock.py
    compute_bd.R  generate_mosse_log_xml.py  summarize_dataset.py  generate_xml.py
  xml_files/             the 10 ready-to-run mosse MCMC XMLs (nx=1024, 1M, data-driven inits)
  mcmc_results/          (added once the runs finish) logs, MCC trees, result summaries
```

## The stages
| # | stage | script | output |
|---|---|---|---|
| 1 | simulate 10 datasets (once; kappa + base freqs shared) | `sim/run_sim.R` | `sim/seq/tree/trait1_0.2_<s>.*`, `sim_truth.csv` |
| 2 | strict-clock BEAST XML (per seed) | `generate_strict_clock_xml.py` | `strict_clock/seed_<s>/strict_<s>.xml` |
| 3 | run strict-clock BEAST (per seed, plain HKY) | `beast -seed <s> strict_<s>.xml` | `strict_<s>.log/.trees` |
| 4 | summarise → clockRate/kappa + init tree | `extract_strict_clock.py --tree-mode last` | `summary_<s>.json` |
| 5 | birth-death fit on true trees (once) | `compute_bd.R` | `bd_estimates.json` |
| 6 | generate the mosse MCMC XML (per seed) | `generate_mosse_log_xml.py` | `sim8_seq1_0.2_<s>.xml` |

The mosse XML's **initial values are data-driven** (subst = log of the strict-clock
clockRate median; curve/extinction params from the birth-death fit), so the chain
starts near a data-informed point — fast burn-in without hard-coding the truth.

## The mosse MCMC config produced
- **Log-scale tip model** (`logscale="true"`; traits fed as `log(trait)`), matching the
  log-rate simulation.
- **HKY** with base frequencies fixed to the observed values; **no across-site gamma**
  (matches the simulation).
- **Fixed topology** (root pinned near truth); **`nx=1024`** grid, `chainLength=1,000,000`.
- Priors chosen so each **brackets its true value** (e.g. subst Normal(−3, 2), epsilon
  Exp(0.1), diffusion Exp(0.1)).
- **13 operators**: one per parameter, plus a `LinearGrowthRate↔diffusion` up-down ridge
  move (`LGRDiffusionUp`) and a de-weighted 6-parameter `UpDownGroups`.

## True parameters (from `sim/main.nopunc.R`, recorded in `sim_truth.csv`)
| param | truth | | param | truth |
|---|---|---|---|---|
| subst (x0, log-rate) | −3.912 = log(0.02) | | epsilon (tip-GLM sd) | 0.2 |
| drift (BM on log-rate) | −0.2 | | beta (tip-GLM slope) | 1 |
| diffusion (BM **variance**) | 0.2 | | kappa (HKY) | 5.49 |
| linearGrowthRate (λ_r) | 30 | | mu / yV (extinction) | 0.05 |
| curveMaxY (λ cap) | 3 | | gamma / punctuation | none / off |

Speciation link: `λ(x) = min(30·exp(x), 3)`. `diffusion` is a **variance rate**
(the per-step sd is `sqrt(dt·diffusion)`), so the value to recover is **0.2**, not 0.04.
To change the truth, edit `sim/main.nopunc.R` and re-run.

## Submit on a cluster (stage 7)
Not run by `run_experiment.sh` (it runs on the cluster). After stages 1–6:
```bash
rsync -a xml_generated/ submit_mcmc.pbs  user@cluster:/path/to/rundir/
# on the cluster: edit RUNDIR/SEEDS/TAG in submit_mcmc.pbs, then
qsub submit_mcmc.pbs        # PBS array 1-10
```
The mosse runs need the `beast-mosse` package plus its native library on the load path
(FFTW3, GSL): `module load beast fftw3 gsl` and
`LD_LIBRARY_PATH=$HOME/.beast/2.7/beast-mosse/lib`. Each job checkpoints; resubmit with
`RESUME="-resume"` in the PBS to accumulate more states.

## Dependencies
- **R packages** — install once:
  ```r
  # diversitree: use the MoSSE FORK, not the CRAN version — this is the environment
  # the experiment was built and verified against.
  remotes::install_github("thomaskf/diversitree-mosse")
  install.packages(c("phangorn", "LaplacesDemon", "ape", "jsonlite"))
  ```
  The simulation itself calls only diversitree's QuaSSE primitives (`make.brownian.with.drift`,
  `constant.x`, `me.to.ape.quasse`), which also exist in stock CRAN diversitree — so it may run on
  CRAN too. But `thomaskf/diversitree-mosse` (github.com/thomaskf/diversitree-mosse) is the
  verified build, so use it for reproducibility.
- **Python 3:** standard library only.
- **BEAST 2.7:** plain BEAST for the strict-clock runs (stage 3); the `beast-mosse`
  package + FFTW/GSL native lib for the mosse runs (stage 7).
- `xmllint`.

## Notes
- `extract_strict_clock.py` defaults to `--tree-mode mcc` (needs `treeannotator`); the
  driver uses `--tree-mode last` to avoid that dependency.
- One seed (83) simulates 19 taxa (vs 20) — handled automatically per seed.
- `generate_xml.py` is present only as a shared helper module (tree/newick utilities)
  imported by the other generators; the mosse XMLs are produced by
  `generate_mosse_log_xml.py`.
