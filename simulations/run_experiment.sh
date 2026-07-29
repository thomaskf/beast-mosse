#!/bin/bash
# =============================================================================
# run_experiment.sh — reproducible MoSSE MCMC recovery experiment
#
# Chains, end-to-end, to produce the 10 mosse MCMC XMLs:
#   STAGE 1  simulate 10 no-punctuation MoSSE datasets      (R / diversitree)
#   STAGE 2  per seed: build a strict-clock BEAST XML       (generate_strict_clock_xml.py)
#   STAGE 3  per seed: run the strict-clock BEAST chain      (BEAST 2.7, plain HKY)
#   STAGE 4  per seed: summarise -> clockRate/kappa + tree   (extract_strict_clock.py)
#   STAGE 5  birth-death fit on the true trees -> lambda/mu  (compute_bd.R)
#   STAGE 6  per seed: generate the mosse MCMC XML           (generate_mosse_log_xml.py)
#
# Output: xml_generated/sim8_seq1_<TAG>_<seed>.xml (10 files), ready to submit.
# NCI/Gadi array submission is a SEPARATE step (submit_mcmc.pbs; see README.md).
#
# Idempotent (completed stages skipped; delete outputs or set FORCE_*=1 to redo).
# Requires: R (diversitree, phangorn, ape, LaplacesDemon, jsonlite), python3,
#           BEAST 2.7 on PATH (var BEAST) for the strict-clock runs, xmllint.
# =============================================================================
set -euo pipefail

# ------------------------- CONFIG (override via env) -------------------------
HERE="$(cd "$(dirname "$0")" && pwd)"                   # simulations/
SIM="${SIM:-$HERE/sim}"                                 # R simulation scripts + outputs
SCR="${SCR:-$HERE/scripts}"                             # pipeline scripts
OUTDIR="${OUTDIR:-$HERE/xml_generated}"                 # where the mosse XMLs land
TAG="${TAG:-0.2}"                                       # epsilon/sig tag in filenames
read -r -a SEEDS <<< "${SEEDS:-10 30 35 38 43 44 67 70 82 83}"  # inner seeds (from run_sim.R)
BEAST="${BEAST:-beast}"                                 # BEAST 2.7 binary (strict clock)
STRICT_CHAIN="${STRICT_CHAIN:-500000}"
MOSSE_CHAIN="${MOSSE_CHAIN:-1000000}"
# -----------------------------------------------------------------------------

mkdir -p "$OUTDIR" "$SIM/strict_clock"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# ---- STAGE 1: simulate (ONCE; kappa + base freqs are shared across all seeds) ----
if [[ ! -f "$SIM/sim_truth.csv" || "${FORCE_SIM:-0}" == "1" ]]; then
  log "STAGE 1  simulate datasets"; Rscript "$SIM/run_sim.R"
else
  log "STAGE 1  skip (sim_truth.csv exists)"
fi

# ---- STAGE 2-4: per-seed strict clock ----
for s in "${SEEDS[@]}"; do
  rd="$SIM/strict_clock/seed_$s"; mkdir -p "$rd"; sm="$rd/summary_$s.json"
  if [[ -f "$sm" && "${FORCE_STRICT:-0}" != "1" ]]; then
    log "STAGE 2-4 seed $s  skip (summary exists)"; continue
  fi
  log "STAGE 2  strict-clock XML  seed $s"
  python3 "$SCR/generate_strict_clock_xml.py" \
      "$SIM/seq1_${TAG}_$s.fasta" "$SIM/tree1_${TAG}_$s.tre" \
      "$rd/strict_$s.xml" --chain "$STRICT_CHAIN"
  log "STAGE 3  run strict-clock BEAST  seed $s   ($BEAST)"
  ( cd "$rd" && "$BEAST" -seed "$s" -overwrite "strict_$s.xml" > "beast_$s.out" 2>&1 )
  log "STAGE 4  extract summary  seed $s"
  python3 "$SCR/extract_strict_clock.py" "$rd" --tree-mode last --out "$sm"
done

# ---- STAGE 5: birth-death fit (ONCE, all seeds) ----
if [[ ! -f "$SIM/bd_estimates.json" || "${FORCE_BD:-0}" == "1" ]]; then
  log "STAGE 5  birth-death fit"; Rscript "$SCR/compute_bd.R" "$SIM" "$SIM/bd_estimates.json"
else
  log "STAGE 5  skip (bd_estimates.json exists)"
fi

# ---- STAGE 6: generate the mosse MCMC XML per seed ----
for s in "${SEEDS[@]}"; do
  out="$OUTDIR/sim8_seq1_${TAG}_$s.xml"
  log "STAGE 6  mosse XML  seed $s"
  python3 "$SCR/generate_mosse_log_xml.py" \
      "$SIM/seq1_${TAG}_$s.fasta" "$SIM/tree1_${TAG}_$s.tre" "$SIM/trait1_${TAG}_$s.csv" \
      "$SIM/strict_clock/seed_$s/summary_$s.json" "$SIM/bd_estimates.json" \
      "$out" --chain "$MOSSE_CHAIN"
  xmllint --noout "$out"
done

log "DONE.  $(ls "$OUTDIR"/sim8_seq1_${TAG}_*.xml 2>/dev/null | wc -l | tr -d ' ') mosse XMLs in $OUTDIR/"
echo "Next: submit on a cluster with submit_mcmc.pbs (see README.md)."
