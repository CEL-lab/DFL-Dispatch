#!/bin/bash
# =====================================================================
# submit_sensitivity.sh — capacity-constant sensitivity sweep (Dr. Le, pt. 2)
#
# Varies the renewable HEADROOM factor (base 1.2) and FLOOR fraction (base 0.05)
# up and down, retraining ICO for each combination at seed 42.
#   HEADROOM in {1.1, 1.2, 1.3}   x   FLOOR in {0.02, 0.05, 0.10}  = 9 runs
#
# The centre cell (1.2, 0.05) is the base case; it has no tag and lands in the
# main results/ico_model/ folder (you already have it from the seed-42 base run),
# so this script skips it and submits the other 8.
#
# Usage:  bash submit_sensitivity.sh
# Watch:  qstat -u faria.farzana
# =====================================================================

set -u
SEED=42
n=0

echo "Submitting capacity sensitivity sweep (seed $SEED)..."
for HR in 1.1 1.2 1.3; do
  for FL in 0.02 0.05 0.1; do
    if [ "$HR" = "1.2" ] && [ "$FL" = "0.05" ]; then
      echo "  skip base cell HEADROOM=1.2 FLOOR=0.05 (already have it in results/ico_model/)"
      continue
    fi
    echo "  qsub -v SEED=$SEED,HEADROOM=$HR,FLOOR=$FL run_ico.pbs"
    qsub -v "SEED=$SEED,HEADROOM=$HR,FLOOR=$FL" run_ico.pbs
    n=$((n+1))
  done
done

echo "Submitted $n sensitivity jobs (base cell reused, not resubmitted)."
echo "Track with:  qstat -u faria.farzana"
