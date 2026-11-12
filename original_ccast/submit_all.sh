#!/bin/bash
# =====================================================================
# submit_all.sh  —  launch every ICO experiment as parallel PBS jobs.
#
# Usage:
#   bash submit_all.sh          # submit everything
#   bash submit_all.sh core     # submit only the core set (7 jobs)
#
# All jobs are independent, so the scheduler runs as many at once as your
# allocation allows — wall-clock time is roughly ONE job (~2-4 h), not the sum.
# Results auto-route to results/ico_model/<tag>/ so nothing collides.
#
# NOTE: run the base smoke-test first (qsub -v SEED=42 run_ico.pbs) and confirm
# it reproduces your paper numbers before launching the full batch.
# =====================================================================

set -u
SEEDS="42 123 456 789 1024"
MODE="${1:-full}"          # "full" (default) or "core"

n=0
submit () {                # submit "$@" as qsub -v args, count it, echo it
    echo "  qsub -v $1 run_ico.pbs"
    qsub -v "$1" run_ico.pbs
    n=$((n+1))
}

echo "=============================================="
echo " Submitting ICO jobs  (mode: $MODE)"
echo "=============================================="

# ── Main results: base 80/10/10 split, all seeds ──
echo "[1] Base split (main results):"
for S in $SEEDS; do
    submit "SEED=$S"
done

# ── Test-next-year study (train past years, test the next full year) ──
echo "[2] Test-next-year folds:"
if [ "$MODE" = "core" ]; then
    # core: one seed, all three folds
    for TY in 2016 2017 2018; do
        submit "SEED=42,TEST_YEAR=$TY"
    done
else
    # full: all seeds x three folds
    for S in $SEEDS; do
        for TY in 2016 2017 2018; do
            submit "SEED=$S,TEST_YEAR=$TY"
        done
    done
fi

# ── Capacity-constant sensitivity: seed 42 only ──
if [ "$MODE" != "core" ]; then
    echo "[3] Capacity sensitivity (headroom x floor, seed 42):"
    for HR in 1.1 1.2 1.3; do
        for FL in 0.02 0.05 0.1; do
            submit "SEED=42,HEADROOM=$HR,FLOOR=$FL"
        done
    done
fi

echo "=============================================="
echo " Submitted $n jobs."
echo " Track them with:  qstat -u faria.farzana"
echo "=============================================="
