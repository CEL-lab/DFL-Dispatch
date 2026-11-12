#!/bin/bash
# Submit one three-model comparison job per seed (5 parallel jobs).
for SEED in 42 123 456 789 1024; do
    echo "Submitting comparison SEED=$SEED ..."
    qsub -v SEED=$SEED run_compare.pbs
done
echo "All 5 comparison jobs submitted. Check with: qstat -u $USER"
