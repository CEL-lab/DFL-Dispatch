#!/bin/bash
# Submit one single-seed ICO training job per seed (5 parallel jobs).
# Usage: bash submit_all_seeds.sh
for SEED in 123 456 789 1024; do
    echo "Submitting SEED=$SEED ..."
    qsub -v SEED=$SEED run_ico.pbs
done
echo "All 5 seeds submitted. Check status with: qstat -u $USER"
