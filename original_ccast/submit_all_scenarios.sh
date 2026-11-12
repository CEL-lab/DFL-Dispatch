#!/bin/bash
# Submit the three scenarios x 2 seeds.
#   S1 = economic re-pricing at EUR 40,000 (reuses baseline forecasts, ~5 min/job)
#   S2/S3 = real retraining stresses (~100 min/job each)
# S1 needs results/comparison/pred_*_seed*.npy from the main comparison run.
# Two most representative seeds (their mean matches the 5-seed mean;
# 123 = weakest ICO advantage, 456 = strongest, so the pair is not cherry-picked).
# Add more seeds here later if a reviewer asks.
SEEDS="123 456"
for SC in S1 S2 S3; do
  for SEED in $SEEDS; do
    echo "Submitting $SC seed $SEED ..."
    qsub -v SCENARIO=$SC,SEED=$SEED run_scenario.pbs
  done
done
echo "Submitted. Track with: qstat -u $USER"
