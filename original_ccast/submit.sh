#!/bin/bash
# Usage (from Embedded_model_01/):   bash submit.sh <stage>
#   smoke    1 short test job (~10 min). Run this FIRST.
#   base     5 seeds, all models, alpha 0.80            <- run after smoke passes
#   alpha    alpha grid 0.5/0.7/0.9/0.95, ICO+ASYM, 5 seeds (for validation-based alpha)
#   s2       demand-volatility scenario, 5 seeds
#   s3       tight-capacity scenario (calibrated on train+val), 5 seeds
#   rolling  test years 2016/2017/2018, all models, 5 seeds
set -e
cd "$(dirname "$0")"
mkdir -p logs
SEEDS="42 123 456 789 1024"
stage=${1:?usage: bash submit.sh smoke|base|alpha|s2|s3|rolling}

sub() {  # $1 = job name, $2 = variables for run_experiment.py, $3 = extra qsub options
  qsub -N "$1" -o "logs/$1.log" $3 -v "$2" run_v2.pbs
}

case $stage in
  smoke)
    sub smoke "SEED=42,SCENARIO=base,ALPHA=0.8,MAX_EPOCHS=7,TRAIN_SUBSAMPLE=40,SAVE_DIR=$PWD/results/smoke" \
        "-l walltime=00:45:00" ;;
  base)
    for s in $SEEDS; do sub "base_$s" "SEED=$s,SCENARIO=base,ALPHA=0.8"; done ;;
  alpha)
    for a in 0.5 0.7 0.9 0.95; do for s in $SEEDS; do
      sub "a${a/./}_$s" "SEED=$s,SCENARIO=base,ALPHA=$a,MODELS=ICO+ASYM"; done; done ;;
  s2)
    for s in $SEEDS; do sub "s2_$s" "SEED=$s,SCENARIO=s2,ALPHA=0.8"; done ;;
  s3)
    for s in $SEEDS; do sub "s3_$s" "SEED=$s,SCENARIO=s3,ALPHA=0.8"; done ;;
  rolling)
    for yr in 2016 2017 2018; do for s in $SEEDS; do
      sub "y${yr}_$s" "SEED=$s,SCENARIO=base,ALPHA=0.8,TEST_YEAR=$yr"; done; done ;;
  *) echo "unknown stage: $stage"; exit 1 ;;
esac
echo "submitted stage '$stage'. Check with: qstat -u \$USER"
