#!/bin/bash
# Usage (on CCAST, from /mmfs1/scratch/harun.pirim/ico_v2/code-data):  bash submit_v3.sh <stage>
set -e
cd "$(dirname "$0")"; mkdir -p logs
SEEDS="42 123 456 789 1024"
stage=${1:?usage: bash submit_v3.sh smoke|base|alpha|s2|s3|rolling|capgrid|avail}
AV=${AVAIL:-proxy}; PFX=""; [ "$AV" != "proxy" ] && PFX="${AV:0:1}_"      # AVAIL=physical bash submit_v3.sh alpha  -> jobs p_a05_42 ...
sub() { qsub -N "${PFX}$1" -o "logs/${PFX}$1.log" $3 -v "$2,AVAIL=$AV" run_v3.pbs; }
case $stage in
  smoke)   sub smoke "SEED=42,SCENARIO=base,ALPHA=0.8,MAX_EPOCHS=7,TRAIN_SUBSAMPLE=40,SAVE_DIR=$PWD/results/smoke" "-l walltime=00:45:00" ;;
  base)    for s in $SEEDS; do sub "base_$s" "SEED=$s,SCENARIO=base,ALPHA=0.8"; done ;;
  alpha)   for a in 0.0 0.5 0.7 0.9 0.95; do for s in $SEEDS; do sub "a${a/./}_$s" "SEED=$s,SCENARIO=base,ALPHA=$a,MODELS=ICO+ASYM"; done; done ;;
  s2)      for s in $SEEDS; do sub "s2_$s" "SEED=$s,SCENARIO=s2,ALPHA=0.8"; done ;;
  s3)      for s in $SEEDS; do sub "s3_$s" "SEED=$s,SCENARIO=s3,ALPHA=0.8"; done ;;
  rolling) for yr in 2016 2017 2018; do for s in $SEEDS; do sub "y${yr}_$s" "SEED=$s,SCENARIO=base,ALPHA=0.8,TEST_YEAR=$yr"; done; done ;;
  capgrid) for h in 1.1 1.2 1.3; do for f in 0.02 0.05 0.10; do [ "$h$f" = "1.20.05" ] && continue; sub "cap_h${h}_f${f}" "SEED=42,SCENARIO=base,ALPHA=0.8,HEADROOM=$h,FLOOR=$f,MODELS=ICO+ASYM"; done; done ;;
  base42)  sub base_42 "SEED=42,SCENARIO=base,ALPHA=0.8" ;;   # re-run after the capgrid default combination overwrote it
  avail)   for av in physical realized; do for s in $SEEDS; do sub "av${av}_$s" "SEED=$s,SCENARIO=base,ALPHA=0.8,AVAIL=$av"; done; done ;;
  *) echo "unknown stage: $stage"; exit 1 ;;
esac
echo "submitted stage '$stage'. Check with: qstat -u \$USER"
