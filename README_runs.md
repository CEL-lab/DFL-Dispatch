# Running the corrected experiments

Data files (`modified_dataset.csv`, `weather_features.csv`) sit in this folder.
Package versions used for the smoke test: `environment_versions.txt`.
Original (pre-audit) runner kept as `run_experiment_v2_original.py`; fixes F1-F16 are listed in the header of `run_experiment.py`.

## Smoke test (minutes, CPU; results are NOT for the paper)
    TRAIN_SUBSAMPLE=20 MAX_EPOCHS=8 SAVE_DIR=results/smoke SEED=42 python run_experiment.py

## Full base run, one seed (ICO ~2 h on GPU node, ~several h on CPU)
    SEED=42 SCENARIO=base ALPHA=0.80 python run_experiment.py

## Five seeds (one job each; use + not , in MODELS when passing through qsub -v)
    for s in 42 123 456 789 1024; do SEED=$s SCENARIO=base python run_experiment.py; done

## Alpha sweep (selection must use the validation metrics in results.json, not test)
    for a in 0.0 0.5 0.8 0.9 1.0; do SEED=42 ALPHA=$a python run_experiment.py; done

## Stress scenarios (S3 scale is calibrated on train+validation)
    SEED=42 SCENARIO=s2 python run_experiment.py
    SEED=42 SCENARIO=s3 python run_experiment.py

## Rolling origin (test = full calendar year; validation = last 10% of prior hours)
    SEED=42 TEST_YEAR=2016 python run_experiment.py   # likewise 2017, 2018

## Capacity grid
    SEED=42 HEADROOM=1.1 FLOOR=0.02 python run_experiment.py   # etc.

## Optional: ECOS inside the training layer (original runs used the diffcp default, SCS)
    QP_SOLVER=ECOS SEED=42 python run_experiment.py

Outputs per job: `results/v2/<tag>/seed<S>/results.json` (config, split timestamps, checks,
environment, per-model train/val/test metrics incl. slack hours, identity residual, solver
statuses, per-source dispatch), `hourly_outputs.npz` (val/test predictions, dispatch,
shortfall, slack, capacities), `history_<model>.npy`, `weights_<model>.pt`.

## On CCAST (Prime)
Working copy of the patched runner and data: `/mmfs1/scratch/harun.pirim/ico_v2/code-data/`
(venv reused from `/mmfs1/projects/harun.pirim/Faria/Embedded_model_01/e2e`; GPU nodes are NVIDIA L4).
    ssh prime
    cd /mmfs1/scratch/harun.pirim/ico_v2/code-data
    bash submit_v3.sh smoke      # then: base | alpha | s2 | s3 | rolling | capgrid | avail
    # avail = AVAIL=physical (solar ceiling from t-24, zero at night) and AVAIL=realized (ex-post bound), 5 seeds each
    qstat -u $USER               # logs/<job>.log is written when a job ends
Outputs: results/v3/<tag>/seed<S>/.  Aggregate locally with
    python aggregate.py <results_root> <out_prefix>     # writes <out_prefix>_summary.md/.csv
Original (pre-audit) scripts: original_ccast/.  Faria's 26-Sep v2 results: ccast_results_v2/ (summary in v2_summary.md).

## Scripts
    make_dataset.py   Kaggle energy_dataset.csv -> modified_dataset.csv (documents the transformation; --compare diffs against the file used)
    aggregate.py      results root -> five-seed tables + block-bootstrap intervals (<prefix>_summary.md/.csv)
    make_tables.py    aggregate prefix -> ../tables/*.tex + numbers.tex macros for Main.tex (--avail physical|proxy picks the sweep source)
    figures.py        results root -> fig_training, fig_comparison, fig_alpha_frontier, fig_conditions, fig_dispatch (PDF+PNG)
    Example:  python aggregate.py ccast_results_v3/results/v3 ccast_results_v3/v3
              python figures.py ccast_results_v3/results/v3 ccast_results_v3/figures --data modified_dataset.csv
## Runner options added 1 Oct 2026
    AVAIL=proxy|physical|realized   availability model for the variable renewables (F17; see header of run_experiment.py)
