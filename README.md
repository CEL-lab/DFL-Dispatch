# DFL-Dispatch

Code and summary results for the paper

> **Decision-focused load forecasting for economic dispatch: when does an embedded optimization layer matter?**
> F. Farzana, H. Pirim, C. Le, O. P. Yadav, T. Le. *Submitted to the European Journal of Operational Research.*

A neural load forecaster is trained on a composite of forecast error and the operational cost of the economic
dispatch its forecast induces. Five models share the same network, features, chronological split and seeds:

| Model | Training objective | Dispatch cost computed by |
|---|---|---|
| **PtO** | mean squared error | – |
| **ICO** | MSE + operational cost | a differentiable QP layer (cvxpylayers) |
| **ASYM** | the same composite loss as ICO | the closed form of Proposition 1 (no solver) |
| **OFFSET** | PtO + a constant offset chosen on validation | – |
| **LS** | ordinary least squares | – |

Every forecast is scored through the same per-hour dispatch QP on the test set.

## Repository contents

| Path | What it is |
|---|---|
| `make_dataset.py` | Builds `modified_dataset.csv` from the raw Kaggle file |
| `run_experiment.py` | One run = one seed × scenario × loss weight; trains and evaluates all five models. Fixes F1–F18 are listed in its header |
| `aggregate.py` | Five-seed summary tables and block-bootstrap intervals |
| `make_tables.py` | LaTeX tables and the `numbers.tex` macros used in the manuscript |
| `figures.py` | Figures of the paper (PDF and PNG) |
| `check_layer_vs_closed.py` | Compares the QP layer's backward values with the closed form near the kinks (Appendix A) |
| `run_v3.pbs`, `submit_v3.sh`, `README_runs.md` | Batch jobs and run notes for the NDSU CCAST cluster |
| `environment_versions.txt` | Package versions |
| `ccast_results_v3/` | Results of the 132 runs reported in the paper: `results.json` and training histories per run, five-seed summaries (`v3_summary.*`), bootstrap intervals (`v3_bootstrap.csv`) and figures |
| `original_ccast/`, `run_experiment_v2_original.py` | The preliminary pipeline, kept for the record (see "Corrections relative to the preliminary study" in the paper). Not used for any reported number |

Not included: the data (see below), and the per-run hourly outputs (`hourly_outputs.npz`) and model weights
(`weights_*.pt`), about 400 MB in total. *[Add the Zenodo DOI here once archived.]*

## Data

The data are public and are not redistributed here. Download
**Hourly energy demand generation and weather** (Jhana, 2019) from
[Kaggle](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather)
and place `energy_dataset.csv` and `weather_features.csv` in the repository folder. Then build the cleaned energy file:

```bash
python make_dataset.py energy_dataset.csv --out modified_dataset.csv
```

This keeps the 13 generation columns and total load and converts time to UTC. Missing values in 44 hours (total
load in 34 of them, generation in 23) are filled with the full-sample mean for the same UTC weekday and hour, and
three hours (2017-11-12 20:00, 2017-11-14 11:00 and 18:00 UTC) are dropped. The result, 35,061 hourly rows,
is identical to the file used for the paper.

## Environment

Python 3.12. The reported runs used torch 2.6.0, cvxpy 1.8.2, cvxpylayers 1.0.4, diffcp 1.1.8, scs 3.2.11,
ecos 2.0.14, numpy, pandas, scikit-learn, scipy and matplotlib (see `environment_versions.txt`).

```bash
pip install -r requirements.txt
```

A GPU is optional; the QP layer runs on the CPU. One ICO training run takes about 1.5 to 3 hours.

## Reproducing the results

**1. Quick check (minutes, CPU; not for the paper):**
```bash
TRAIN_SUBSAMPLE=20 MAX_EPOCHS=8 SAVE_DIR=results/smoke SEED=42 python run_experiment.py
```

**2. The runs in the paper.** Each run writes `results/<scenario tag>/seed<S>/results.json`. Main configuration,
five seeds, physical availability ceiling:
```bash
for s in 42 123 456 789 1024; do
  SEED=$s SCENARIO=base ALPHA=0.8 AVAIL=physical python run_experiment.py
done
```
Loss-weight sweep (`ALPHA=0 0.5 0.7 0.9 0.95`), stress scenarios (`SCENARIO=s2`, `SCENARIO=s3`), rolling origin
(`TEST_YEAR=2016 2017 2018`), capacity grid (`HEADROOM`, `FLOOR`) and availability models
(`AVAIL=proxy|physical|realized`) use the same command with the variables shown. `README_runs.md` and
`submit_v3.sh` list every configuration.

**3. Tables and figures from the included results** (no training needed):
```bash
python make_tables.py ccast_results_v3/v3 tables          # LaTeX tables + numbers.tex
python figures.py ccast_results_v3/results/v3 figures     # training, comparison, alpha-frontier, conditions figures
```
To check the summaries, rerun the aggregation into a new prefix; it reproduces `ccast_results_v3/v3_summary.csv`
exactly from the included `results.json` files:
```bash
mkdir -p check
python aggregate.py ccast_results_v3/results/v3 check/v3
```
Do not write to the prefix `ccast_results_v3/v3`: the block-bootstrap intervals (`v3_bootstrap.csv`) and the
dispatch figure need the hourly outputs, which are not included, so rerunning there would replace the stored
intervals with an empty file.

## License

*[To be decided by the authors.]*

## Citation

*[Add after publication.]*
