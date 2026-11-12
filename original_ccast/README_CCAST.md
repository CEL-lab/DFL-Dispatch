# Running the corrected experiments on CCAST

## Files
| File | What it is |
|---|---|
| `run_experiment.py` | Replaces `ico_train_single_seed.py`, `three_model_comparison.py`, `scenario_s2.py`, `scenario_s3.py`. One run trains PtO, ICO, ASYM, fits LS, computes the validation offset, and evaluates all of them. |
| `run_v2.pbs` | PBS job for one run. |
| `submit.sh` | Submits a whole stage (all seeds) with one command. |

Old scripts and old results stay untouched. New results go to `results/v2/`.

## Step 1: copy the files
Put the three files in `/mmfs1/projects/harun.pirim/Faria/Embedded_model_01/`, the folder that contains `data/modified_dataset.csv` and `data/weather_features.csv`.

## Step 2: nothing to edit
`run_v2.pbs` already uses the settings from your `run_ico.pbs`: queue `gpus`, 16 CPUs, 64 GB, 1 GPU,
14 h walltime, group `x-ccast-prj-hpirim`, modules `python/3.12.2` and `cuda11.8`, and the `e2e` venv.
Email goes to faria.farzana@ndsu.edu only when a job ends or aborts (`-m ae`), not at every start,
because the later stages submit up to 20 jobs at once.

## Step 3: smoke test (about 10 min)
```bash
cd /mmfs1/projects/harun.pirim/Faria/Embedded_model_01
bash submit.sh smoke
qstat -u $USER            # wait until it disappears
tail -15 logs/smoke.log
```
It should end with a table of PtO / ICO / ASYM / LS / OFFSET and `exit code 0`.
The numbers are meaningless (7 epochs on 1/40 of the data). Only check that it ran.
If it fails, send me `logs/smoke.log`.

## Step 4: base runs (5 jobs, about 3 h each)
```bash
bash submit.sh base
```
When they finish, send me the five files
`results/v2/base_a0.8/seed{42,123,456,789,1024}/results.json`.

## Later stages (only after I check the base results)
```bash
bash submit.sh alpha     # 20 jobs, alpha chosen on validation
bash submit.sh s2        # 5 jobs
bash submit.sh s3        # 5 jobs
bash submit.sh rolling   # 15 jobs
```

## Useful commands
```bash
qstat -u $USER                     # running / queued jobs
tail -f logs/base_42.log           # watch one job
grep -l "exit code 0" logs/*.log   # finished jobs
```
