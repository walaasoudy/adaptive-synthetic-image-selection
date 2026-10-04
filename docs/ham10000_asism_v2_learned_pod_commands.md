# ASISM v2 (learned ranker and stopping rule) on RunPod: the command order

**STATUS, 2026-10-04: NO GPU STEP IN THIS FILE IS APPROVED. DO NOT RUN ANYTHING BELOW YET.**

This file is the order of commands for after Walaa approves the utility measurement. It is written
before the approval so that the order can be read and checked first. Two things are still closed:

- The measure phase refuses to run: `MEASUREMENT_APPROVED = False` in
  `scripts/followup/ham10000_asism_v2_utility.py`. Opening it is a dated entry in the contract (§8)
  followed by a commit of its own. Nothing in this file changes that flag.
- Stage 4 of this design (step 7) needs its own approval after the selection is read (step 6).
- Stage 5 on `final_eval_heldout` is not in this file. It waits for the supervisor's test-set policy
  (contract §12 and §13, decision 8).

The same path was run on a CPU with fixture images and a planted utility formula by
`scripts/smoke/asism_v2_learned_e2e_smoke.py` (wiring only, not evidence). The older file
`docs/ham10000_runpod_commands.md` is the v1 order and is not changed.

What each step costs, from recorded runs on an RTX 5090 (about $0.72 per hour when E4 was planned;
check the current price):

| Step | Where | Runs | Time | Cost |
|---|---|---|---|---|
| 2 plan | CPU | — | seconds | — |
| 3 measure | GPU | 1,000 × about 14 s | about 4 h | about $3 |
| 4 G1 | CPU | — | seconds | — |
| 5 accept, fit, select, stability | CPU | 200 bootstrap models × 5 fit seeds | about 1.5 h on a laptop CPU | — |
| 7 Stage 4 | GPU | 80 × about 578 s (A, B, C, D × 20 seeds) | about 13 h | about $9 |
| 8 aggregate, compare on classifier_val | CPU | — | minutes | — |

---

## 0. On the laptop, before a pod is started

The full gate, with OneDrive sync paused (two suites fail under a syncing OneDrive with
`PermissionError [WinError 5]` and pass alone):

```bash
python tests/run_all.py
```

The wiring smoke (CPU, about 20 minutes; the work directory must be new and outside the repo):

```bash
python -m scripts.smoke.asism_v2_learned_e2e_smoke --candidates C:/Users/walaa/ham10000_work/v3_signals/stage2/all_candidates.csv --scores-dir C:/Users/walaa/ham10000_work/v3_signals/four_signal/gonogo_signal_set --work-dir C:/Users/walaa/ham10000_work/smoke_learned_e2e_NN
```

### 0a. The run line and the bundle

The order matters: the pod runs the code it is given, and the measure phase reads
`MEASUREMENT_APPROVED` from that code. So the approval commit is made first and the bundle second.

1. Walaa's approval is written as a dated entry in the contract (§8), in a commit of its own.
2. `MEASUREMENT_APPROVED = True` is set in a second commit of its own.
3. Both sit on a run line, `run/ham10000-asism-v2-learned`, cut from the reviewed branch. That branch
   already contains `main`, `run/ham10000-e4` and every ASISM v2 branch, so one branch carries
   everything the pod needs. Nothing has to be pushed to GitHub for the pod.

```bash
git bundle create C:/Users/walaa/ham10000_work/bundles/ham10000-asism-v2-learned.bundle run/ham10000-asism-v2-learned
```

```bash
git rev-parse --short run/ham10000-asism-v2-learned
```

Write the printed commit down: step 1 checks it on the pod.

### 0b. What is uploaded (Jupyter file browser, into `/workspace/upload/`)

- `ham10000-asism-v2-learned.bundle`
- the eight files of `C:\Users\walaa\ham10000_work\v3_signals\four_signal\gonogo_signal_set\`
  (four `*_scores.parquet` and four `*_scores.provenance.json`)

### 0c. The pod itself

The pod must be started on the network volume that holds `/workspace/master` (the frozen candidate
manifest stores absolute paths under `/workspace/master/outputs/ham10000/stage2/ham-stratified-v1/full/`,
so `PROJECT_ROOT` must be `/workspace/master`). The recorded timings are for an RTX 5090; the whole
measurement should run on one GPU type.

## 1. Pod preflight (CPU on the pod, no training)

A new terminal (after a reconnect, for example) knows neither the environment nor `$NS`, `$CAND`
and `$SCORES`. The one line marked below sets all of them and is repeated in every new terminal; the
measure command in step 3 does not depend on it.

```bash
source /workspace/env.sh && cd $PROJECT_ROOT && git status --short && git log --oneline -1
```

Bring the code to the run line (expected: no local changes above; the commit printed at the end is
the one written down in step 0a):

```bash
source /workspace/env.sh && cd $PROJECT_ROOT && git fetch /workspace/upload/ham10000-asism-v2-learned.bundle run/ham10000-asism-v2-learned:run/ham10000-asism-v2-learned && git checkout run/ham10000-asism-v2-learned && git log --oneline -1
```

The signal set goes where the commands below expect it:

```bash
mkdir -p /workspace/signals/four_signal/gonogo_signal_set && cp /workspace/upload/*_scores.parquet /workspace/upload/*_scores.provenance.json /workspace/signals/four_signal/gonogo_signal_set/
```

The three variables every later block uses (repeat this line in any new terminal):

```bash
source /workspace/env.sh && cd $PROJECT_ROOT && export NS=ham-stratified-v1 CAND=$PROJECT_ROOT/outputs/ham10000/stage2/ham-stratified-v1/all_candidates.csv SCORES=/workspace/signals/four_signal/gonogo_signal_set
```

The candidate pool must be the frozen one. Expected:
`bf8047b5dde96259c1268e5d8a3db6bac0d37392088df45182fde350ac7ef51a`.

```bash
sha256sum $CAND
```

Every candidate image, and the three splits the path reads, must be on the volume. Expected: 3168
candidates, 0 missing; at least as many image files as the split has rows (contract §2:
`classifier_train` 1,641, `classifier_val` 1,401, `asism_tuning_heldout` 1,377); eight files in
`$SCORES`.

```bash
python -c "import pandas as pd, pathlib as p; d=pd.read_csv('$CAND'); print(len(d), 'candidates,', sum(not p.Path(x).is_file() for x in d.image_path), 'missing')"
```

```bash
for s in classifier_train classifier_val asism_tuning_heldout; do echo $s $(ls $PROJECT_ROOT/data/ham10000/processed/images/$NS/$s | wc -l); done
```

```bash
ls $SCORES
```

The gate on the pod:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

```bash
python tests/run_all.py
```

**Stop if anything above is not as expected.**

## 2. Plan (CPU, seconds)

Freezes the roles, the 200 designed subsets and the 1,000 cells. Written once; a second run is
refused.

```bash
python -m scripts.followup.ham10000_asism_v2_utility --phase plan --candidates $CAND --scores-dir $SCORES
```

Output: `$PROJECT_ROOT/outputs/ham10000/stage3_asism_v2_ranker/$NS/utility_plan.json`.

## 3. Measure (GPU, about 4 hours) — THE STEP THAT NEEDS APPROVAL

Refused in code until the approval commit exists. Resumable: a restart continues with the cells
that are not yet written, and refuses if the inputs changed.

**Never start it twice.** Two copies would both write the same cells, and G1 refuses a repeated cell;
the run files are evidence and are not edited by hand. So the first command, before a start and
before every restart, is the check that nothing is running (expected: no output):

```bash
pgrep -af ham10000_asism_v2_utility
```

Start, and restart after an interruption, with the same line. The log is appended to, not replaced:

```bash
source /workspace/env.sh && cd $PROJECT_ROOT && setsid nohup python -m scripts.followup.ham10000_asism_v2_utility --phase measure --candidates $PROJECT_ROOT/outputs/ham10000/stage2/ham-stratified-v1/all_candidates.csv --i-understand-this-trains-real-models < /dev/null >> /workspace/asism_v2_measure.log 2>&1 &
```

The first two lines of the log are `measure inputs: frozen` (`identical` on a restart) and
`1000 of 1000 utility runs pending on <GPU name>` (fewer on a restart). **If the device is `cpu`,
stop it at once** (`pkill -f ham10000_asism_v2_utility`): the trainer falls back to the CPU silently.
The first run also downloads the ImageNet DenseNet-121 weights if the volume's cache does not hold
them.

Look at the first ten runs before leaving it: each line should take about 14 s and print a macro
AUROC; the two run files should be growing.

```bash
tail -n 12 /workspace/asism_v2_measure.log
```

```bash
wc -l $PROJECT_ROOT/outputs/ham10000/stage3_asism_v2_ranker/$NS/utility_runs_*.jsonl
```

Done when the two files hold 1,000 lines together (800 fit, 200 test) and `pgrep` prints nothing.
The test file is not opened by anything before step 5.

If a restart stops with `measure inputs changed since the first run`, a split, the candidate
manifest, the plan or the measurement code differs from the first run: stop and find out which, do
not delete `measure_inputs.json`. If it stops with a JSON error while reading a run file, the pod
died in the middle of writing a line: stop and report it; the file is not repaired by hand.

## 4. G1 (CPU, seconds)

```bash
python -m scripts.followup.ham10000_asism_v2_utility --phase gate
```

Written once to `g1_reliability.json`. If G1 fails, the path ends here and that is the result.

## 5. Accept, fit, select, stability (CPU)

Each phase refuses unless the one before it succeeded.

```bash
python -m scripts.followup.ham10000_asism_v2_learned_select --phase accept --candidates $CAND --scores-dir $SCORES
```

`ranker_acceptance.json` is written once and reads the test subsets once. **If `accepted` is false,
the learned ranking is not used, nothing below is run, and that is the reported result.**

```bash
python -m scripts.followup.ham10000_asism_v2_learned_select --phase fit --candidates $CAND --scores-dir $SCORES
```

```bash
python -m scripts.followup.ham10000_asism_v2_learned_select --phase select --candidates $CAND --scores-dir $SCORES
```

```bash
python -m scripts.followup.ham10000_asism_v2_learned_select --phase stability --candidates $CAND --scores-dir $SCORES
```

## 6. Stop. Read the selection before Stage 4.

```bash
python -c "import json; m=json.load(open('$PROJECT_ROOT/outputs/ham10000/stage3_asism_v2_ranker/$NS/asism_v2_learned_selection_manifest.json')); print(m['selection_outcome'], m['n_selected_c'], m['per_class'])"
```

The outcome decides the Stage 4 protocol. Nobody chooses it:

| `selection_outcome` | Protocol | Conditions trained |
|---|---|---|
| `subset` | `asism_v2_learned` | A, B, C, D |
| `all` or `none` | `asism_v2_learned_all_or_none` | A, B (C is B's or A's training set by definition) |

Download the whole `stage3_asism_v2_ranker/$NS/` directory before going on. **Stage 4 needs its own
approval.** The pod can be stopped here.

## 7. Stage 4 (GPU, about 13 hours for A, B, C, D × 20 seeds)

For `selection_outcome = subset`:

```bash
export PROTO=asism_v2_learned; export CONDS="A B C D"
```

For `all` or `none`:

```bash
export PROTO=asism_v2_learned_all_or_none; export CONDS="A B"
```

A run that already wrote its manifest is skipped, so the loop can be restarted:

```bash
setsid nohup bash -c 'for c in $CONDS; do for s in $(seq 42 61); do [ -f $PROJECT_ROOT/outputs/ham10000/stage4_asism_v2_learned/$NS/$c/seed$s/run_manifest.json ] || python scripts/classify/ham10000_train_conditions.py --protocol $PROTO --condition $c --seed $s || exit 1; done; done' < /dev/null > /workspace/asism_v2_stage4.log 2>&1 &
```

```bash
ls $PROJECT_ROOT/outputs/ham10000/stage4_asism_v2_learned/$NS/*/seed*/run_manifest.json | wc -l
```

## 8. Aggregate and compare on classifier_val (CPU)

```bash
python scripts/classify/ham10000_aggregate_conditions.py --protocol $PROTO
```

```bash
python -m scripts.followup.ham10000_asism_v2_compare --protocol $PROTO --source classifier_val
```

This comparison is on the Stage 4 monitoring split and is labelled MONITORING everywhere. It is not
the final evaluation and decides nothing.

## 9. Download, then stop the pod

- `outputs/ham10000/stage3_asism_v2_ranker/$NS/` (plan, run files, G1, acceptance, ranker, C and D,
  manifest, stability)
- `outputs/ham10000/stage4_asism_v2_learned/$NS/` without the `model.pt` files unless Stage 5 is
  going to run from this machine
- `/workspace/asism_v2_measure.log`, `/workspace/asism_v2_stage4.log`

```bash
runpodctl stop pod $RUNPOD_POD_ID
```
