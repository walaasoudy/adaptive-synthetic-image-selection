#!/usr/bin/env bash
# One-shot dev-subset shakedown pipeline (Stages 1-5) for namespace dev-10k-v1.
#
# Re-runnable: every completed step drops a marker in .pipeline_state/dev-10k-v1/ and is skipped
# on the next run, so if the script dies at step N you just run it again and it resumes at N.
# To force a step to re-run, delete its marker (e.g. rm .pipeline_state/dev-10k-v1/36_train_learned.done).
#
# Run it detached so it survives an SSH drop:
#     export PROJECT_ROOT=/workspace/chest-synth-thesis
#     cd $PROJECT_ROOT
#     nohup bash run_dev_subset.sh > logs/dev10k_run.log 2>&1 &
#     tail -f logs/dev10k_run.log
#
# For the FULL production run instead: revert the three dev_subset lines in
# configs/stage1_lora_sdxl.yaml, use `--namespace production --run-id production-thesis-v1 --freeze`
# in 02b, `--namespace production-thesis-v1` everywhere else, and a >= 150 GB persistent volume.
set -Eeuo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/workspace/chest-synth-thesis}"
cd "$PROJECT_ROOT"
mkdir -p logs

NS=dev-10k-v1
STATE="$PROJECT_ROOT/.pipeline_state/$NS"
mkdir -p "$STATE"
AUTO_APPROVE_PILOT="${AUTO_APPROVE_PILOT:-1}"   # dev shakedown: auto-approve the Stage-2 pilot.
                                               # Set to 0 to pause and inspect the pilot images.

CURRENT_STEP="(init)"
trap 'echo "!!! FAILED at step: ${CURRENT_STEP} (exit $?). Fix it, then re-run this script to resume."; exit 1' ERR

run () {                        # run <marker-name> <command...>
  local name="$1"; shift
  CURRENT_STEP="$name"
  if [ -f "$STATE/$name.done" ]; then echo ">>> SKIP  $name"; return 0; fi
  echo; echo "================ $name  ::  $*"; date
  "$@"
  touch "$STATE/$name.done"
  echo ">>> OK    $name"
}

echo "### dev-subset pipeline | namespace=$NS | PROJECT_ROOT=$PROJECT_ROOT | $(date)"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'OK')"

# ---------------- data + Stage 1 (LoRA) ----------------
run 00_download       python scripts/data/00_download_dataset.py
run 01b_dev_subset    python scripts/data/01b_build_dev_subset.py
run 02b_splits        python scripts/data/02b_build_sixway_splits.py --namespace dev --run-id "$NS"
run 03_preprocess     python scripts/data/03_preprocess_images.py --namespace "$NS" --splits gen_train gen_val classifier_train classifier_val asism_tuning_heldout final_eval_heldout
run 04_captions       python scripts/data/04_generate_captions.py --namespace "$NS" --splits gen_train gen_val

# Stage 1 LoRA in the FOREGROUND so the pipeline waits for it. (launch_resumable.sh backgrounds
# training and returns immediately, which lets Stage 2 start before any checkpoint exists.)
# STAGE1_MAX_STEPS keeps the dev shakedown short; the production run uses the config's 15000-30000.
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-2000}"
# optimizer.name=adamw: the RunPod pytorch-2.8/cu128 image ships a bitsandbytes without a CUDA
# binary. train_lora_sdxl.py now probes a real AdamW8bit step and falls back on its own, so this
# override is no longer required to avoid the crash -- it is kept as an explicit, recorded choice
# so this pipeline's optimizer does not depend on what a given pod image happens to ship.
run 10_stage1_lora    accelerate launch --config_file configs/accelerate_config.yaml scripts/train/train_lora_sdxl.py split.namespace="$NS" optimizer.name=adamw training.max_train_steps="$STAGE1_MAX_STEPS" training.min_train_steps="$STAGE1_MAX_STEPS"

# Point Stage 2 at the LoRA checkpoint step 10 produced (02_generate_synthetic_images.py reads
# configs/stage2_generation.yaml directly and takes no CLI override).
set_lora_ckpt () {
  local rid ckpt
  rid=$(python3 -c "import json; print(json.load(open('checkpoints/stage1_lora_sdxl/latest_run.json'))['run_id'])")
  ckpt="checkpoints/stage1_lora_sdxl/$rid/final"
  [ -d "$ckpt" ] || ckpt=$(ls -d "checkpoints/stage1_lora_sdxl/$rid/lora_weights/step_"* 2>/dev/null | sort -V | tail -1)
  [ -n "${ckpt:-}" ] && [ -d "$ckpt" ] || { echo "no LoRA checkpoint under checkpoints/stage1_lora_sdxl/$rid"; return 1; }
  sed -i "s|^\(\s*lora_weights_dir:\).*|\1 $ckpt|" configs/stage2_generation.yaml
  grep -n "lora_weights_dir:" configs/stage2_generation.yaml
}
run 11_set_lora_ckpt  set_lora_ckpt

# ---------------- Stage 2 (synthetic generation) ----------------
run 20_recipes        python scripts/generate/01_sample_label_recipes.py --namespace "$NS"
run 21_pilot          python scripts/generate/02_generate_synthetic_images.py --mode pilot --namespace "$NS"
if [ "$AUTO_APPROVE_PILOT" = "1" ]; then
  run 22_approve_pilot python scripts/generate/02_generate_synthetic_images.py --approve-pilot --reviewer walaa --notes "dev-subset shakedown" --namespace "$NS"
else
  echo ">>> PAUSED for pilot review. Inspect the 4 pilot images, then rerun:  AUTO_APPROVE_PILOT=1 bash $0"
  exit 0
fi
run 23_generate_full  python scripts/generate/02_generate_synthetic_images.py --mode full --namespace "$NS"

# ---------------- auxiliary classifier + Stage 3 (ASISM) ----------------
run 30_aux_classifier python scripts/classify/00_train_auxiliary_classifier.py --namespace "$NS" --max-steps 8000
run 31_signals        python scripts/asism/01_compute_signals.py --signal all --namespace "$NS"
run 32_gonogo         python scripts/asism/02_gonogo.py --namespace "$NS"
run 33_feasibility    python scripts/asism/04_build_utility_subsets.py --phase feasibility --namespace "$NS"
run 34_build_subsets  python scripts/asism/04_build_utility_subsets.py --phase build --namespace "$NS"
run 35_eval_subsets   python scripts/asism/04b_evaluate_utility_subsets.py --phase run --namespace "$NS"
run 36_train_learned  python scripts/asism/05_train_learned_asism.py --namespace "$NS"
run 37_fixed_thresh   python scripts/asism/06_learn_thresholds_select.py --namespace "$NS"
run 38_contexts       python scripts/asism/07_build_threshold_contexts.py --namespace "$NS"
run 39_verify_thresh  python scripts/asism/07b_verify_thresholds_proxy.py --phase run --i-understand-this-trains-real-models --namespace "$NS"
run 40_thresh_net     python scripts/asism/08_train_threshold_network.py --namespace "$NS"
run 41_full_policy    python scripts/asism/08b_verify_full_policy_proxy.py --phase run --i-understand-this-trains-real-models --namespace "$NS"
run 42_finalize       python scripts/asism/09_finalize_learned_selection.py --namespace "$NS"

# ---------------- Stage 4 + Stage 5 ----------------
run 50_conditions     python scripts/classify/01_train_conditions.py --condition all --namespace "$NS"
run 51_stage5_eval    python scripts/eval/stage5_evaluate.py --final-eval-run-id dev10k-final-v1 --namespace "$NS"
run 52_compare        python scripts/eval/compare_conditions.py --run-dir "outputs/stage5/$NS/dev10k-final-v1"

echo
echo "### DONE. Stage 5 comparison in: outputs/stage5/$NS/dev10k-final-v1/"
