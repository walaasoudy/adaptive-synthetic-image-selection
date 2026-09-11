"""Stage 1 (SDXL + LoRA) on Modal — runs the repository's own scripts, unmodified.

Why Modal runs the repo scripts directly (not the Kaggle notebook): the repo's Stage 2-5 gates
check a provenance chain (frozen split manifest -> LoRA metadata -> generation manifest -> ...).
Running `02b_build_sixway_splits.py` and `train_lora_sdxl.py` here produces that chain natively,
so no Stage-1 -> Stage-2 adapter is needed.

Layout inside every container
    /root/repo        scripts/ configs/ environment/ copied from this checkout (the code identity)
    /vol/project      PROJECT_ROOT, on a persistent Modal Volume (data, caches, checkpoints, logs)

Nothing under scripts/ or configs/ is edited. Production-only values (namespace, raw_dir,
dev_subset off) are supplied through THESIS_CONFIG_OVERLAY, a file on the volume that
scripts/utils/config.py merges at load time, so `code_identity()` stays equal to this checkout.

One-time setup (on your machine):
    pip install modal
    python -m modal setup
    modal secret create kaggle KAGGLE_USERNAME=<your-user> KAGGLE_KEY=<your-key>

Run from the repository root. `--detach` keeps the job running if your laptop sleeps or the
terminal closes; follow it on the Modal dashboard or with `modal app logs chest-synth-stage1`.

    modal run --detach cloud/modal_stage1.py::prepare_data   # CPU, a few hours, ~$1
    modal run cloud/modal_stage1.py::smoke                   # GPU ~20 min: train, checkpoint, resume
    modal run --detach cloud/modal_stage1.py::train --hours 6
    modal run cloud/modal_stage1.py::status

GPU: A10 (24 GB, bf16) by default — the Starter plan without a payment method allows only T4, L4
and A10. To use another, set STAGE1_GPU before `modal run`, e.g. on Windows cmd:  set STAGE1_GPU=L4
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

REPO_LOCAL = Path(__file__).resolve().parents[1]
REPO = "/root/repo"
VOL_MOUNT = "/vol"
PROJECT_ROOT = Path(VOL_MOUNT) / "project"
OVERLAY = PROJECT_ROOT / "thesis_overlay.yaml"
RAW_LOCATION = PROJECT_ROOT / "raw_location.json"

NAMESPACE = "production-thesis-v1"
KAGGLE_SLUG = "ashery/chexpert"
SPLITS = ["gen_train", "gen_val", "classifier_train", "classifier_val",
          "asism_tuning_heldout", "final_eval_heldout"]

CKPT_ROOT = PROJECT_ROOT / "checkpoints" / "stage1_lora_sdxl"
SMOKE_CKPT_ROOT = PROJECT_ROOT / "checkpoints" / "stage1_smoke"
LOG_DIR = PROJECT_ROOT / "logs" / "modal"

# The Starter plan without a payment method may use only T4, L4 and A10 (L40S/A100/H100 are
# refused at launch; see cloud/modal_gpu_probe.py). A10 and L4 both run the frozen bf16 protocol.
GPU = os.environ.get("STAGE1_GPU", "A10")
# $/hour, from modal.com/pricing (per-second rates x 3600). Used only for printed estimates.
GPU_PRICE_PER_HOUR = {"T4": 0.59, "L4": 0.80, "A10": 1.10, "A10G": 1.10, "L40S": 1.95,
                      "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95}

# Latent moments are (8, res/8, res/8) fp16; prompt embeddings are (77x2048 + 1280) fp16.
LATENT_BYTES_768 = 8 * 96 * 96 * 2
PROMPT_BYTES = (77 * 2048 + 1280) * 2

_CODE_IGNORE = ["**/__pycache__", "**/*.pyc"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Frozen matrix (environment/RUNPOD_MATRIX.md): torch 2.8.0 / torchvision 0.23.0 / CUDA 12.8.
    .pip_install("torch==2.8.0", "torchvision==0.23.0",
                 index_url="https://download.pytorch.org/whl/cu128")
    .pip_install_from_requirements(str(REPO_LOCAL / "environment" / "requirements.txt"))
    .env({"STAGE1_GPU": GPU})   # so the container prints cost estimates for the GPU actually chosen
    .add_local_dir(REPO_LOCAL / "scripts", f"{REPO}/scripts", ignore=_CODE_IGNORE)
    .add_local_dir(REPO_LOCAL / "configs", f"{REPO}/configs", ignore=_CODE_IGNORE)
    .add_local_dir(REPO_LOCAL / "environment", f"{REPO}/environment", ignore=_CODE_IGNORE)
)

volume = modal.Volume.from_name("chest-synth-thesis", create_if_missing=True, version=2)
app = modal.App("chest-synth-stage1")


# ------------------------------------------------------------------------------------------------
# helpers (run inside the container)
# ------------------------------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "THESIS_CONFIG_OVERLAY": str(OVERLAY),
        "HF_HOME": str(PROJECT_ROOT / ".cache" / "huggingface"),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "KAGGLEHUB_CACHE": str(PROJECT_ROOT / ".cache" / "kagglehub"),
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TQDM_MININTERVAL": "60",   # one progress line a minute instead of one per step
    })
    return env


def _log_path(name: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{name}_{_utcnow()}.log"


def _run(cmd: list[str], log: Path) -> None:
    """Run a repo script from /root/repo, streaming output to stdout and a log on the volume."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=REPO, env=_env(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"command failed (exit {code}): {' '.join(cmd)} — see {log}")


def _find_data_root(base: Path) -> Path:
    """Same search as scripts/data/00_download_dataset.py::find_data_root."""
    for candidate in (base, base / "CheXpert-v1.0-small", base / "CheXpert-v1.0"):
        if (candidate / "train.csv").exists() and (candidate / "train").is_dir():
            return candidate
    for train_csv in base.rglob("train.csv"):
        if (train_csv.parent / "train").is_dir():
            return train_csv.parent
    raise FileNotFoundError(f"No CheXpert train.csv with a sibling train/ under {base}")


def _write_overlay(raw_root: Path) -> None:
    """Production values for configs/stage1_lora_sdxl.yaml, applied without editing the repo.

    These are exactly the three 'revert for production' lines run_dev_subset.sh documents, plus
    raw_dir pointing at the dataset on the volume (no symlinks, no second copy of 11 GB)."""
    OVERLAY.parent.mkdir(parents=True, exist_ok=True)
    OVERLAY.write_text(
        "stage1:\n"
        "  dev_subset:\n"
        "    enabled: false\n"
        "  split:\n"
        f"    namespace: \"{NAMESPACE}\"\n"
        f"    input_csv: \"{raw_root / 'train.csv'}\"\n"
        "  paths:\n"
        f"    raw_dir: \"{raw_root}\"\n",
        encoding="utf-8",
    )


def _require_overlay() -> None:
    if not OVERLAY.is_file():
        raise SystemExit(f"{OVERLAY} is missing. Run prepare_data first:\n"
                         "  modal run --detach cloud/modal_stage1.py::prepare_data")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resume_args(ckpt_root: Path) -> tuple[list[str], str | None, int]:
    """Mirror of scripts/train/launch_resumable.sh: continue the latest run from its newest
    full checkpoint. Returns (extra CLI args, run_id, resumed step)."""
    latest_run = ckpt_root / "latest_run.json"
    if not latest_run.is_file():
        return [], None, 0
    run_id = _read_json(latest_run)["run_id"]
    latest = ckpt_root / run_id / "latest.json"
    if not latest.is_file():
        return ["--run-id", run_id], run_id, 0
    info = _read_json(latest)
    return ["--resume-from", info["checkpoint_dir"], "--run-id", run_id], run_id, int(info["step"])


def _latest_step(ckpt_root: Path) -> int:
    _, _, step = _resume_args(ckpt_root)
    return step


def _train_cmd(*overrides: str, extra: list[str] | None = None) -> list[str]:
    return ["accelerate", "launch", "--config_file", "configs/accelerate_config.yaml",
            "scripts/train/train_lora_sdxl.py", *(extra or []),
            f"split.namespace={NAMESPACE}", *overrides]


# ------------------------------------------------------------------------------------------------
# 1. data preparation (CPU)
# ------------------------------------------------------------------------------------------------

@app.function(image=image, volumes={VOL_MOUNT: volume},
              secrets=[modal.Secret.from_name("kaggle")],
              # Billed at max(request, usage). 03_preprocess_images.py is single-process and the
              # split build is a pandas groupby over 224k rows, so 2 cores / 8 GiB is enough.
              cpu=2.0, memory=8192, timeout=24 * 3600)
def prepare_data(splits: str = "all") -> None:
    """Download CheXpert, verify it, build + freeze the production split, preprocess, caption.

    Every step is idempotent: re-running skips what is already done (the split run is immutable,
    03_preprocess_images.py resumes per split, captions are rebuilt deterministically)."""
    # kagglehub runs in THIS process, so its cache location must be set here, not only in the
    # subprocess env — otherwise the 11 GB lands on ephemeral container disk and is lost on exit.
    os.environ.update(_env())
    log = _log_path("prepare_data")
    wanted = SPLITS if splits == "all" else splits.split(",")

    if RAW_LOCATION.is_file():
        raw_root = Path(_read_json(RAW_LOCATION)["raw_root"])
        print(f"dataset already on the volume: {raw_root}", flush=True)
    else:
        import kagglehub

        print(f"downloading {KAGGLE_SLUG} via kagglehub into the volume ...", flush=True)
        download = Path(kagglehub.dataset_download(KAGGLE_SLUG))
        raw_root = _find_data_root(download)
        for archive in download.rglob("*.zip"):   # extracted already; don't keep a second copy
            archive.unlink()
        RAW_LOCATION.write_text(json.dumps({"raw_root": str(raw_root), "slug": KAGGLE_SLUG,
                                            "downloaded_at_utc": _utcnow()}, indent=2))
        volume.commit()
    _write_overlay(raw_root)

    _run([sys.executable, "scripts/data/01_verify_download.py"], log)

    split_dir = PROJECT_ROOT / "data" / "chexpert" / "processed" / "splits" / NAMESPACE
    if split_dir.exists():
        print(f"split run {NAMESPACE} already frozen at {split_dir} — immutable, reusing", flush=True)
    else:
        _run([sys.executable, "scripts/data/02b_build_sixway_splits.py", "--namespace", "production",
              "--run-id", NAMESPACE, "--input-csv", str(raw_root / "train.csv"), "--freeze"], log)
    volume.commit()

    _run([sys.executable, "scripts/data/03_preprocess_images.py", "--namespace", NAMESPACE,
          "--splits", *wanted], log)
    volume.commit()
    _run([sys.executable, "scripts/data/04_generate_captions.py", "--namespace", NAMESPACE,
          "--splits", "gen_train", "gen_val"], log)
    volume.commit()

    manifest = _read_json(split_dir / "split_manifest_v2.json")
    print("\n" + "=" * 78)
    print(f"SPLIT  {NAMESPACE}  frozen={manifest.get('frozen')}  "
          f"support={'PASSED' if manifest.get('support_check', {}).get('passed') else 'FAILED'}")
    print(f"       manifest_hash={manifest['manifest_hash']}")
    for name, n in manifest.get("patients_per_split", {}).items():
        print(f"       {name:24s} {n:>7,} patients  {manifest['images_per_split'][name]:>8,} images")

    captions_dir = PROJECT_ROOT / "data" / "chexpert" / "processed" / "captions" / NAMESPACE
    total_gb = 0.0
    for split in ("gen_train", "gen_val"):
        path = captions_dir / f"{split}_captions.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        unique = {c for r in records for c in r["caption_variants"]}
        latent_gb = len(records) * LATENT_BYTES_768 / 1024 ** 3
        prompt_gb = len(unique) * PROMPT_BYTES / 1024 ** 3
        total_gb += latent_gb + prompt_gb
        print(f"CACHE  {split:10s} {len(records):>7,} images -> latents {latent_gb:5.1f} GB, "
              f"{len(unique):>7,} unique captions -> prompts {prompt_gb:5.1f} GB")
    print(f"       trainer loads both caches into RAM: ~{total_gb:.0f} GB "
          f"(train() requests 32 GB and can grow; tell Claude if this is far above 32)")
    print("=" * 78)
    print("Next: review the split above, then  modal run cloud/modal_stage1.py::smoke")


# ------------------------------------------------------------------------------------------------
# 2. smoke test (GPU): train -> checkpoint -> resume -> final export, on 64 images
# ------------------------------------------------------------------------------------------------

@app.function(image=image, gpu=GPU, volumes={VOL_MOUNT: volume},
              cpu=4.0, memory=16384, timeout=2 * 3600)
def smoke(samples: int = 64, optimizer: str = "adamw") -> None:
    """Exercises every Stage 1 code path the long run depends on, in isolated directories
    (checkpoints/stage1_smoke, separate small caches), so it can never be resumed into by train().

    Leg A trains 10 steps and checkpoints at step 10. Leg B resumes from that checkpoint and runs
    to step 20. Passing means bf16, the optimizer, caching, save_state, load_state and the final
    LoRA export all work on this GPU before any paid long run starts."""
    _require_overlay()
    log = _log_path("smoke")
    shutil.rmtree(SMOKE_CKPT_ROOT, ignore_errors=True)
    common = [f"optimizer.name={optimizer}",
              "checkpointing.save_every_n_steps=10", "validation.val_every_n_steps=10",
              f"paths.checkpoints_dir={SMOKE_CKPT_ROOT}",
              f"paths.logs_dir={PROJECT_ROOT / 'logs' / 'stage1_smoke'}"]
    started = time.time()

    print("\n### smoke leg A: fresh run to step 10", flush=True)
    _run(_train_cmd("training.max_train_steps=10", "training.min_train_steps=10", *common,
                    extra=["--max-samples-per-split", str(samples)]), log)
    resume, run_id, step = _resume_args(SMOKE_CKPT_ROOT)
    if step != 10:
        raise SystemExit(f"SMOKE FAILED: expected a checkpoint at step 10, found step {step}")

    print(f"\n### smoke leg B: resume run {run_id} from step {step} to step 20", flush=True)
    _run(_train_cmd("training.max_train_steps=20", "training.min_train_steps=20", *common,
                    extra=["--max-samples-per-split", str(samples), *resume]), log)
    final = SMOKE_CKPT_ROOT / run_id / "final"
    meta = _read_json(final / "metadata.json")
    weights = sorted(final.glob("*.safetensors"))
    if meta.get("step") != 20 or not weights:
        raise SystemExit(f"SMOKE FAILED: final step={meta.get('step')}, weights={weights}")
    volume.commit()

    minutes = (time.time() - started) / 60
    print("\n" + "=" * 78)
    print(f"SMOKE PASSED on {GPU}: trained 10 -> resumed -> 20 -> exported "
          f"{[w.name for w in weights]}")
    print(f"  split_manifest_hash in LoRA metadata: {meta.get('split_manifest_hash')}")
    print(f"  wall time {minutes:.1f} min  (~${minutes / 60 * GPU_PRICE_PER_HOUR.get(GPU, 0):.2f})")
    print("Next:  modal run --detach cloud/modal_stage1.py::train --hours 6")
    print("=" * 78)


# ------------------------------------------------------------------------------------------------
# 3. full training (GPU), resumable across invocations
# ------------------------------------------------------------------------------------------------

@app.function(image=image, gpu=GPU, volumes={VOL_MOUNT: volume},
              # Training reads pre-cached latents, so CPU load is light; RAM holds both caches.
              # If Modal preempts the container, re-running train() resumes from the last checkpoint.
              cpu=4.0, memory=32768, timeout=24 * 3600)
def train(hours: float = 6.0, max_steps: int = 15000, optimizer: str = "adamw") -> None:
    """Continue (or start) the production Stage 1 run.

    Stops cleanly at the first checkpoint written after `hours` have passed, so no steps are ever
    thrown away; call it again to continue. The first invocation also builds the VAE latent and
    text-embedding caches (a one-time GPU cost, reused by every later invocation).

    max_steps: 15000 matches the protocol used in the Kaggle notebook and the config's
    min_train_steps; configs/stage1_lora_sdxl.yaml's max_train_steps (30000) is only an upper
    bound. Keep it identical across invocations — it also sets the cosine LR schedule length.
    optimizer: 'adamw' follows run_dev_subset.sh (the pinned bitsandbytes 0.45.2 has no CUDA 12.8
    kernel); the config documents adamw as the in-protocol fallback for adamw_8bit."""
    _require_overlay()
    if not 0 < hours <= 22:
        raise SystemExit("--hours must be in (0, 22]; the container hard timeout is 24 h")

    resume, run_id, start_step = _resume_args(CKPT_ROOT)
    if run_id and (CKPT_ROOT / run_id / "final" / "metadata.json").is_file():
        meta = _read_json(CKPT_ROOT / run_id / "final" / "metadata.json")
        print(f"Stage 1 already COMPLETE: run {run_id}, step {meta['step']}. Nothing to do.")
        return

    price = GPU_PRICE_PER_HOUR.get(GPU, 0)
    print("=" * 78)
    print(f"STAGE 1  namespace={NAMESPACE}  gpu={GPU}  target={max_steps:,} steps")
    print(f"  resume   : {'run ' + run_id + ' from step ' + str(start_step) if resume else 'fresh start'}")
    print(f"  budget   : stop at the first checkpoint after {hours} h  (~${hours * price:.0f}, "
          f"plus up to one checkpoint interval)")
    print("=" * 78, flush=True)

    log = _log_path("train")
    cmd = _train_cmd(f"optimizer.name={optimizer}",
                     f"training.max_train_steps={max_steps}",
                     f"training.min_train_steps={min(15000, max_steps)}",
                     extra=resume)
    print(f"$ {' '.join(cmd)}", flush=True)

    started = time.time()
    deadline = started + hours * 3600
    stop_reason = {"value": None}
    handle = open(log, "a", encoding="utf-8")
    # New session so the whole accelerate -> python process group can be signalled together.
    proc = subprocess.Popen(cmd, cwd=REPO, env=_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)

    def watchdog() -> None:
        step_at_deadline = None
        while proc.poll() is None:
            time.sleep(60)
            if time.time() < deadline:
                continue
            step = _latest_step(CKPT_ROOT)
            if step_at_deadline is None:
                step_at_deadline = step
                print(f"\n[watchdog] {hours} h reached at checkpoint step {step}; "
                      "stopping after the next checkpoint is written", flush=True)
            elif step > step_at_deadline:
                stop_reason["value"] = f"time budget reached; stopped cleanly after checkpoint {step}"
                print(f"\n[watchdog] {stop_reason['value']}", flush=True)
                time.sleep(30)   # let retention cleanup finish after latest.json is written
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                return

    threading.Thread(target=watchdog, daemon=True).start()
    for line in proc.stdout:
        print(line, end="", flush=True)
        handle.write(line)
    code = proc.wait()
    handle.close()
    volume.commit()

    elapsed_h = (time.time() - started) / 3600
    end_step = _latest_step(CKPT_ROOT)
    _, run_id, _ = _resume_args(CKPT_ROOT)
    complete = bool(run_id) and (CKPT_ROOT / run_id / "final" / "metadata.json").is_file()
    print("\n" + "=" * 78)
    if complete:
        print(f"STAGE 1 COMPLETE — final LoRA in {CKPT_ROOT / run_id / 'final'}")
    elif stop_reason["value"]:
        print(f"PAUSED: {stop_reason['value']}")
    else:
        print(f"TRAINER EXITED with code {code} before completion — read {log}")
    print(f"  steps    : {start_step:,} -> {end_step:,} of {max_steps:,} this session")
    print(f"  elapsed  : {elapsed_h:.2f} h  (~${elapsed_h * price:.2f} on {GPU})")
    if end_step > start_step and not complete:
        rate = (end_step - start_step) / elapsed_h
        left_h = (max_steps - end_step) / rate
        print(f"  remaining: ~{left_h:.1f} h  (~${left_h * price:.0f})  at {rate:,.0f} steps/h")
        print("  continue : modal run --detach cloud/modal_stage1.py::train --hours 6")
    print("=" * 78)
    if code != 0 and not stop_reason["value"]:
        raise SystemExit(code)


# ------------------------------------------------------------------------------------------------
# 4. status (CPU, cheap)
# ------------------------------------------------------------------------------------------------

@app.function(image=image, volumes={VOL_MOUNT: volume}, cpu=1, memory=2048, timeout=900)
def status() -> None:
    """What is on the volume and how far Stage 1 has got."""
    print(f"overlay      : {'present' if OVERLAY.is_file() else 'MISSING (run prepare_data)'}")
    if RAW_LOCATION.is_file():
        print(f"dataset      : {_read_json(RAW_LOCATION)['raw_root']}")
    split_manifest = (PROJECT_ROOT / "data" / "chexpert" / "processed" / "splits" / NAMESPACE
                      / "split_manifest_v2.json")
    if split_manifest.is_file():
        m = _read_json(split_manifest)
        print(f"split        : frozen={m.get('frozen')} hash={m['manifest_hash'][:16]}...")
    images = PROJECT_ROOT / "data" / "chexpert" / "processed" / "images_768" / NAMESPACE
    for split in SPLITS:
        d = images / split
        if d.is_dir():
            print(f"  preprocessed {split:24s} {sum(1 for _ in d.glob('*.jpg')):>8,} images")

    _, run_id, step = _resume_args(CKPT_ROOT)
    if run_id:
        final = CKPT_ROOT / run_id / "final" / "metadata.json"
        snaps = sorted((CKPT_ROOT / run_id / "lora_weights").glob("step_*"),
                       key=lambda p: int(p.name.split("_")[-1]))
        print(f"stage 1 run  : {run_id}")
        print(f"  checkpoint : step {step:,}   complete={final.is_file()}")
        print(f"  lora snaps : {[p.name for p in snaps][-5:]}")
    else:
        print("stage 1 run  : not started")

    for label, path in [("raw dataset", PROJECT_ROOT / ".cache" / "kagglehub"),
                        ("hf models", PROJECT_ROOT / ".cache" / "huggingface"),
                        ("processed", PROJECT_ROOT / "data" / "chexpert" / "processed"),
                        ("checkpoints", PROJECT_ROOT / "checkpoints")]:
        if path.exists():
            out = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True).stdout
            print(f"disk {label:12s}: {out.split()[0] if out else '?'}")

    logs = sorted(LOG_DIR.glob("train_*.log"))
    if logs:
        tail = logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
        print(f"\nlast train log ({logs[-1].name}):")
        for line in tail:
            print(f"  {line[-160:]}")
