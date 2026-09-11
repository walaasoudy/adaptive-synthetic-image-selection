"""Before/after benchmarks for the pipeline optimizations, on Modal, with synthetic data.

Every benchmark runs the ORIGINAL behaviour and the OPTIMIZED behaviour through the same repo
code, in the same container, on the same synthetic inputs, so the comparison isolates the change.
No CheXpert data and no Kaggle secret are needed.

    modal run cloud/modal_bench.py::bench_classifier
    set BENCH_GPU=L4 && modal run cloud/modal_bench.py::bench_classifier    (Windows cmd)

Reported per mode: wall time, throughput, GPU utilization (sampled from nvidia-smi), peak VRAM,
peak host RSS, estimated cost; plus an equivalence check of the outputs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

REPO_LOCAL = Path(__file__).resolve().parents[1]
REPO = "/root/repo"
GPU = os.environ.get("BENCH_GPU", "A10")   # Starter plan: only T4, L4, A10 are allowed
GPU_PRICE_PER_HOUR = {"T4": 0.59, "L4": 0.80, "A10": 1.10, "A10G": 1.10, "L40S": 1.95,
                      "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95}
CPU_CORES, MEMORY_GIB = 4.0, 16
CPU_PRICE_CORE_HOUR, MEM_PRICE_GIB_HOUR = 0.0000131 * 3600, 0.00000222 * 3600
_CODE_IGNORE = ["**/__pycache__", "**/*.pyc"]

# Identical layer chain to cloud/modal_stage1.py, so Modal reuses the cached image.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "torchvision==0.23.0",
                 index_url="https://download.pytorch.org/whl/cu128")
    .pip_install_from_requirements(str(REPO_LOCAL / "environment" / "requirements.txt"))
    .env({"STAGE1_GPU": GPU})
    .add_local_dir(REPO_LOCAL / "scripts", f"{REPO}/scripts", ignore=_CODE_IGNORE)
    .add_local_dir(REPO_LOCAL / "configs", f"{REPO}/configs", ignore=_CODE_IGNORE)
    .add_local_dir(REPO_LOCAL / "environment", f"{REPO}/environment", ignore=_CODE_IGNORE)
    .add_local_dir(REPO_LOCAL / "tests", f"{REPO}/tests", ignore=_CODE_IGNORE + ["_runtime/**"])
)
app = modal.App("chest-synth-bench")

# Same Volume and Hugging Face cache path as production (cloud/modal_stage1.py), so SDXL is
# downloaded once and reused by both the benchmarks and the real run.
volume = modal.Volume.from_name("chest-synth-thesis", create_if_missing=True, version=2)
VOL_MOUNT = "/vol"
PROJECT_ROOT = "/vol/project"
HF_CACHE = f"{PROJECT_ROOT}/.cache/huggingface"
SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
# Only the files diffusers/transformers load for this pipeline (fp32 weights, as the repo loads
# them). The full repo snapshot also carries ONNX/OpenVINO exports and single-file checkpoints
# (~70 GB); none of those are ever read.
SDXL_FILES = ["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*",
              "text_encoder/config.json", "text_encoder/model.safetensors",
              "text_encoder_2/config.json", "text_encoder_2/model.safetensors",
              "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
              "unet/config.json", "unet/diffusion_pytorch_model.safetensors"]


def _repo_env() -> None:
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    os.environ.update({"PROJECT_ROOT": PROJECT_ROOT, "HF_HOME": HF_CACHE,
                       "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_OFFLINE": "1"})
    os.environ.pop("THESIS_CONFIG_OVERLAY", None)


@app.function(image=image, cpu=4.0, memory=8192, timeout=3600)
def run_tests(smoke: bool = True) -> None:
    """CPU-only: the repo's own test suite and CPU smoke pipeline, against the code as it is now
    on your machine (the same checks the RunPod STOP gate requires)."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    env.pop("THESIS_CONFIG_OVERLAY", None)
    commands = [[sys.executable, "-m", "compileall", "-q", "scripts", "tests"],
                [sys.executable, "tests/run_all.py"]]
    if smoke:
        commands.append([sys.executable, "scripts/smoke/run_smoke_pipeline.py", "--phase", "local"])
    for command in commands:
        print(f"\n$ {' '.join(command)}", flush=True)
        code = subprocess.run(command, cwd=REPO, env=env).returncode
        print(f"exit {code}", flush=True)
        if code != 0:
            raise SystemExit(f"FAILED: {' '.join(command)}")
    print("\nALL CHECKS PASSED")


@app.function(image=image, cpu=2.0, memory=4096, timeout=1800)
def run_namespace_regression() -> None:
    """CPU: the namespace regression test on the current code (must pass) and on a copy with the
    `cfg.split_namespace = namespace` fix removed from the six learned stages (must fail) — proof
    that the test detects the bug rather than passing vacuously."""
    import shutil

    test = "tests/test_learned_asism.py::test_learned_stage_uses_cli_namespace_not_config_default"
    stages = ["05_train_learned_asism.py", "06_learn_thresholds_select.py",
              "07b_verify_thresholds_proxy.py", "08_train_threshold_network.py",
              "08b_verify_full_policy_proxy.py", "09_finalize_learned_selection.py"]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    env.pop("THESIS_CONFIG_OVERLAY", None)

    print("### with the fix (current code): expect 6 passed", flush=True)
    fixed = subprocess.run([sys.executable, "-m", "pytest", "-q", test], cwd=REPO, env=env).returncode

    control = Path("/tmp/repo_without_fix")
    shutil.rmtree(control, ignore_errors=True)
    shutil.copytree(REPO, control)
    for name in stages:
        path = control / "scripts" / "asism" / name
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [line for line in lines if line.strip() != "cfg.split_namespace = namespace"]
        assert len(kept) == len(lines) - 1, f"expected exactly one fix line in {name}"
        path.write_text("".join(kept), encoding="utf-8")
    print("\n### without the fix (negative control): expect 6 failed", flush=True)
    unfixed = subprocess.run([sys.executable, "-m", "pytest", "-q", test], cwd=control, env=env).returncode

    print(f"\nwith fix exit={fixed} (0 = all passed) | without fix exit={unfixed} (non-zero = detected)")
    if fixed != 0 or unfixed == 0:
        raise SystemExit("namespace regression test did not behave as expected")
    print("NAMESPACE REGRESSION TEST: passes with the fix, fails without it")


@app.function(image=image, gpu=GPU, volumes={VOL_MOUNT: volume}, cpu=CPU_CORES,
              memory=32 * 1024, timeout=3 * 3600)
def run_gpu_smoke() -> None:
    """GPU: the repo's full end-to-end smoke (scripts/smoke/run_smoke_pipeline.py --phase all) on
    the 60-patient fixture — Stage 1 LoRA -> Stage 2 -> auxiliary classifier -> all ASISM signals ->
    learned ASISM 04-09 -> Stage 4 -> Stage 5, every GPU code path at toy size. Fixture pilot is
    approved with --approve-smoke-pilot (namespace dev-smoke-v1 only; never production)."""
    workspace = Path(REPO) / "outputs" / "smoke" / "dev-smoke-v1"
    hf_link = workspace / ".cache" / "huggingface"
    hf_link.parent.mkdir(parents=True, exist_ok=True)
    if not hf_link.exists():
        hf_link.symlink_to(HF_CACHE, target_is_directory=True)   # reuse the SDXL already on the volume
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1",
           "TQDM_MININTERVAL": "30"}
    env.pop("THESIS_CONFIG_OVERLAY", None)
    started = time.time()
    code = subprocess.run([sys.executable, "scripts/smoke/run_smoke_pipeline.py", "--phase", "all",
                           "--approve-smoke-pilot"], cwd=REPO, env=env).returncode
    minutes = (time.time() - started) / 60
    print(f"\nGPU SMOKE exit={code} in {minutes:.1f} min (~${_cost(minutes * 60):.2f} on {_detected_gpu()})")
    if code != 0:
        raise SystemExit(f"GPU smoke failed (exit {code})")


@app.function(image=image, volumes={VOL_MOUNT: volume}, cpu=2.0, memory=4096, timeout=3600)
def download_models() -> None:
    """CPU-only: fetch the pinned SDXL files into the shared cache (so no GPU idles on a download)."""
    from huggingface_hub import snapshot_download

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    started = time.time()
    path = snapshot_download(SDXL_ID, revision=SDXL_REVISION, cache_dir=HF_CACHE,
                             allow_patterns=SDXL_FILES)
    volume.commit()
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1024 ** 3
    print(f"SDXL {SDXL_REVISION[:8]} -> {path}  ({size:.1f} GiB, {time.time() - started:.0f}s)")


# ------------------------------------------------------------------------------------------------
# measurement helpers
# ------------------------------------------------------------------------------------------------

class GpuSampler:
    """Samples GPU utilization (%) from nvidia-smi, and the TOTAL resident memory of this process
    plus all its children (DataLoader workers included), while a block runs."""

    def __init__(self, interval: float = 0.25):
        self.interval, self.samples, self.rss_peak, self._stop = interval, [], 0, threading.Event()

    def _tree_rss(self) -> int:
        """Physical RAM actually used by the container. Summing RSS over forked DataLoader workers
        would count the parent's copy-on-write pages once per worker, so prefer the cgroup
        counter; fall back to main RSS + each child's unique (USS) memory."""
        for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
            try:
                return int(Path(path).read_text().strip())
            except (OSError, ValueError):
                continue
        import psutil

        me = psutil.Process()
        total = me.memory_info().rss
        for child in me.children(recursive=True):
            try:
                total += child.memory_full_info().uss
            except psutil.Error:
                pass
        return total

    def _loop(self) -> None:
        while not self._stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True)
            try:
                self.samples.append(float(out.stdout.strip().splitlines()[0]))
            except (ValueError, IndexError):
                pass
            self.rss_peak = max(self.rss_peak, self._tree_rss())
            time.sleep(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)

    @property
    def mean(self) -> float:
        return round(sum(self.samples) / len(self.samples), 1) if self.samples else float("nan")


def _detected_gpu() -> str:
    """The GPU this container really got (BENCH_GPU is only read on your machine)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.upper()
    for key in ("H100", "A100", "L40S", "A10", "L4", "T4"):
        if key in out:
            return "A100-40GB" if key == "A100" else key
    return GPU


def _cost(seconds: float) -> float:
    hourly = (GPU_PRICE_PER_HOUR.get(_detected_gpu(), 0) + CPU_CORES * CPU_PRICE_CORE_HOUR
              + MEMORY_GIB * MEM_PRICE_GIB_HOUR)
    return seconds / 3600 * hourly


def _measure(label: str, fn):
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with GpuSampler() as sampler:
        started = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    stats = {"seconds": round(seconds, 2), "gpu_util_pct": sampler.mean,
             "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
             "peak_ram_gib": round(sampler.rss_peak / 1024 ** 3, 2),
             "cost_usd": round(_cost(seconds), 4)}
    print(f"  {label:34s} {json.dumps(stats)}", flush=True)
    return result, stats


def _synthetic_cxr_jpegs(directory: Path, count: int, size: int = 768, seed: int = 0) -> list[Path]:
    """CXR-like 768px JPEGs (smooth anatomy-scale structure + fine noise, grey RGB, quality 95 —
    the same size/format 03_preprocess_images.py writes), so decode cost matches the real data."""
    import numpy as np
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    paths = []
    for index in range(count):
        path = directory / f"img_{index:05d}.jpg"
        paths.append(path)
        if path.exists():
            continue
        fx, fy, phase = rng.uniform(2, 9), rng.uniform(2, 9), rng.uniform(0, 6.3)
        base = 0.5 + 0.25 * np.sin(fx * 6.28 * xx + phase) * np.cos(fy * 6.28 * yy)
        base += rng.normal(0, 0.06, (size, size))
        gray = np.clip(base * 255, 0, 255).astype(np.uint8)
        Image.fromarray(gray, mode="L").convert("RGB").save(path, format="JPEG", quality=95)
    return paths


def _set_mode(optimized: bool) -> None:
    """Toggle the optimization flags the repo code reads; '0' restores the original behaviour."""
    import torch

    os.environ["THESIS_LOADER_WORKERS"] = "" if optimized else "0"
    os.environ["THESIS_CUDNN_BENCHMARK"] = "1" if optimized else "0"
    torch.backends.cudnn.benchmark = False   # the repo code turns it on when optimized


# ------------------------------------------------------------------------------------------------
# 1. classifier data loading (scripts/utils/classifier.py)
# ------------------------------------------------------------------------------------------------

@app.function(image=image, gpu=GPU, cpu=CPU_CORES, memory=MEMORY_GIB * 1024, timeout=3600)
def bench_classifier(n_train: int = 1536, n_eval: int = 1536, steps: int = 48,
                     mc_passes: int = 5, mc_images: int = 256) -> dict:
    """Original (num_workers=0) vs optimized (workers + pinned memory + cudnn.benchmark) for the
    three ways the pipeline uses the classifier: training steps, a full evaluation pass, and MC
    Dropout passes. Equivalence: batches must be bit-identical; predictions and trained weights
    may differ only by GPU floating-point noise (measured by repeating the original run)."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, REPO)
    os.chdir(REPO)
    import scripts.utils.classifier as clf

    print(f"GPU {_detected_gpu()}  cpu={CPU_CORES}  os.cpu_count={os.cpu_count()}  torch {torch.__version__}")
    rng = np.random.default_rng(1)
    paths = _synthetic_cxr_jpegs(Path("/tmp/bench_cxr"), n_train + n_eval)
    labels = clf.CLASSIFIER_TARGET_LABELS
    choices = np.array([1.0, 0.0, -1.0, np.nan])

    def make(p):
        return {"image_path": str(p), "is_synthetic": False,
                "labels": {l: float(rng.choice(choices, p=[.2, .5, .1, .2])) for l in labels}}

    train_records = [make(p) for p in paths[:n_train]]
    eval_records = [make(p) for p in paths[n_train:]]
    mc_records = eval_records[:mc_images]

    # --- equivalence of the data pipeline itself: identical batches, identical order ----------
    def first_batches(optimized: bool, k: int = 12):
        _set_mode(optimized)
        ds = clf.CXRRecordDataset(train_records, resolution=320)
        loader = DataLoader(ds, batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(7),
                            **clf._loader_settings("cuda", None))
        out = []
        for i, b in enumerate(loader):
            out.append(b)
            if i + 1 == k:
                break
        return out

    a, b = first_batches(False), first_batches(True)
    batches_identical = all(torch.equal(x[key], y[key]) for x, y in zip(a, b)
                            for key in ("image", "target", "mask", "index"))
    print(f"data pipeline: first {len(a)} batches bit-identical = {batches_identical}")

    budget_kw = dict(max_steps=steps, batch_size=32, learning_rate=1e-4, eval_every_n_steps=0, seed=42)

    # Warm-up (not measured): CUDA context, ImageNet weight download, first-kernel compilation.
    # Without it the first measured mode would pay one-off costs and look slower than it is.
    _set_mode(False)
    warm = clf.train_classifier(train_records[:64], [], clf.TrainingBudget(**{**budget_kw, "max_steps": 2}),
                                0.2, "imagenet", 320, progress_desc="warmup")[0]
    clf.predict_probabilities(warm, eval_records[:64], 320)
    del warm
    torch.cuda.empty_cache()

    results, models, predictions = {}, {}, {}
    for mode in ("original", "original_repeat", "optimized"):
        optimized = mode == "optimized"
        _set_mode(optimized)
        print(f"\n[{mode}] loader={clf._loader_settings('cuda', None)}")
        model, train_stats = _measure("train %d steps" % steps, lambda: clf.train_classifier(
            train_records, [], clf.TrainingBudget(**budget_kw), 0.2, "imagenet", 320,
            progress_desc=mode)[0])
        probs, eval_stats = _measure("eval pass %d imgs" % n_eval,
                                     lambda: clf.predict_probabilities(model, eval_records, 320)[0])
        (_, mc_std), mc_stats = _measure("mc-dropout %dx%d imgs" % (mc_passes, mc_images),
                                         lambda: clf.predict_probabilities(
                                             model, mc_records, 320, mc_dropout_passes=mc_passes))
        results[mode] = {
            "train": {**train_stats, "img_per_s": round(steps * 32 / train_stats["seconds"], 1)},
            "eval": {**eval_stats, "img_per_s": round(n_eval / eval_stats["seconds"], 1)},
            "mc_dropout": {**mc_stats, "img_per_s": round(mc_passes * mc_images / mc_stats["seconds"], 1)},
        }
        models[mode] = {k: v.detach().float().cpu() for k, v in model.state_dict().items()
                        if v.dtype.is_floating_point}
        predictions[mode] = probs

    def weight_diff(x, y):
        return max(float((models[x][k] - models[y][k]).abs().max()) for k in models[x])

    equivalence = {
        "batches_bit_identical": batches_identical,
        "weights_maxabs_optimized_vs_original": weight_diff("optimized", "original"),
        "weights_maxabs_noise_floor_original_vs_repeat": weight_diff("original_repeat", "original"),
        "probs_maxabs_optimized_vs_original": float(np.abs(predictions["optimized"] - predictions["original"]).max()),
        "probs_maxabs_noise_floor": float(np.abs(predictions["original_repeat"] - predictions["original"]).max()),
    }
    speedup = {phase: round(results["original"][phase]["seconds"] / results["optimized"][phase]["seconds"], 2)
               for phase in ("train", "eval", "mc_dropout")}
    report = {"gpu": _detected_gpu(), "results": results, "speedup": speedup, "equivalence": equivalence}
    print("\n" + json.dumps({"speedup": speedup, "equivalence": equivalence}, indent=2))
    return report


def _load_script(relative: str, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, f"{REPO}/{relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------------------------
# 4. Stage 1 training step: gradient checkpointing and micro-batch shape (trainer's own model)
# ------------------------------------------------------------------------------------------------

@app.function(image=image, gpu=GPU, volumes={VOL_MOUNT: volume}, cpu=CPU_CORES,
              memory=32 * 1024, timeout=3600)
def bench_stage1_step(variants: str = "8x2:on,16x1:on,4x4:off", steps: int = 6,
                      max_train_steps: int = 15000) -> dict:
    """Seconds per optimizer step and peak VRAM for the Stage 1 step (768px, bf16, LoRA r32, EMA),
    built by the trainer's own build_models(). Each variant is micro-batch x accumulation, so every
    optimizer step still averages the loss over 16 samples; ':on'/':off' is gradient checkpointing.
    8x2:on is the frozen configuration. Cached-latent training reads no images, so synthetic
    latents and embeddings give the same compute as the real run."""
    import torch
    import torch.nn.functional as F
    from accelerate import Accelerator
    from diffusers.optimization import get_scheduler

    _repo_env()
    trainer = _load_script("scripts/train/train_lora_sdxl.py", "trainer")
    from scripts.utils.config import load_stage1_config

    cfg = load_stage1_config()
    res = int(cfg.data.resolution)
    models = trainer.build_models(cfg, torch.bfloat16)
    for frozen_only in ("vae", "text_encoder_one", "text_encoder_two"):
        models.pop(frozen_only)
    unet, scheduler = models["unet"], models["noise_scheduler"]

    # Accumulation is done by hand (loss / A per micro-batch, one optimizer step) so the batch
    # shape can change between variants without rebuilding the model; the arithmetic per optimizer
    # step is the same the trainer performs through accelerator.accumulate().
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")
    params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg.optimizer.learning_rate), weight_decay=float(cfg.optimizer.weight_decay))
    lr_scheduler = get_scheduler(str(cfg.optimizer.lr_scheduler), optimizer=optimizer,
                                 num_warmup_steps=int(cfg.optimizer.lr_warmup_steps), num_training_steps=max_train_steps)
    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)
    ema = trainer.LoraEMA(accelerator.unwrap_model(unet), float(cfg.training.ema.decay))
    device, dtype = accelerator.device, torch.bfloat16

    def batch_of(size):
        return {"latents": torch.randn(size, 4, res // 8, res // 8, device=device, dtype=dtype),
                "prompt_embeds": torch.randn(size, 77, 2048, device=device, dtype=dtype),
                "pooled_embeds": torch.randn(size, 1280, device=device, dtype=dtype),
                "time_ids": torch.tensor([[res, res, 0, 0, res, res]] * size, device=device, dtype=dtype)}

    def optimizer_steps(n, micro, accumulation, fake):
        for _ in range(n):
            for _ in range(accumulation):
                noise = torch.randn_like(fake["latents"])
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (micro,), device=device).long()
                noisy = scheduler.add_noise(fake["latents"], noise, timesteps)
                pred = unet(noisy, timesteps, encoder_hidden_states=fake["prompt_embeds"],
                            added_cond_kwargs={"text_embeds": fake["pooled_embeds"], "time_ids": fake["time_ids"]},
                            return_dict=False)[0]
                accelerator.backward(F.mse_loss(pred.float(), noise.float()) / accumulation)
            accelerator.clip_grad_norm_(params, float(cfg.optimizer.max_grad_norm))
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(accelerator.unwrap_model(unet))

    price = GPU_PRICE_PER_HOUR.get(_detected_gpu(), 0)
    results = {}
    for variant in [v.strip() for v in variants.split(",") if v.strip()]:
        shape, gc = variant.split(":")
        micro, accumulation = (int(x) for x in shape.split("x"))
        raw = accelerator.unwrap_model(unet)
        raw.enable_gradient_checkpointing() if gc == "on" else raw.disable_gradient_checkpointing()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        try:
            fake = batch_of(micro)
            optimizer_steps(1, micro, accumulation, fake)   # warm-up
            _, stats = _measure(f"{variant}: {steps} optimizer steps",
                                lambda: optimizer_steps(steps, micro, accumulation, fake))
            per_step = stats["seconds"] / steps
            results[variant] = {**stats, "s_per_step": round(per_step, 3),
                                "stage1_hours": round(per_step * max_train_steps / 3600, 1),
                                "stage1_gpu_cost_usd": round(per_step * max_train_steps / 3600 * price, 1)}
        except torch.cuda.OutOfMemoryError as exc:
            results[variant] = {"oom": True, "error": str(exc).splitlines()[0][:160],
                                "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)}
            print(f"  {variant}: CUDA OOM on {_detected_gpu()}")
        finally:
            fake = None
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    report = {"gpu": _detected_gpu(),
              "vram_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1),
              "results": results}
    print()
    print(json.dumps(report, indent=2))
    return report
