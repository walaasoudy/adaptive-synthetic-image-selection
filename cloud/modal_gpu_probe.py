"""Which GPU types this Modal workspace may use (the Starter plan without a payment method
refuses some types at launch). Runs `nvidia-smi` for a few seconds on the requested GPU.

    set PROBE_GPU=L4 && modal run cloud/modal_gpu_probe.py
"""

import os

import modal

GPU = os.environ.get("PROBE_GPU", "T4")
app = modal.App(f"gpu-probe-{GPU.lower()}")


@app.function(gpu=GPU, image=modal.Image.debian_slim(), timeout=120)
def probe() -> str:
    import subprocess

    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"PROBE_OK {GPU}: {out}")
    return out
