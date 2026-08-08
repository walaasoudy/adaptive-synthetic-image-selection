# Frozen RunPod dependency and model matrix

Production runs use a RunPod image with Python 3.11, CUDA 12.8, PyTorch 2.8.0,
torchvision 0.23.0, and one NVIDIA A40 48 GB GPU. Verify these exact runtime versions before any
paid training; `environment/requirements.txt` pins every application-layer dependency and must not
replace the image-provided PyTorch packages.

The two remotely resolved model inputs are immutable:

| Component | Repository | Revision | License declared upstream |
|---|---|---|---|
| SDXL base | `stabilityai/stable-diffusion-xl-base-1.0` | `462165984030d82259a11f4367a4eed129e94a7b` | OpenRAIL++ |
| DINOv2 ViT-S/14 | `timm/vit_small_patch14_dinov2.lvd142m` | `936966a8732c5442c9def5d126f2cc4ad4243dba` | Apache-2.0 |

Run this gate immediately after environment installation:

```bash
python -c "import torch,torchvision,diffusers,transformers,accelerate,peft,timm,pyarrow; print(torch.__version__,torch.version.cuda,torchvision.__version__)"
python -m compileall -q scripts tests
python tests/run_all.py
```

STOP if the printed torch/CUDA/torchvision versions differ, CUDA is unavailable, model revisions
cannot be resolved, or any test fails. After the successful pod smoke test, capture `pip freeze`,
the container image identifier, NVIDIA driver version, and `nvidia-smi` output in the run manifest.
