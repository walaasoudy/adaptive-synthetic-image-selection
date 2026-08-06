"""Load and resolve the Stage 1 YAML configs (configs/stage1_lora_sdxl.yaml,
configs/dataset_config.yaml) via OmegaConf, with PROJECT_ROOT-relative path resolution.

Every script under scripts/ should load config through this module rather than parsing YAML
directly, so path resolution (PROJECT_ROOT env var, see docs/stage1_plan.md §4) stays consistent.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf, DictConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def load_stage1_config(overrides: list[str] | None = None) -> DictConfig:
    """Load configs/stage1_lora_sdxl.yaml with PROJECT_ROOT-based interpolation resolved.

    `overrides` accepts OmegaConf dotlist entries, e.g. ["training.train_batch_size=4"], for
    CLI-driven experiment overrides without editing the checked-in YAML.
    """
    config = OmegaConf.load(CONFIGS_DIR / "stage1_lora_sdxl.yaml")
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(config)
    return config


def load_dataset_config() -> DictConfig:
    config = OmegaConf.load(CONFIGS_DIR / "dataset_config.yaml")
    OmegaConf.resolve(config)
    return config


def ensure_dirs(config: DictConfig) -> None:
    """Create the directories under config.paths that scripts write into (idempotent)."""
    for key in (
        "raw_dir",
        "processed_dir",
        "images_dir",
        "splits_dir",
        "captions_dir",
        "checkpoints_dir",
        "logs_dir",
        "outputs_dir",
        "hf_cache_dir",
    ):
        Path(config.paths[key]).mkdir(parents=True, exist_ok=True)
