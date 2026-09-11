from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import CONFIGS_DIR, ensure_dirs, load_dataset_config, load_stage1_config  # noqa: E402

VERIFY_SCRIPT = Path(__file__).resolve().parent / "01_verify_download.py"


def find_data_root(base: Path) -> Path:
    """Locate the directory that directly contains train.csv + a train/ subdir, searching a few
    likely nesting levels before falling back to a full recursive search."""
    candidates = [base, base / "CheXpert-v1.0-small", base / "CheXpert-v1.0"]
    for candidate in candidates:
        if (candidate / "train.csv").exists() and (candidate / "train").is_dir():
            return candidate

    for train_csv in base.rglob("train.csv"):
        if (train_csv.parent / "train").is_dir():
            return train_csv.parent

    raise FileNotFoundError(
        f"Could not locate a CheXpert 'train.csv' (with sibling 'train/' dir) anywhere under {base}"
    )


def link_or_copy(source: Path, target: Path, force_copy: bool) -> str:
    """Idempotent per-entry: never overwrites an existing target. Prefers a symlink (dataset stays
    out of the repo/persistent-volume duplication); falls back to copying if symlinking isn't
    permitted (e.g. Windows without Developer Mode / admin rights)."""
    if target.exists() or target.is_symlink():
        return "skipped (already present)"

    if not force_copy:
        try:
            target.symlink_to(source, target_is_directory=source.is_dir())
            return "symlinked"
        except (OSError, NotImplementedError) as e:
            print(
                f"  Symlink failed for {target.name} ({e}); falling back to copy. "
                "(On Windows, enable Developer Mode or run as Administrator to allow symlinks.)"
            )

    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return "copied"


def populate_raw_dir(data_root: Path, raw_dir: Path, force_copy: bool) -> dict[str, str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for entry in sorted(data_root.iterdir()):
        target = raw_dir / entry.name
        results[entry.name] = link_or_copy(entry, target, force_copy)
    return results


def patch_dataset_config_slug(slug: str) -> bool:
    """Update configs/dataset_config.yaml's kaggle_dataset_slug if it doesn't already match `slug`
    (idempotent text patch — preserves the rest of the file/comments, which OmegaConf round-tripping
    would not)."""
    config_path = CONFIGS_DIR / "dataset_config.yaml"
    text = config_path.read_text(encoding="utf-8")

    if f'kaggle_dataset_slug: "{slug}"' in text:
        return False

    pattern = re.compile(r'kaggle_dataset_slug:\s*".*?"')
    if not pattern.search(text):
        return False

    patched = pattern.sub(f'kaggle_dataset_slug: "{slug}"', text, count=1)
    config_path.write_text(patched, encoding="utf-8")
    return True


def run_verification(sample_images: int) -> int:
    result = subprocess.run([sys.executable, str(VERIFY_SCRIPT), "--sample-images", str(sample_images)])
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download and re-link even if raw_dir already looks valid")
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlinking into data/chexpert/raw/")
    parser.add_argument("--sample-images", type=int, default=200, help="Passed through to 01_verify_download.py")
    args = parser.parse_args()

    stage1_cfg = load_stage1_config()
    ensure_dirs(stage1_cfg)
    raw_dir = Path(stage1_cfg.paths.raw_dir)

    if not args.force and (raw_dir / "train.csv").exists() and (raw_dir / "valid.csv").exists():
        print(f"Found existing train.csv/valid.csv in {raw_dir}; verifying before deciding whether to download...")
        if run_verification(args.sample_images) == 0:
            print("\nAlready downloaded and verified — skipping download (idempotent). Use --force to redo it anyway.")
            return 0
        print("\nExisting data did not pass verification; proceeding to (re)download.\n")

    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Run: pip install kagglehub")
        return 1

    dataset_cfg = load_dataset_config()
    slug = dataset_cfg.source.kaggle_dataset_slug
    print(f"Downloading Kaggle dataset '{slug}' via kagglehub (this can take a while on first run)...")
    download_path = Path(kagglehub.dataset_download(slug))
    print(f"kagglehub cached the dataset at: {download_path}")

    data_root = find_data_root(download_path)
    print(f"Located dataset root: {data_root}")

    results = populate_raw_dir(data_root, raw_dir, force_copy=args.copy)
    for name, action in results.items():
        print(f"  {action}: {raw_dir / name}")

    if patch_dataset_config_slug(slug):
        print(f"Updated configs/dataset_config.yaml kaggle_dataset_slug to '{slug}'")

    print("\nRunning scripts/data/01_verify_download.py to confirm the download is usable...")
    rc = run_verification(args.sample_images)
    if rc != 0:
        print("\nDownload completed but verification FAILED — see errors above.")
        return rc

    print("\nOK: dataset downloaded, linked into data/chexpert/raw/, and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
