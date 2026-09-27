"""
CT / MRI Brain Scan Preprocessing Pipeline
===========================================
Standalone script version of main_program.ipynb.

Usage:
    python pipeline.py

All parameters are controlled via the CONFIG dictionary below.
Outputs:
    - Processed .npy slices under dataset/train|val|test/CT|MRI/images|masks/
    - dataset_manifest.csv  — paired image ↔ mask paths per slice
    - qc_report.csv         — volume-level quality control audit log
    - pipeline.log          — timestamped execution log
"""

import os
import sys
import json
import hashlib
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
import pydicom
import nibabel as nib
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# Edit the paths below to point to your dataset location.
# ─────────────────────────────────────────────────────────────
CONFIG = {
    # Input paths
    "ct_path":       "JUH_MR-CT_dataset/CT/image_CT",
    "mri_path":      "JUH_MR-CT_dataset/MR/image_MR",
    "ct_mask_root":  "JUH_MR-CT_dataset/CT/mask_CT",
    "mri_mask_root": "JUH_MR-CT_dataset/MR/mask_MR",

    # Output path
    "output_root":   "dataset",

    # CT windowing (Hounsfield Units — soft-tissue brain window)
    "ct_window_center": 40,
    "ct_window_width":  80,

    # MRI normalization (percentile clipping, background excluded)
    "mri_clip_low":  1.0,
    "mri_clip_high": 99.0,

    # Slice settings
    "target_size":        (256, 256),   # Output (H, W) per slice
    "min_brain_fraction": 0.02,         # Skip slices with < 2% non-zero tissue

    # Dataset split (must sum to 1.0)
    "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
    "seed":         42,                 # Random seed for reproducibility

    # Storage format: "npy" (lossless, recommended) or "png"
    "save_format":  "npy",
}


# ─────────────────────────────────────────────────────────────
# LOGGER SETUP
# ─────────────────────────────────────────────────────────────
def setup_logger() -> logging.Logger:
    """Configure a dual-handler logger: file (DEBUG) + console (INFO)."""
    log = logging.getLogger("pipeline")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"
    )

    # File handler — full DEBUG log written to pipeline.log
    fh = logging.FileHandler("pipeline.log", mode="w")
    fh.setFormatter(fmt)

    # Console handler — INFO and above shown in terminal
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    log.addHandler(fh)
    log.addHandler(sh)
    return log


LOG = setup_logger()


# ─────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────
def discover_files(root: str, modality: str, is_mask: bool = False) -> List[Dict]:
    """
    Recursively find all valid DICOM (.dcm) and NIfTI (.nii / .nii.gz) files.

    Applies the following filters:
      - Non-imaging extensions are skipped
      - Files smaller than 512 bytes are skipped (truncated / empty)
      - Mask files found inside an image directory are skipped (cross-contamination guard)

    Returns a list of metadata dictionaries, one per file.
    """
    root = Path(root)
    if not root.exists():
        LOG.error(f"Path not found: {root}")
        return []

    records = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue

        s = p.suffix.lower()
        if s not in (".dcm", ".nii", ".gz"):
            continue
        if s == ".gz" and ".nii" not in p.stem:
            continue
        if p.stat().st_size < 512:
            LOG.warning(f"Skipping tiny file: {p.name}")
            continue

        # Cross-contamination guard
        path_is_mask = "mask" in str(p).lower()
        if path_is_mask and not is_mask:
            LOG.warning(f"SKIPPED — mask file found in image folder: {p.name}")
            continue

        records.append({
            "path":      str(p),
            "modality":  modality,
            "file_type": "dicom" if s == ".dcm" else "nifti",
            "is_mask":   is_mask,
        })

    label = "masks" if is_mask else "images"
    LOG.info(f"{modality} ({label}): found {len(records)} files")
    return records


# ─────────────────────────────────────────────────────────────
# VALIDATION & LOADING
# ─────────────────────────────────────────────────────────────
def validate_and_load(record: Dict) -> Tuple[Optional[np.ndarray], str]:
    """
    Load a medical file and run structural quality-control checks.

    For DICOM CT scans, raw pixel values are converted to Hounsfield Units (HU):
        HU = pixel * RescaleSlope + RescaleIntercept

    Returns:
        (np.ndarray, 'ok')               on success
        (None,       '<error_code>')     on failure
    """
    try:
        if record["file_type"] == "dicom":
            ds   = pydicom.dcmread(record["path"])
            data = ds.pixel_array.astype(np.float32)
            if record["modality"] == "CT":
                data = (data * float(getattr(ds, "RescaleSlope", 1))
                        + float(getattr(ds, "RescaleIntercept", 0)))
        else:
            data = nib.load(record["path"]).get_fdata().astype(np.float32)

        if data.size == 0:             return None, "empty"
        if np.all(data == 0):          return None, "all_zeros"
        if np.any(np.isnan(data)):     return None, "contains_nan"
        if np.any(np.isinf(data)):     return None, "contains_inf"
        if data.ndim not in (2, 3, 4): return None, f"bad_ndim_{data.ndim}"

        return data, "ok"
    except Exception as exc:
        return None, f"load_error:{exc}"


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────
def preprocess_ct(data: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply HU windowing and min-max normalize CT to [0, 1]."""
    lo = cfg["ct_window_center"] - cfg["ct_window_width"] / 2
    hi = cfg["ct_window_center"] + cfg["ct_window_width"] / 2
    return np.clip(
        (np.clip(data, lo, hi) - lo) / (hi - lo + 1e-8), 0, 1
    ).astype(np.float32)


def preprocess_mri(data: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply percentile clipping on non-zero voxels and normalize MRI to [0, 1]."""
    nz = data[data > 0]
    if nz.size == 0:
        return data.astype(np.float32)
    lo = np.percentile(nz, cfg["mri_clip_low"])
    hi = np.percentile(nz, cfg["mri_clip_high"])
    return np.clip(
        (np.clip(data, lo, hi) - lo) / (hi - lo + 1e-8), 0, 1
    ).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# SLICE EXTRACTION & RESIZING
# ─────────────────────────────────────────────────────────────
def extract_and_resize(volume: np.ndarray, cfg: dict) -> List[np.ndarray]:
    """
    Convert a 3D volume into a list of valid 2D axial slices.

    - 4D volumes: first timepoint is used
    - 2D inputs: wrapped into a single-slice volume
    - Slices with < min_brain_fraction non-zero pixels are discarded
    - Each slice is resized to cfg['target_size'] using bilinear interpolation
    """
    if volume.ndim == 4:
        volume = volume[..., 0]
    if volume.ndim == 2:
        volume = volume[:, :, None]

    slices = []
    for z in range(volume.shape[2]):
        sl = volume[:, :, z]
        if np.count_nonzero(sl) / sl.size < cfg["min_brain_fraction"]:
            continue
        sl = cv2.resize(
            sl,
            (cfg["target_size"][1], cfg["target_size"][0]),
            interpolation=cv2.INTER_LINEAR,
        )
        slices.append(sl.astype(np.float32))
    return slices


# ─────────────────────────────────────────────────────────────
# SAVING
# ─────────────────────────────────────────────────────────────
def save_slice(sl: np.ndarray, out_dir: str, vol_id: str, index: int, fmt: str) -> None:
    """Save a preprocessed image slice to disk."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(f"{out_dir}/{vol_id}_z{index:04d}.npy", sl)
    else:
        cv2.imwrite(
            f"{out_dir}/{vol_id}_z{index:04d}.png",
            (sl * 255).clip(0, 255).astype(np.uint8),
        )


def save_mask_slice(sl: np.ndarray, out_dir: str, vol_id: str, index: int, fmt: str) -> None:
    """Save a segmentation mask slice using nearest-neighbor resizing to preserve labels."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sl_resized = cv2.resize(
        sl,
        (CONFIG["target_size"][1], CONFIG["target_size"][0]),
        interpolation=cv2.INTER_NEAREST,
    )
    if fmt == "npy":
        np.save(f"{out_dir}/{vol_id}_z{index:04d}.npy", sl_resized.astype(np.uint8))
    else:
        cv2.imwrite(f"{out_dir}/{vol_id}_z{index:04d}.png", sl_resized.astype(np.uint8))


# ─────────────────────────────────────────────────────────────
# PATIENT-LEVEL SPLITTING
# ─────────────────────────────────────────────────────────────
def split_records(records: List[Dict], cfg: dict) -> Dict[str, List[Dict]]:
    """
    Randomly split patient volumes into train / val / test sets.

    Splitting is done at the volume (patient) level — NOT at the slice level —
    to prevent patient identity leakage between splits.
    """
    rng   = np.random.default_rng(cfg["seed"])
    index = np.arange(len(records))
    rng.shuffle(index)

    n  = len(records)
    nt = int(n * cfg["split_ratios"]["train"])
    nv = int(n * cfg["split_ratios"]["val"])

    return {
        "train": [records[i] for i in index[:nt]],
        "val":   [records[i] for i in index[nt : nt + nv]],
        "test":  [records[i] for i in index[nt + nv :]],
    }


# ─────────────────────────────────────────────────────────────
# BATCH VOLUME PROCESSING
# ─────────────────────────────────────────────────────────────
def process_volume_list(
    vols: List[Dict],
    modality: str,
    split: str,
    cfg: dict,
    is_mask: bool,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Validate, preprocess, extract slices, and save a batch of patient volumes.

    Returns:
        qc_rows   — one QC record per volume (for qc_report.csv)
        manifest  — one record per saved slice (for dataset_manifest.csv)
    """
    subfolder = "masks" if is_mask else "images"
    out_dir   = f"{cfg['output_root']}/{split}/{modality}/{subfolder}"
    qc_rows   = []
    manifest  = []

    for rec in tqdm(vols, desc=f"{modality}/{split}/{subfolder}", leave=False):
        row  = {**rec, "split": split, "slices_saved": 0, "issue": ""}
        data, status = validate_and_load(rec)

        if status != "ok":
            row["issue"] = status
            LOG.warning(f"SKIP {Path(rec['path']).name}: {status}")
        else:
            try:
                vol_id = Path(rec["path"]).stem.replace(".nii", "")

                if is_mask:
                    vol = data
                    if vol.ndim == 4: vol = vol[..., 0]
                    if vol.ndim == 2: vol = vol[:, :, None]
                    slices = [
                        vol[:, :, z]
                        for z in range(vol.shape[2])
                        if np.count_nonzero(vol[:, :, z]) / vol[:, :, z].size
                        >= cfg["min_brain_fraction"]
                    ]
                    for i, sl in enumerate(slices):
                        save_mask_slice(sl, out_dir, vol_id, i, cfg["save_format"])
                else:
                    vol    = preprocess_ct(data, cfg) if modality == "CT" else preprocess_mri(data, cfg)
                    slices = extract_and_resize(vol, cfg)
                    for i, sl in enumerate(slices):
                        fpath = f"{out_dir}/{vol_id}_z{i:04d}.{cfg['save_format']}"
                        save_slice(sl, out_dir, vol_id, i, cfg["save_format"])
                        manifest.append({
                            "subject_id":  vol_id,
                            "modality":    modality,
                            "split":       split,
                            "slice_index": i,
                            "image_path":  fpath,
                            "mask_path":   "",
                            "source_file": rec["path"],
                        })

                row["slices_saved"] = len(slices)

            except Exception as exc:
                row["issue"] = str(exc)
                LOG.error(exc)

        qc_rows.append(row)

    return qc_rows, manifest


# ─────────────────────────────────────────────────────────────
# MANIFEST BUILDER
# ─────────────────────────────────────────────────────────────
def build_manifest(manifest_rows: List[Dict]) -> pd.DataFrame:
    """
    Join image slice records with their corresponding mask paths.

    Uses an O(N) dictionary lookup (rather than O(N²) nested loops) to pair
    each image slice with its matching mask based on:
        (subject_id, modality, split, slice_index)

    Also normalises mask filename suffixes (_m → _s) to handle
    naming mismatches in the raw clinical dataset.
    """
    df = pd.DataFrame(manifest_rows)
    if df.empty:
        return df

    img_df = df[df["modality"].isin(["CT", "MRI"])].copy()

    # Build mask lookup — O(N) hash-map construction
    mask_lookup: Dict[Tuple, str] = {}
    for modality in ["CT", "MRI"]:
        for split in ["train", "val", "test"]:
            mask_dir = Path(CONFIG["output_root"]) / split / modality / "masks"
            if not mask_dir.exists():
                continue
            for f in sorted(mask_dir.glob(f"*.{CONFIG['save_format']}")):
                parts = f.stem.rsplit("_z", 1)
                if len(parts) != 2:
                    continue
                vol_id, z_str = parts
                try:
                    z_idx = int(z_str)
                except ValueError:
                    continue
                # Normalise _m → _s to fix naming mismatch in raw dataset
                normalized_id = vol_id.replace("_m", "_s")
                mask_lookup[(normalized_id, modality, split, z_idx)] = str(f)

    def lookup_mask(row: pd.Series) -> str:
        key = (row["subject_id"], row["modality"], row["split"], row["slice_index"])
        return mask_lookup.get(key, "")

    img_df["mask_path"] = img_df.apply(lookup_mask, axis=1)

    cols = ["subject_id", "modality", "split", "slice_index",
            "image_path", "mask_path", "source_file"]
    img_df = img_df[cols].sort_values(["split", "modality", "subject_id", "slice_index"])
    return img_df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────
def run_pipeline(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full preprocessing pipeline:
      1. Discover files  →  2. Split patients  →  3. Preprocess & save slices
      4. Build manifest  →  5. Export QC report
    """
    qc, manifest = [], []

    jobs = [
        ("CT",  cfg["ct_path"],       False),
        ("MRI", cfg["mri_path"],      False),
        ("CT",  cfg["ct_mask_root"],  True),
        ("MRI", cfg["mri_mask_root"], True),
    ]

    for modality, root, is_mask in jobs:
        if not root:
            continue
        records = discover_files(root, modality, is_mask=is_mask)
        splits  = split_records(records, cfg)
        for split, vols in splits.items():
            qc_rows, mf_rows = process_volume_list(vols, modality, split, cfg, is_mask)
            qc.extend(qc_rows)
            manifest.extend(mf_rows)

    # Quality-control report (one row per volume)
    qc_df = pd.DataFrame(qc)
    qc_df.to_csv("qc_report.csv", index=False)

    # Training manifest (one row per slice, with paired image + mask paths)
    manifest_df = build_manifest(manifest)
    manifest_df.to_csv("dataset_manifest.csv", index=False)

    issues = (qc_df["issue"] != "").sum()
    LOG.info(f"Done — {len(manifest_df)} slices in manifest, {issues} volume issues")
    LOG.info("Manifest saved → dataset_manifest.csv")
    LOG.info("QC report  saved → qc_report.csv")

    print("\n--- Manifest preview (first 5 rows) ---")
    print(manifest_df.head().to_string(index=False))
    print(f"\nTotal slices: {len(manifest_df)}")
    print(manifest_df.groupby(["split", "modality"]).size().rename("slices").to_string())

    print("\nOutput structure:")
    for p in sorted(Path(cfg["output_root"]).rglob("*")):
        if p.is_dir():
            print(" ", p)

    return qc_df, manifest_df


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    QC, MANIFEST = run_pipeline(CONFIG)
