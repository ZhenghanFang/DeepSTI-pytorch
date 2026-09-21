#!/usr/bin/env python3
"""Generate DeepSTI training patches from the whole-image arrays."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np


PATCH_SIZE = 64
MIN_MASK_VOXELS = 32_768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Downloaded DeepSTI dataset root (default: data/synthetic)",
    )
    parser.add_argument(
        "--train_list",
        default="train_input.txt",
        help="List in sti_sub/partition/partition_data6_list",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report the patch count without writing files",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing patch arrays"
    )
    return parser.parse_args()


def patch_ranges(shape: tuple[int, ...]) -> tuple[list[int], list[int], list[int]]:
    if shape[:3] == (144, 144, 90):
        return [0, 20, 40, 60, 80], [0, 20, 40, 60, 80], [0, 26]
    if shape[:3] == (224, 224, 110):
        return (
            [0, 32, 64, 96, 128, 160],
            [0, 32, 64, 96, 128, 160],
            [0, 9, 18, 27, 36, 46],
        )
    raise ValueError(f"Unsupported whole-image shape: {shape[:3]}")


def save_array(path: Path, array: np.ndarray, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def link_or_copy(source: Path, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not overwrite:
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def read_training_requirements(path: Path) -> dict[str, dict[str, set[str]]]:
    requirements: dict[str, dict[str, set[str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            subject, patch, *orientations = line.split()
            item = requirements.setdefault(
                subject, {"patches": set(), "orientations": set()}
            )
            item["patches"].add(patch)
            item["orientations"].update(orientations)
    return requirements


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    whole = data_dir / "sti_sub/whole"
    partition = data_dir / "sti_sub/partition"
    list_path = partition / "partition_data6_list" / args.train_list
    requirements = read_training_requirements(list_path)

    total_patches = 0
    for subject, required in sorted(requirements.items()):
        tensor = np.load(
            whole / "sti_data" / subject / f"{subject}_sim_tensor.npy",
            mmap_mode="r",
        )
        anisotropy = np.load(
            whole / "ani_data" / subject / f"{subject}_sim_ani.npy",
            mmap_mode="r",
        )
        mask = np.load(
            whole / "mask_data" / subject / f"{subject}_sim_mask.npy",
            mmap_mode="r",
        )
        if tensor.shape[:3] != mask.shape or anisotropy.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch for {subject}: tensor={tensor.shape}, "
                f"anisotropy={anisotropy.shape}, mask={mask.shape}"
            )

        x_starts, y_starts, z_starts = patch_ranges(mask.shape)
        valid_patches: list[tuple[str, tuple[slice, slice, slice]]] = []
        patch_number = 0
        for x in x_starts:
            for y in y_starts:
                for z in z_starts:
                    region = np.s_[
                        x : x + PATCH_SIZE,
                        y : y + PATCH_SIZE,
                        z : z + PATCH_SIZE,
                    ]
                    if np.sum(mask[region]) > MIN_MASK_VOXELS:
                        valid_patches.append((f"p{patch_number}", region))
                        patch_number += 1

        generated_names = {name for name, _ in valid_patches}
        if generated_names != required["patches"]:
            missing = sorted(required["patches"] - generated_names)
            extra = sorted(generated_names - required["patches"])
            raise ValueError(
                f"Patch numbering mismatch for {subject}; "
                f"missing={missing}, extra={extra}"
            )

        total_patches += len(valid_patches)
        print(
            f"{subject}: {len(valid_patches)} patches, "
            f"{len(required['orientations'])} orientations"
        )
        if args.dry_run:
            continue

        orientations = sorted(required["orientations"])
        for patch_name, region in valid_patches:
            save_array(
                partition
                / "sti_pdata"
                / subject
                / f"{subject}_sim_tensor_{patch_name}.npy",
                tensor[region + (slice(None),)],
                args.overwrite,
            )
            save_array(
                partition
                / "ani_pdata"
                / subject
                / f"{subject}_sim_ani_{patch_name}.npy",
                anisotropy[region],
                args.overwrite,
            )

            first_mask = (
                partition
                / "mask_pdata"
                / subject
                / orientations[0]
                / f"{subject}_sim_{orientations[0]}_mask_{patch_name}.npy"
            )
            save_array(first_mask, mask[region], args.overwrite)
            for orientation in orientations[1:]:
                link_or_copy(
                    first_mask,
                    partition
                    / "mask_pdata"
                    / subject
                    / orientation
                    / f"{subject}_sim_{orientation}_mask_{patch_name}.npy",
                    args.overwrite,
                )

    action = "Validated" if args.dry_run else "Generated"
    print(f"{action} {total_patches} training patches in {data_dir}")


if __name__ == "__main__":
    main()
