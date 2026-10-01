"""Proportional YOLO dataset sharding.

federated_training.build_yolo_shards splits a dataset into equal slices. The v4
scheduler produces a *sample vector* instead, one entry per admitted worker, so
this module materialises shards of arbitrary unequal sizes.

The v3 training pipeline is untouched: shards produced here have exactly the
layout worker.train_batch already expects, namely images/train, labels/train,
optional images/val and labels/val, and a data.yaml at the shard root.
"""

from __future__ import annotations

import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from federated_training import (
    IMAGE_SUFFIXES,
    discover_yolo_split_dirs,
    write_data_yaml,
    _copy_split_pairs,
)


def list_split_images(
    dataset_root: Path,
    manifest: Optional[Path] = None,
    splits: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the split directories plus the sorted image lists.

    A manifest restricts the training list to named files. That is how dataset
    subsets work: a 1000-image subset of a 10000-image parent is a list of
    filenames, not a second copy of the images on disk. Copying would make a
    scaling sweep across 100, 1000 and 10000 images cost several times the
    dataset in disk and minutes in setup, for no scientific gain.

    `splits` supplies the four directories directly, skipping discovery. Uploaded
    zips are found by directory name, but a standard dataset fetched through
    Ultralytics puts its images somewhere like VisDrone2019-DET-train/images,
    which no naming convention would guess. Its own config already names the
    paths, so when the caller has them it passes them in rather than searching.
    """
    if splits and splits.get("train_images"):
        split_dirs = {
            "root": Path(splits.get("root") or dataset_root),
            "train_images": Path(splits["train_images"]),
            "train_labels": Path(splits["train_labels"]) if splits.get("train_labels") else None,
            "val_images": Path(splits["val_images"]) if splits.get("val_images") else None,
            "val_labels": Path(splits["val_labels"]) if splits.get("val_labels") else None,
        }
        if split_dirs["train_labels"] is None:
            raise ValueError("The training labels directory could not be resolved")
    else:
        split_dirs = discover_yolo_split_dirs(dataset_root)
    train_images = sorted(
        path
        for path in split_dirs["train_images"].rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if manifest is not None and manifest.is_file():
        wanted = {
            line.strip()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        base = split_dirs["train_images"]
        train_images = [
            path for path in train_images if path.relative_to(base).as_posix() in wanted
        ]

    val_images: List[Path] = []
    if split_dirs["val_images"] is not None:
        val_images = sorted(
            path
            for path in split_dirs["val_images"].rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return {"dirs": split_dirs, "train": train_images, "val": val_images}


def write_subset_manifest(
    dataset_root: Path,
    destination: Path,
    sample_count: int,
    seed: int = 0,
    splits: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Choose a reproducible subset of the training split and record it.

    The validation split is deliberately untouched. Every subset in a scaling
    sweep is evaluated against exactly the same held-out images, otherwise a
    change in measured accuracy could be a change in the test set rather than a
    change in the training set, and the sweep would answer nothing.
    """
    listing = list_split_images(dataset_root, splits=splits)
    base = listing["dirs"]["train_images"]
    relative = [path.relative_to(base).as_posix() for path in listing["train"]]

    if sample_count >= len(relative):
        chosen = relative
    else:
        chosen = random.Random(seed).sample(relative, sample_count)

    chosen.sort()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(chosen), encoding="utf-8")
    return {
        "manifest_path": str(destination),
        "train_count": len(chosen),
        "val_count": len(listing["val"]),
        "parent_train_count": len(relative),
        "seed": seed,
    }


def count_training_samples(
    dataset_root: Path,
    manifest: Optional[Path] = None,
    splits: Optional[Dict[str, Any]] = None,
) -> int:
    return len(list_split_images(dataset_root, manifest=manifest, splits=splits)["train"])


def partition(images: Sequence[Path], sizes: Sequence[int], seed: int) -> List[List[Path]]:
    """Split images into contiguous groups of the requested sizes.

    The pool is shuffled with a per-round seed first. Sharding by sorted order
    would hand every worker a class-correlated slice, because most YOLO exports
    are ordered by capture session, and that biases the local gradients before
    aggregation ever sees them. A deterministic shuffle keeps each shard an
    unbiased sample while staying reproducible for the paper.
    """
    pool = list(images)
    random.Random(seed).shuffle(pool)

    groups: List[List[Path]] = []
    cursor = 0
    for size in sizes:
        groups.append(pool[cursor : cursor + size])
        cursor += size

    # Rounding can leave a tail. Give it to the largest shard, which by
    # construction belongs to the fastest worker.
    if cursor < len(pool) and groups:
        largest = max(range(len(groups)), key=lambda index: len(groups[index]))
        groups[largest].extend(pool[cursor:])
    return groups


def build_proportional_shards(
    dataset_root: Path,
    shard_output_dir: Path,
    sizes: Sequence[int],
    class_names: List[str],
    seed: int = 0,
    replicate_val: bool = True,
    manifest: Optional[Path] = None,
    splits: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Materialise one zipped shard per entry in sizes.

    Every shard carries the full validation split when replicate_val is set, so
    each worker reports mAP against the same held-out data and the per-round
    numbers are comparable across heterogeneous shards.
    """
    if not sizes:
        raise ValueError("At least one shard size is required")

    listing = list_split_images(dataset_root, manifest=manifest, splits=splits)
    split_dirs = listing["dirs"]
    train_images = listing["train"]
    val_images = listing["val"]

    if not train_images:
        raise ValueError("No training images were found in the dataset")

    groups = partition(train_images, sizes, seed)
    shard_output_dir.mkdir(parents=True, exist_ok=True)
    shards: List[Dict[str, Any]] = []

    for shard_index, shard_images in enumerate(groups):
        shard_root = shard_output_dir / ("shard_%d" % shard_index)
        if shard_root.exists():
            shutil.rmtree(shard_root)
        shard_root.mkdir(parents=True, exist_ok=True)

        train_count = _copy_split_pairs(
            shard_images,
            split_dirs["train_images"],
            split_dirs["train_labels"],
            shard_root,
            "train",
        )

        val_count = 0
        if replicate_val and split_dirs["val_images"] is not None and split_dirs["val_labels"] is not None:
            val_count = _copy_split_pairs(
                val_images,
                split_dirs["val_images"],
                split_dirs["val_labels"],
                shard_root,
                "val",
            )

        val_relative = "images/val" if val_count > 0 else "images/train"
        data_yaml = write_data_yaml(shard_root, class_names, val_relative=val_relative)

        shard_zip = shard_output_dir / ("shard_%d.zip" % shard_index)
        if shard_zip.exists():
            shard_zip.unlink()
        with zipfile.ZipFile(shard_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in shard_root.rglob("*"):
                if file_path.is_file():
                    archive.write(file_path, file_path.relative_to(shard_root))

        shards.append(
            {
                "shard_index": shard_index,
                "root": str(shard_root),
                "zip_path": str(shard_zip),
                "data_yaml_path": str(data_yaml),
                "train_count": train_count,
                "val_count": val_count,
                "bytes": shard_zip.stat().st_size,
            }
        )

    return shards


# ---------------------------------------------------------------------------
# v5: shards as file lists
# ---------------------------------------------------------------------------
#
# v4 materialised every shard every round: copy each image and label into a
# shard directory, then zip it with deflate. For a 1000-image round that is two
# thousand file copies and a compression pass over JPEGs that do not compress,
# on the coordinator, which in leg 1 was also one of the two workers and the
# one that trained at half speed. v5 plans a shard as a list of files. Workers
# keep the images they have seen and fetch only what is new (`bundle`), and the
# old zip is built on demand only for an agent that asks for it.


def plan_shard_lists(
    dataset_root: Path,
    sizes: Sequence[int],
    seed: int = 0,
    manifest: Optional[Path] = None,
    splits: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Partition the training images into shards of `sizes`, without copying."""
    if not sizes:
        raise ValueError("At least one shard size is required")
    listing = list_split_images(dataset_root, manifest=manifest, splits=splits)
    if not listing["train"]:
        raise ValueError("No training images were found in the dataset")
    groups = partition(listing["train"], sizes, seed)
    base = listing["dirs"]["train_images"]
    return {
        "dirs": listing["dirs"],
        "val": listing["val"],
        "groups": [[path.relative_to(base).as_posix() for path in group] for group in groups],
    }


def label_for(dirs: Dict[str, Any], image_rel: str) -> Optional[Path]:
    labels = dirs.get("train_labels")
    if labels is None:
        return None
    return Path(labels) / Path(image_rel).with_suffix(".txt")


def shard_manifest(dirs: Dict[str, Any], files: Sequence[str]) -> List[Dict[str, Any]]:
    """What a worker needs to check its cache: path and size of every pair."""
    base = Path(dirs["train_images"])
    entries = []
    for rel in files:
        image = base / rel
        label = label_for(dirs, rel)
        try:
            size = image.stat().st_size
        except OSError:
            continue
        label_size = label.stat().st_size if label is not None and label.is_file() else None
        entries.append({"image": rel, "size": size, "label_size": label_size})
    return entries


def _resolve_inside(base: Path, rel: str) -> Path:
    target = (base / rel).resolve()
    root = base.resolve()
    if os.path.commonpath([str(root), str(target)]) != str(root):
        raise ValueError("path escapes the dataset: %s" % rel)
    return target


def write_bundle(dirs: Dict[str, Any], files: Sequence[str], destination: Path) -> int:
    """Zip the requested image and label pairs, uncompressed, for a worker's cache.

    Members are named images/train/<rel> and labels/train/<rel>.txt, the layout
    the worker's cache and Ultralytics both expect. Stored rather than deflated:
    JPEG and PNG are already compressed, so deflate only spends host CPU.
    """
    base = Path(dirs["train_images"])
    count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for rel in files:
            image = _resolve_inside(base, rel)
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            archive.write(image, "images/train/%s" % rel)
            label = label_for(dirs, rel)
            if label is not None and label.is_file():
                archive.write(label, "labels/train/%s" % Path(rel).with_suffix(".txt").as_posix())
            count += 1
    return count


def materialize_shard_zip(
    dirs: Dict[str, Any],
    files: Sequence[str],
    val_images: Sequence[Path],
    class_names: List[str],
    workdir: Path,
    replicate_val: bool = True,
) -> Path:
    """The v4 shard archive, built only when an agent asks for one."""
    shard_root = workdir / "shard"
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)
    base = Path(dirs["train_images"])
    _copy_split_pairs([base / rel for rel in files], base, Path(dirs["train_labels"]), shard_root, "train")
    val_count = 0
    if replicate_val and dirs.get("val_images") is not None and dirs.get("val_labels") is not None:
        val_count = _copy_split_pairs(
            list(val_images), Path(dirs["val_images"]), Path(dirs["val_labels"]), shard_root, "val"
        )
    write_data_yaml(shard_root, class_names, val_relative="images/val" if val_count > 0 else "images/train")
    archive_path = workdir / "shard.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for file_path in shard_root.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, file_path.relative_to(shard_root))
    shutil.rmtree(shard_root, ignore_errors=True)
    return archive_path


def infer_class_names(dataset_root: Path) -> List[str]:
    """Read class names from a data.yaml shipped inside the uploaded dataset."""
    import yaml

    for candidate in sorted(dataset_root.rglob("*.yaml")) + sorted(dataset_root.rglob("*.yml")):
        try:
            config = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        names = config.get("names")
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names, key=lambda k: int(k))]
        if isinstance(names, list) and names:
            return [str(name) for name in names]
    return []
