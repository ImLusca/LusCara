"""
Finds the best verification threshold on LFW for both the floating‑point and
quantized Facenet ONNX models located in the local `models` folder.

Usage:
  python3 find_lfw_thresholds.py \
      --lfw-root datasets/lfw-deepfunneled \
      --pairs-train datasets/matchpairsDevTrain.csv \
      --mismatch-train datasets/mismatchpairsDevTrain.csv \
      --pairs-test datasets/matchpairsDevTest.csv \
      --mismatch-test datasets/mismatchpairsDevTest.csv

If you only want to sweep thresholds on the original LFW pairs file, pass
`--pairs-file datasets/pairs.csv` and omit the *_train/test flags.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image
import onnxruntime as ort


# -----------------------------------------------------------------------------
# Data loading


@dataclass(frozen=True)
class ImageId:
    person: str
    index: int

    def path(self, root: Path) -> Path:
        filename = f"{self.person}_{self.index:04d}.jpg"
        return root / self.person / filename


@dataclass
class Pair:
    left: ImageId
    right: ImageId
    is_match: bool


def read_match_pairs(path: Path) -> List[Pair]:
    lines = path.read_text().strip().splitlines()
    header_skipped = lines[1:] if lines and lines[0].lower().startswith("name") else lines
    pairs: List[Pair] = []
    for line in header_skipped:
        name, idx1, idx2, *_ = line.split(",")
        pairs.append(
            Pair(ImageId(name, int(idx1)), ImageId(name, int(idx2)), True)
        )
    return pairs


def read_mismatch_pairs(path: Path) -> List[Pair]:
    lines = path.read_text().strip().splitlines()
    header_skipped = lines[1:] if lines and lines[0].lower().startswith("name") else lines
    pairs: List[Pair] = []
    for line in header_skipped:
        name1, idx1, name2, idx2, *_ = line.split(",")
        pairs.append(
            Pair(ImageId(name1, int(idx1)), ImageId(name2, int(idx2)), False)
        )
    return pairs


def load_pairs_from_single_file(path: Path) -> List[Pair]:
    """
    LFW's original pairs.txt/csv can contain both match and mismatch blocks.
    In this repo version (pairs.csv) we only have matches, so this helper
    assumes matches. If you provide a mixed file, extend this parser.
    """
    return read_match_pairs(path)


# -----------------------------------------------------------------------------
# Model + preprocessing


def preprocess_image(image_path: Path, size: int = 160) -> np.ndarray:
    """
    Returns an array shaped (1, 3, size, size) normalized to [-1, 1].
    """
    img = Image.open(image_path).convert("RGB").resize((size, size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # scale to [-1, 1]
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return arr[np.newaxis, ...]


class FacenetONNX:
    def __init__(self, model_path: Path, providers: Tuple[str, ...]):
        self.session = ort.InferenceSession(
            str(model_path),
            providers=list(providers),
        )
        self.input_name = self.session.get_inputs()[0].name

    @lru_cache(maxsize=8192)
    def embed(self, image_path: Path) -> np.ndarray:
        x = preprocess_image(image_path)
        outputs = self.session.run(None, {self.input_name: x})
        emb = outputs[0].squeeze()
        norm = np.linalg.norm(emb) + 1e-9
        return emb / norm


# -----------------------------------------------------------------------------
# Threshold search


def compute_distances(
    model: FacenetONNX, pairs: Iterable[Pair], lfw_root: Path
) -> Tuple[np.ndarray, np.ndarray]:
    dists = []
    labels = []
    for pair in pairs:
        left = model.embed(pair.left.path(lfw_root))
        right = model.embed(pair.right.path(lfw_root))
        dist = np.linalg.norm(left - right)
        dists.append(dist)
        labels.append(1 if pair.is_match else 0)
    return np.asarray(dists), np.asarray(labels)


def best_threshold(distances: np.ndarray, labels: np.ndarray, steps: int = 400) -> Tuple[float, float]:
    """
    Returns (threshold, accuracy) that maximizes verification accuracy.
    """
    low, high = distances.min(), distances.max()
    thresholds = np.linspace(low, high, steps)
    best_acc = -1.0
    best_thr = thresholds[0]
    for thr in thresholds:
        preds = distances <= thr
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return float(best_thr), float(best_acc)


def evaluate_split(model: FacenetONNX, pairs: List[Pair], lfw_root: Path) -> Tuple[float, float]:
    dists, labels = compute_distances(model, pairs, lfw_root)
    return best_threshold(dists, labels)


# -----------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find ideal LFW thresholds for Facenet models.")
    p.add_argument("--lfw-root", type=Path, default=Path("datasets/lfw-deepfunneled"))
    p.add_argument("--pairs-train", type=Path, default=Path("datasets/matchpairsDevTrain.csv"))
    p.add_argument("--mismatch-train", type=Path, default=Path("datasets/mismatchpairsDevTrain.csv"))
    p.add_argument("--pairs-test", type=Path, default=Path("datasets/matchpairsDevTest.csv"))
    p.add_argument("--mismatch-test", type=Path, default=Path("datasets/mismatchpairsDevTest.csv"))
    p.add_argument(
        "--pairs-file",
        type=Path,
        default=None,
        help="Optional single pairs file (all matches). If set, train/test flags are ignored.",
    )
    p.add_argument("--fp32-model", type=Path, default=Path("models/facenet.onnx"))
    p.add_argument("--int8-model", type=Path, default=Path("models/facenet_int8.onnx"))
    p.add_argument(
        "--providers",
        nargs="+",
        default=("CPUExecutionProvider",),
        help="ONNX Runtime providers to try (default: CPUExecutionProvider).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lfw_root = args.lfw_root

    if args.pairs_file:
        train_pairs = load_pairs_from_single_file(args.pairs_file)
        test_pairs = []
    else:
        train_pairs = read_match_pairs(args.pairs_train) + read_mismatch_pairs(args.mismatch_train)
        test_pairs = read_match_pairs(args.pairs_test) + read_mismatch_pairs(args.mismatch_test)

    print(f"Loaded {len(train_pairs)} train pairs"
          f"{' and ' + str(len(test_pairs)) + ' test pairs' if test_pairs else ''}.")

    models = {
        "fp32": FacenetONNX(args.fp32_model, tuple(args.providers)),
        "int8": FacenetONNX(args.int8_model, tuple(args.providers)),
    }

    for name, model in models.items():
        print(f"\nEvaluating {name} model ({model.session._model_path}):")
        thr_train, acc_train = evaluate_split(model, train_pairs, lfw_root)
        print(f"  Train threshold: {thr_train:.4f} | accuracy: {acc_train*100:.2f}%")
        if test_pairs:
            dists_test, labels_test = compute_distances(model, test_pairs, lfw_root)
            acc_test = ((dists_test <= thr_train) == labels_test).mean()
            print(f"  Test accuracy @ train threshold: {acc_test*100:.2f}%")
            thr_test, acc_test_best = best_threshold(dists_test, labels_test)
            print(f"  Best test threshold: {thr_test:.4f} | accuracy: {acc_test_best*100:.2f}%")
        else:
            print("  No test split provided.")


if __name__ == "__main__":
    main()
