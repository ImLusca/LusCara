"""
Benchmark de acurácia e latência no LFW para modelos Facenet ONNX.

Saída:
  - Total de pares
  - Acurácia (%) usando distância L2 com threshold informado
  - Estatísticas de latência (mean, p50, p90, p95, p99) apenas da chamada
    de inferência (sem I/O de disco ou pré-processamento)

Exemplo:
  python3 benchmark_lfw.py \
      --model ../models/facenet_int8.onnx \
      --threshold 1.1185 \
      --lfw-root datasets/lfw-deepfunneled \
      --pairs-match datasets/matchpairsDevTrain.csv datasets/matchpairsDevTest.csv \
      --pairs-mismatch datasets/mismatchpairsDevTrain.csv datasets/mismatchpairsDevTest.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import onnxruntime as ort


# -----------------------------------------------------------------------------
# Dados


@dataclass(frozen=True)
class ImageId:
    person: str
    index: int

    def path(self, root: Path) -> Path:
        return root / self.person / f"{self.person}_{self.index:04d}.jpg"


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
        if not line:
            continue
        name, idx1, idx2, *_ = line.split(",")
        pairs.append(Pair(ImageId(name, int(idx1)), ImageId(name, int(idx2)), True))
    return pairs


def read_mismatch_pairs(path: Path) -> List[Pair]:
    lines = path.read_text().strip().splitlines()
    header_skipped = lines[1:] if lines and lines[0].lower().startswith("name") else lines
    pairs: List[Pair] = []
    for line in header_skipped:
        if not line:
            continue
        name1, idx1, name2, idx2, *_ = line.split(",")
        pairs.append(Pair(ImageId(name1, int(idx1)), ImageId(name2, int(idx2)), False))
    return pairs


# -----------------------------------------------------------------------------
# Modelo e pré-processamento


def preprocess_image(image_path: Path, size: int = 160) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((size, size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # [-1, 1]
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return arr[np.newaxis, ...]


class FacenetONNX:
    def __init__(self, model_path: Path, providers: Sequence[str]):
        self.session = ort.InferenceSession(str(model_path), providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name

    def embed_pair(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Executa inferência em batch=2 e retorna embeddings normalizados + tempo em ms."""
        batch = np.concatenate([a, b], axis=0)
        t0 = perf_counter()
        out = self.session.run(None, {self.input_name: batch})[0]
        t1 = perf_counter()
        embs = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
        latency_ms = (t1 - t0) * 1000
        return embs[0], embs[1], latency_ms


# -----------------------------------------------------------------------------
# Benchmark


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    arr = np.sort(np.asarray(values))
    k = (p / 100.0) * (len(arr) - 1)
    lo, hi = int(np.floor(k)), int(np.ceil(k))
    if lo == hi:
        return float(arr[lo])
    frac = k - lo
    return float(arr[lo] * (1 - frac) + arr[hi] * frac)


def load_pairs(match_files: Sequence[Path], mismatch_files: Sequence[Path]) -> List[Pair]:
    pairs: List[Pair] = []
    for f in match_files:
        if f.exists():
            pairs.extend(read_match_pairs(f))
    for f in mismatch_files:
        if f.exists():
            pairs.extend(read_mismatch_pairs(f))
    return pairs


def benchmark(
    model: FacenetONNX,
    pairs: Iterable[Pair],
    lfw_root: Path,
    threshold: float,
    image_size: int,
) -> None:
    correct = 0
    total = 0
    latencies: List[float] = []

    for pair in pairs:
        a = preprocess_image(pair.left.path(lfw_root), image_size)
        b = preprocess_image(pair.right.path(lfw_root), image_size)

        emb_a, emb_b, ms = model.embed_pair(a, b)
        latencies.append(ms)

        dist = l2_distance(emb_a, emb_b)
        pred = dist <= threshold
        correct += int(pred == pair.is_match)
        total += 1

    acc = correct / total if total else 0.0

    print(f"Total de pares: {total}")
    print(f"Acurácia: {acc*100:.2f}%")
    if latencies:
        mean = float(np.mean(latencies))
        print(
            "Latência runInference (ms): "
            f"mean={mean:.2f}, p50={percentile(latencies,50):.2f}, "
            f"p90={percentile(latencies,90):.2f}, p95={percentile(latencies,95):.2f}, "
            f"p99={percentile(latencies,99):.2f}, calls={len(latencies)}"
        )


# -----------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark LFW com Facenet ONNX (acurácia + latência)")
    p.add_argument("--model", type=Path, default=Path("../../models/facenet_int8.onnx"))
    p.add_argument("--threshold", type=float, default=1.1185)
    p.add_argument("--lfw-root", type=Path, default=Path("datasets/lfw-deepfunneled"))
    p.add_argument("--image-size", type=int, default=160)
    p.add_argument("--providers", nargs="+", default=("CPUExecutionProvider",))
    p.add_argument("--pairs-match", nargs="+", type=Path, default=(
        Path("datasets/matchpairsDevTrain.csv"),
        Path("datasets/matchpairsDevTest.csv"),
    ))
    p.add_argument("--pairs-mismatch", nargs="+", type=Path, default=(
        Path("datasets/mismatchpairsDevTrain.csv"),
        Path("datasets/mismatchpairsDevTest.csv"),
    ))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pairs = load_pairs(args.pairs_match, args.pairs_mismatch)
    if not pairs:
        raise SystemExit("Nenhum par encontrado. Verifique os CSVs.")

    model = FacenetONNX(args.model, args.providers)
    benchmark(
        model=model,
        pairs=pairs,
        lfw_root=args.lfw_root,
        threshold=args.threshold,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
