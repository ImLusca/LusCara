"""
Avalia acurácia 10-fold no LFW para os modelos Facenet FP32 e INT8,
usando thresholds já obtidos previamente.

Por padrão, combina os arquivos:
- datasets/matchpairsDevTrain.csv
- datasets/mismatchpairsDevTrain.csv
- datasets/matchpairsDevTest.csv
- datasets/mismatchpairsDevTest.csv

Uso básico:
  python3 evaluate_10fold.py \
      --threshold-fp32 1.1292 \
      --threshold-int8 1.1185

Opções:
  --lfw-root            Raiz do LFW (default: datasets/lfw-deepfunneled)
  --providers           Provedores ONNX Runtime (default: CPUExecutionProvider)
  --folds               Número de folds (default: 10)
  --seed                Semente do embaralhamento (default: 42)
  --fp32-model          Caminho do modelo FP32 (default: models/facenet.onnx)
  --int8-model          Caminho do modelo INT8 (default: models/facenet_int8.onnx)
  --pairs-match         Lista de CSVs só de pares positivos
  --pairs-mismatch      Lista de CSVs só de pares negativos
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import onnxruntime as ort
from time import perf_counter


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
    if lines and lines[0].lower().startswith("name"):
        lines = lines[1:]
    pairs: List[Pair] = []
    for line in lines:
        if not line:
            continue
        name, idx1, idx2, *_ = line.split(",")
        pairs.append(Pair(ImageId(name, int(idx1)), ImageId(name, int(idx2)), True))
    return pairs


def read_mismatch_pairs(path: Path) -> List[Pair]:
    lines = path.read_text().strip().splitlines()
    if lines and lines[0].lower().startswith("name"):
        lines = lines[1:]
    pairs: List[Pair] = []
    for line in lines:
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
        self._path = model_path
        self._cache: dict[Path, np.ndarray] = {}

    def embed(self, image_path: Path) -> np.ndarray:
        if image_path in self._cache:
            return self._cache[image_path]
        x = preprocess_image(image_path)
        out = self.session.run(None, {self.input_name: x})[0].squeeze()
        norm = np.linalg.norm(out) + 1e-9
        emb = out / norm
        self._cache[image_path] = emb
        return emb


# -----------------------------------------------------------------------------
# Avaliação


def compute_distances(model: FacenetONNX, pairs: Iterable[Pair], root: Path) -> Tuple[np.ndarray, np.ndarray]:
    dists, labels = [], []
    for p in pairs:
        a = model.embed(p.left.path(root))
        b = model.embed(p.right.path(root))
        dists.append(np.linalg.norm(a - b))
        labels.append(1 if p.is_match else 0)
    return np.asarray(dists), np.asarray(labels)


def accuracy_at_threshold(distances: np.ndarray, labels: np.ndarray, thr: float) -> float:
    return float(((distances <= thr) == labels).mean())


def kfold_indices(n: int, k: int) -> List[np.ndarray]:
    idx = np.arange(n)
    return np.array_split(idx, k)


def evaluate_kfold(
    model: FacenetONNX,
    pairs: List[Pair],
    root: Path,
    thr: float,
    k: int,
    seed: int,
) -> Tuple[float, float, List[float]]:
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    dists, labels = compute_distances(model, pairs, root)
    folds = kfold_indices(len(pairs), k)
    accs = []
    for fold_idx in folds:
        acc = accuracy_at_threshold(dists[fold_idx], labels[fold_idx], thr)
        accs.append(acc)
    accs = np.asarray(accs)
    return float(accs.mean()), float(accs.std()), accs.tolist()


# -----------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Avaliação 10-fold no LFW usando thresholds pré-definidos.")
    p.add_argument("--lfw-root", type=Path, default=Path("datasets/lfw-deepfunneled"))
    p.add_argument("--providers", nargs="+", default=("CPUExecutionProvider",))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp32-model", type=Path, default=Path("models/facenet.onnx"))
    p.add_argument("--int8-model", type=Path, default=Path("models/facenet_int8.onnx"))
    p.add_argument("--threshold-fp32", type=float, default=1.1292)
    p.add_argument("--threshold-int8", type=float, default=1.1185)
    p.add_argument(
        "--pairs-match",
        nargs="+",
        type=Path,
        default=[
            Path("datasets/matchpairsDevTrain.csv"),
            Path("datasets/matchpairsDevTest.csv"),
        ],
    )
    p.add_argument(
        "--pairs-mismatch",
        nargs="+",
        type=Path,
        default=[
            Path("datasets/mismatchpairsDevTrain.csv"),
            Path("datasets/mismatchpairsDevTest.csv"),
        ],
    )
    return p.parse_args()


def load_all_pairs(match_files: Sequence[Path], mismatch_files: Sequence[Path]) -> List[Pair]:
    pairs: List[Pair] = []
    for f in match_files:
        if f.exists():
            pairs.extend(read_match_pairs(f))
    for f in mismatch_files:
        if f.exists():
            pairs.extend(read_mismatch_pairs(f))
    return pairs


def main() -> None:
    args = parse_args()
    pairs = load_all_pairs(args.pairs_match, args.pairs_mismatch)
    if not pairs:
        raise SystemExit("Nenhum par encontrado. Verifique os caminhos dos CSVs.")

    print(f"Total de pares carregados: {len(pairs)} ({sum(p.is_match for p in pairs)} match / "
          f"{sum(not p.is_match for p in pairs)} mismatch)")

    models = {
        "fp32": (FacenetONNX(args.fp32_model, args.providers), args.threshold_fp32),
        "int8": (FacenetONNX(args.int8_model, args.providers), args.threshold_int8),
    }

    for name, (model, thr) in models.items():
        mean_acc, std_acc, accs = evaluate_kfold(
            model=model,
            pairs=pairs.copy(),  # cópia para embaralhar sem afetar o outro modelo
            root=args.lfw_root,
            thr=thr,
            k=args.folds,
            seed=args.seed,
        )
        accs_pct = [a * 100 for a in accs]
        print(f"\nModelo {name} ({model._path}):")
        print(f"  Threshold fixo: {thr:.4f}")
        print(f"  Acurácia por fold (%): {[round(a,2) for a in accs_pct]}")
        print(f"  Média: {mean_acc*100:.2f}% | Desvio-padrão: {std_acc*100:.2f}%")
        # Latência não é mais reportada aqui; mantenha este bloco para métricas de acurácia apenas.


if __name__ == "__main__":
    main()
