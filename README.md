# LusCara Projeto

Projeto pessoal para geração e avaliação de embeddings Facenet em ONNX, com duas partes principais:

- `cpp/`: binários C++ (facenet_embedder e benchmark_lfw) com interface binária via stdin/stdout.
- `LuscaUtils/`: scripts Python para avaliação 10-fold, busca de thresholds e benchmark de latência.

## Estrutura

- `cpp/` — código C++, CMake, ferramentas de benchmark.
- `LuscaUtils/` — scripts Python e datasets auxiliares (não versionar os dados grandes).
- `models/` — modelos ONNX (ignorado no git).
- `third_party/` — dependências vendorizadas (ex.: ONNX Runtime).

## Requisitos

- C++: g++17, OpenCV 4, ONNX Runtime (headers + libs).
- Python: `numpy`, `pillow`, `onnxruntime`, `opencv-python` (veja `LuscaUtils/README.md`).

## Build rápido (C++)

```bash
cmake -S cpp -B build \
  -DONNX_DIR=/caminho/onnxruntime   # opcional; default usa third_party
cmake --build build --target facenet_embedder benchmark_lfw
```

## Uso resumido

- Binário principal: leia/ escreva em binário (vide `cpp/README.md`).
- Benchmark C++: `./benchmark_lfw` (veja envs `FACENET_INTRA_THREADS`, `BENCH_BATCH_IMAGES`).
- Scripts Python: veja `LuscaUtils/README.md` para exemplos (`evaluate_10fold.py`, `find_lfw_thresholds.py`, `benchmark_lfw.py`).

## Dados e modelos

- Modelos ONNX em `models/` (ex.: `facenet.onnx`, `facenet_int8.onnx`).
- LFW e CSVs de pares em `LuscaUtils/datasets/` (não versionados).
