# LuscaUtils (Python)

Scripts auxiliares para FaceNet (ONNX) em LFW: avaliar acurácia, buscar thresholds e benchmark de latência.

## Scripts
- `evaluate_10fold.py`: avalia acurácia 10-fold no LFW usando thresholds fornecidos (não mede latência).
- `find_lfw_thresholds.py`: encontra thresholds ótimos (train/test ou arquivo único de pares).
- `benchmark_lfw.py`: mede acurácia e latência do modelo ONNX em LFW (sem cache de embeddings).
- `PTQ.py`: quantização dinâmica (FP32 → INT8) com ONNX Runtime.

## Requisitos
```
numpy
pillow
onnxruntime
opencv-python
```
Opcional: `onnx` para `PTQ.py`.

## Exemplos rápidos
### 10-fold com thresholds já conhecidos
```bash
python3 evaluate_10fold.py \
  --threshold-fp32 1.1292 \
  --threshold-int8 1.1185
```

### Buscar thresholds (train/test CSVs)
```bash
python3 find_lfw_thresholds.py \
  --lfw-root datasets/lfw-deepfunneled \
  --pairs-train datasets/matchpairsDevTrain.csv \
  --mismatch-train datasets/mismatchpairsDevTrain.csv \
  --pairs-test datasets/matchpairsDevTest.csv \
  --mismatch-test datasets/mismatchpairsDevTest.csv
```

### Benchmark de latência/acurácia
```bash
python3 benchmark_lfw.py \
  --model ../models/facenet_int8.onnx \
  --threshold 1.1185 \
  --lfw-root datasets/lfw-deepfunneled \
  --pairs-match datasets/matchpairsDevTrain.csv datasets/matchpairsDevTest.csv \
  --pairs-mismatch datasets/mismatchpairsDevTrain.csv datasets/mismatchpairsDevTest.csv
```

## Dados e modelos
- Coloque os modelos ONNX em `models/` (ex.: `facenet.onnx`, `facenet_int8.onnx`).
- Coloque o LFW extraído em `datasets/lfw-deepfunneled/` e os CSVs de pares em `datasets/`.
- Dados grandes não devem ser versionados; mantenha-os fora do git ou use scripts de download próprios.
