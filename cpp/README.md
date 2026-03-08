# Facenet Embedder (C++)

Binários C++ para geração e benchmark de embeddings Facenet (ONNX Runtime), com entrada/saída binária para integração fácil com outras linguagens.

## Estrutura
- `facenet_embedder`: binário de produção (stdin → stdout).
- `benchmark_lfw`: benchmark de acurácia/latência no LFW.
- `src/`, `include/`: código-fonte principal.
- `tools/`: utilitários (benchmark).

## Dependências
- g++ (C++17)
- OpenCV 4 (core, imgproc, imgcodecs)
- ONNX Runtime (headers + libs). Default: `third_party/onnxruntime/onnxruntime-linux-x64-1.17.0`.

## Build
```bash
cmake -S cpp -B build \
  -DONNX_DIR=/caminho/onnxruntime   # opcional; default usa third_party
cmake --build build --target facenet_embedder benchmark_lfw
```

## Variáveis de ambiente relevantes
- `FACENET_MODEL_PATH`: caminho do modelo (default `../models/facenet_int8.onnx` a partir do binário).
- `FACENET_INTRA_THREADS`: controla threads intra-op do ORT. `1` (default) = single; `-1` = hardware_concurrency; `N>1` = força N.
- `BENCH_BATCH_IMAGES`: (benchmark) nº de imagens por chamada de inferência; precisa ser par. Ex.: `16` (8 pares).

## Interface binária (facenet_embedder)
Entrada (stdin, little-endian):
```
uint32 num_images
repetir num_images vezes:
  uint32 image_size
  byte image_data[image_size]
```
Saída (stdout):
```
uint32 embedding_size   # tipicamente 512
float embedding[embedding_size]  # L2-normalizado
```
Erros: nada em stdout, mensagem em stderr, exit != 0.

## Pipeline resumido
1) Decodifica e normaliza (RGB, 160x160, (x-0.5)/0.5, CHW).
2) Inference no ONNX Runtime.
3) L2-normaliza cada embedding.
4) Filtra outliers via similaridade com a média (threshold de similaridade). [Considere ajustar thresholds de filtro vs. decisão conforme seu caso de uso.]
5) Retorna a média final normalizada.

## Benchmark (benchmark_lfw)
Compara match/mismatch do LFW e mede latência de `runInference`.
Exemplo:
```bash
sudo env FACENET_INTRA_THREADS=1 BENCH_BATCH_IMAGES=2 \
  FACENET_MODEL_PATH=../models/facenet_int8.onnx \
  ./benchmark_lfw \
  --lfw-root ../../LuscaUtils/datasets/lfw-deepfunneled \
  --pairs-match ../../LuscaUtils/datasets/matchpairsDevTrain.csv,../../LuscaUtils/datasets/matchpairsDevTest.csv \
  --pairs-mismatch ../../LuscaUtils/datasets/mismatchpairsDevTrain.csv,../../LuscaUtils/datasets/mismatchpairsDevTest.csv \
  --threshold 1.1185
```
Ajuste `FACENET_INTRA_THREADS` e `BENCH_BATCH_IMAGES` para testar single vs multi-thread e batches maiores.

## Notas
- `INSTALL_RPATH` está configurado para `$ORIGIN/lib`; mantenha `libonnxruntime.so` acessível ao lado do binário ou ajuste `LD_LIBRARY_PATH`.
- Dados/modelos grandes não devem ser versionados; veja README na raiz para instruções de download.
