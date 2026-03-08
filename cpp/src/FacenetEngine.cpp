#include <FacenetEngine.hpp>
#include <MathUtils.hpp>
#include <thread>

FacenetEngine::FacenetEngine(const std::string &modelPath) : env(ORT_LOGGING_LEVEL_WARNING, "Facenet"), session(initSession(modelPath)) {}

Ort::Session FacenetEngine::initSession(const std::string &modelPath)
{
  Ort::SessionOptions opts;
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  // Configuração de threads via env: FACENET_INTRA_THREADS
  // - ausente ou "1": single thread (bom para batch pequeno/latência)
  // - "-1": usa hardware_concurrency
  // - qualquer outro inteiro >1: força esse valor
  int intra = 1;
  if (const char *envThreads = std::getenv("FACENET_INTRA_THREADS"))
  {
    intra = std::atoi(envThreads);
    if (intra == -1)
    {
      auto hw = std::thread::hardware_concurrency();
      intra = hw > 0 ? static_cast<int>(hw) : 1;
    }
    if (intra <= 0)
      intra = 1;
  }
  opts.SetIntraOpNumThreads(intra);

  return Ort::Session(env, modelPath.c_str(), opts);
}

std::vector<std::vector<float>> FacenetEngine::runInference(const std::vector<float> &batchTensor, int batchSize)
{
  std::vector<int64_t> inputShape = {batchSize, CHANNELS, IMAGE_SIZE, IMAGE_SIZE};

  const size_t expected = static_cast<size_t>(batchSize) * CHANNELS * IMAGE_SIZE * IMAGE_SIZE;
  if (batchTensor.size() != expected)
    throw std::runtime_error("Tamanho inesperado do batchTensor");

  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      allocator.GetInfo(),
      const_cast<float *>(batchTensor.data()),
      batchTensor.size(),
      inputShape.data(),
      inputShape.size());

  auto inputName = session.GetInputNameAllocated(0, allocator);
  auto outputName = session.GetOutputNameAllocated(0, allocator);
  const char *inputNames[] = {inputName.get()};
  const char *outputNames[] = {outputName.get()};

  auto outputs = session.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

  float *rawOutput = outputs[0].GetTensorMutableData<float>();
  std::vector<std::vector<float>> embeddings;
  embeddings.reserve(batchSize);

  for (int i = 0; i < batchSize; ++i)
  {
    std::vector<float> emb(rawOutput + i * EMBEDDING_SIZE, rawOutput + (i + 1) * EMBEDDING_SIZE);
    MathUtils::l2Normalize(emb);
    embeddings.push_back(std::move(emb));
  }

  return embeddings;
}
