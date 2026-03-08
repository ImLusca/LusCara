#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <thread>
#include <cstdlib>

class FacenetEngine
{
public:
  FacenetEngine(const std::string &modelPath);
  std::vector<std::vector<float>> runInference(const std::vector<float> &batchTensor, int batchSize);

private:
  Ort::Session initSession(const std::string &modelPath);

  Ort::Env env;
  Ort::Session session;
  Ort::AllocatorWithDefaultOptions allocator;

  static constexpr int EMBEDDING_SIZE = 512;
  static constexpr int IMAGE_SIZE = 160;
  static constexpr int CHANNELS = 3;
};
