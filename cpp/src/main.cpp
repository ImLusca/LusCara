#include <iostream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cstdlib>

#include "FacenetEngine.hpp"
#include "ImageProcessor.hpp"
#include "MathUtils.hpp"

// ------------------------------------------------------------
// Utilitários de Leitura
// ------------------------------------------------------------
template <typename T>
T readValueFromStdin()
{
  T value;
  std::cin.read(reinterpret_cast<char *>(&value), sizeof(T));
  if (!std::cin)
    throw std::runtime_error("Erro ao ler valor do STDIN");
  return value;
}

std::vector<uint8_t> readBufferFromStdin(uint32_t size)
{
  std::vector<uint8_t> buffer(size);
  std::cin.read(reinterpret_cast<char *>(buffer.data()), size);
  if (!std::cin)
    throw std::runtime_error("Erro ao ler buffer do STDIN");
  return buffer;
}

// ------------------------------------------------------------
// Loop Principal
// ------------------------------------------------------------
int main()
{
  try
  {
    const char *modelEnv = std::getenv("FACENET_MODEL_PATH");
    const std::string modelPath = modelEnv ? modelEnv : "../models/facenet_int8.onnx";

    FacenetEngine engine(modelPath);

    // Threshold equivalente ao L2=1.1185 (cos ~0.3745)
    const float SIMILARITY_THRESHOLD = 0.3745f;

    const int TARGET_SIZE = 160;

    while (true)
    {
      uint32_t numImages;
      if (!std::cin.read(reinterpret_cast<char *>(&numImages), sizeof(uint32_t)))
        break;

      if (numImages == 0 || numImages > 32)
        throw std::runtime_error("Número inválido de imagens (Max: 32)");

      std::vector<float> batchTensor;
      batchTensor.reserve(numImages * 3 * TARGET_SIZE * TARGET_SIZE);

      for (uint32_t i = 0; i < numImages; ++i)
      {
        uint32_t imageSize = readValueFromStdin<uint32_t>();
        auto imageBytes = readBufferFromStdin(imageSize);

        cv::Mat image = ImageProcessor::decodeAndNormalizeImage(imageBytes, TARGET_SIZE);
        auto chw = ImageProcessor::convertHWCtoCHW(image);

        batchTensor.insert(batchTensor.end(), chw.begin(), chw.end());
      }

      auto embeddings = engine.runInference(batchTensor, static_cast<int>(numImages));

      auto finalEmbedding = MathUtils::getFilteredMean(embeddings, SIMILARITY_THRESHOLD);

      uint32_t outputSize = static_cast<uint32_t>(finalEmbedding.size());
      std::cout.write(reinterpret_cast<char *>(&outputSize), sizeof(uint32_t));
      std::cout.write(reinterpret_cast<char *>(finalEmbedding.data()), outputSize * sizeof(float));
      std::cout.flush();
    }
  }
  catch (const std::exception &ex)
  {
    std::cerr << "Erro fatal: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
