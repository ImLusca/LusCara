#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>

class MathUtils
{
public:
  static float dotProduct(const std::vector<float> &a, const std::vector<float> &b);
  static float l2Norm(const std::vector<float> &vec);
  static void l2Normalize(std::vector<float> &vec);
  static std::vector<float> computeMeanEmbedding(const std::vector<std::vector<float>> &embeddings);
  static std::vector<float> getFilteredMean(const std::vector<std::vector<float>> &embeddings, float threshold);
};