#include <MathUtils.hpp>

float MathUtils::dotProduct(const std::vector<float> &a, const std::vector<float> &b)
{
  float sum = 0.f;
  for (size_t i = 0; i < a.size(); ++i)
    sum += a[i] * b[i];
  return sum;
}

float MathUtils::l2Norm(const std::vector<float> &vec)
{
  return std::sqrt(dotProduct(vec, vec));
}

void MathUtils::l2Normalize(std::vector<float> &vec)
{
  float norm = l2Norm(vec);
  if (norm > 0.f)
    for (float &x : vec)
      x /= norm;
}

std::vector<float> MathUtils::computeMeanEmbedding(const std::vector<std::vector<float>> &embeddings)
{
  if (embeddings.empty())
    return {};

  size_t size = embeddings[0].size();
  std::vector<float> mean(size, 0.f);

  for (const auto &emb : embeddings)
    for (size_t i = 0; i < size; ++i)
      mean[i] += emb[i];

  for (float &v : mean)
    v /= embeddings.size();

  return mean;
}

std::vector<float> MathUtils::getFilteredMean(const std::vector<std::vector<float>> &embeddings, float threshold)
{
  auto mean = computeMeanEmbedding(embeddings);
  l2Normalize(mean);

  // Filtra outliers
  std::vector<std::vector<float>> filtered;
  for (const auto &e : embeddings)
  {
    if (dotProduct(e, mean) >= threshold)
    {
      filtered.push_back(e);
    }
  }

  if (filtered.empty())
    throw std::runtime_error("Todos embeddings foram descartados como outliers");

  auto finalMean = computeMeanEmbedding(filtered);
  l2Normalize(finalMean);
  return finalMean;
}