#include <FacenetEngine.hpp>
#include <ImageProcessor.hpp>
#include <MathUtils.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <cmath>
#include <cctype>

namespace fs = std::filesystem;

struct ImageId
{
  std::string person;
  int index{};

  fs::path path(const fs::path &root) const
  {
    std::ostringstream name;
    name << person << "_" << std::setfill('0') << std::setw(4) << index << ".jpg";
    return root / person / name.str();
  }
};

struct Pair
{
  ImageId left;
  ImageId right;
  bool is_match{};
};

static bool has_header(std::string_view line)
{
  std::string lower(line);
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c)
                 { return static_cast<char>(std::tolower(c)); });
  return lower.find("name") != std::string::npos;
}

static std::vector<Pair> read_match_pairs(const fs::path &csv)
{
  std::ifstream file(csv);
  if (!file.is_open())
    throw std::runtime_error("Não foi possível abrir " + csv.string());

  std::vector<Pair> pairs;
  std::string line;
  bool skip_header = true;
  while (std::getline(file, line))
  {
    if (line.empty())
      continue;
    if (skip_header && has_header(line))
    {
      skip_header = false;
      continue;
    }
    std::stringstream ss(line);
    std::string name, idx1, idx2;
    std::getline(ss, name, ',');
    std::getline(ss, idx1, ',');
    std::getline(ss, idx2, ',');
    pairs.push_back({ImageId{name, std::stoi(idx1)}, ImageId{name, std::stoi(idx2)}, true});
  }
  return pairs;
}

static std::vector<Pair> read_mismatch_pairs(const fs::path &csv)
{
  std::ifstream file(csv);
  if (!file.is_open())
    throw std::runtime_error("Não foi possível abrir " + csv.string());

  std::vector<Pair> pairs;
  std::string line;
  bool skip_header = true;
  while (std::getline(file, line))
  {
    if (line.empty())
      continue;
    if (skip_header && has_header(line))
    {
      skip_header = false;
      continue;
    }
    std::stringstream ss(line);
    std::string name1, idx1, name2, idx2;
    std::getline(ss, name1, ',');
    std::getline(ss, idx1, ',');
    std::getline(ss, name2, ',');
    std::getline(ss, idx2, ',');
    pairs.push_back({ImageId{name1, std::stoi(idx1)}, ImageId{name2, std::stoi(idx2)}, false});
  }
  return pairs;
}

static float l2_distance(const std::vector<float> &a, const std::vector<float> &b)
{
  if (a.size() != b.size())
    throw std::runtime_error("Tamanhos diferentes de embedding");
  float sum = 0.f;
  for (size_t i = 0; i < a.size(); ++i)
  {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return std::sqrt(sum);
}

static double percentile(std::vector<double> v, double p)
{
  if (v.empty())
    return 0.0;
  std::sort(v.begin(), v.end());
  double idx = (p / 100.0) * (v.size() - 1);
  size_t lo = static_cast<size_t>(std::floor(idx));
  size_t hi = static_cast<size_t>(std::ceil(idx));
  if (lo == hi)
    return v[lo];
  double frac = idx - lo;
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

struct Options
{
  fs::path lfw_root = "../../LuscaUtils/datasets/lfw-deepfunneled";
  std::vector<fs::path> match_csvs{
      "../../LuscaUtils/datasets/matchpairsDevTrain.csv",
      "../../LuscaUtils/datasets/matchpairsDevTest.csv"};
  std::vector<fs::path> mismatch_csvs{
      "../../LuscaUtils/datasets/mismatchpairsDevTrain.csv",
      "../../LuscaUtils/datasets/mismatchpairsDevTest.csv"};
  float threshold = 1.1185f;
  int image_size = 160;
  int batch_images = 2; // número de imagens por chamada de inferência
};

static Options parse_args(int argc, char **argv)
{
  Options opt;
  for (int i = 1; i < argc; ++i)
  {
    std::string arg = argv[i];
    auto next = [&]() -> std::string
    {
      if (i + 1 >= argc)
        throw std::runtime_error("Argumento faltando após " + arg);
      return argv[++i];
    };

    if (arg == "--lfw-root")
    {
      opt.lfw_root = next();
    }
    else if (arg == "--threshold")
    {
      opt.threshold = std::stof(next());
    }
    else if (arg == "--image-size")
    {
      opt.image_size = std::stoi(next());
    }
    else if (arg == "--batch-images")
    {
      opt.batch_images = std::stoi(next());
    }
    else if (arg == "--pairs-match")
    {
      opt.match_csvs.clear();
      std::stringstream ss(next());
      std::string item;
      while (std::getline(ss, item, ','))
        opt.match_csvs.emplace_back(item);
    }
    else if (arg == "--pairs-mismatch")
    {
      opt.mismatch_csvs.clear();
      std::stringstream ss(next());
      std::string item;
      while (std::getline(ss, item, ','))
        opt.mismatch_csvs.emplace_back(item);
    }
    else
    {
      throw std::runtime_error("Argumento desconhecido: " + arg);
    }
  }
  if (opt.batch_images < 2)
    opt.batch_images = 2;
  if (opt.batch_images % 2 != 0)
    opt.batch_images += 1;
  return opt;
}

int main(int argc, char **argv)
{
  try
  {
    Options opt = parse_args(argc, argv);

    if (const char *envBatch = std::getenv("BENCH_BATCH_IMAGES"))
    {
      opt.batch_images = std::max(2, std::atoi(envBatch));
      if (opt.batch_images % 2 != 0)
        opt.batch_images += 1; // precisa ser par (pares de comparação)
    }

    const char *modelEnv = std::getenv("FACENET_MODEL_PATH");
    const std::string modelPath = modelEnv ? modelEnv : "../../models/facenet_int8.onnx";

    FacenetEngine engine(modelPath);

    std::vector<Pair> pairs;
    for (const auto &p : opt.match_csvs)
    {
      if (fs::exists(p))
      {
        auto v = read_match_pairs(p);
        pairs.insert(pairs.end(), v.begin(), v.end());
      }
    }
    for (const auto &p : opt.mismatch_csvs)
    {
      if (fs::exists(p))
      {
        auto v = read_mismatch_pairs(p);
        pairs.insert(pairs.end(), v.begin(), v.end());
      }
    }

    if (pairs.empty())
      throw std::runtime_error("Nenhum par encontrado. Verifique os CSVs.");

    size_t correct = 0;
    std::vector<double> timings_ms;
    timings_ms.reserve(pairs.size());

    const int pairs_per_batch = opt.batch_images / 2;

    for (size_t idx = 0; idx < pairs.size();)
    {
      int currentPairs = static_cast<int>(std::min<size_t>(pairs_per_batch, pairs.size() - idx));
      int imagesInBatch = currentPairs * 2;
      std::vector<float> batchTensor;
      batchTensor.reserve(static_cast<size_t>(imagesInBatch) * 3 * opt.image_size * opt.image_size);

      // carrega até batch_images (pares empilhados em imagens)
      for (int b = 0; b < currentPairs; ++b)
      {
        const auto &pair = pairs[idx + b];
        auto load_image = [&](const ImageId &imgId)
        {
          auto path = imgId.path(opt.lfw_root);
          std::ifstream fin(path, std::ios::binary);
          if (!fin.is_open())
            throw std::runtime_error("Falha ao abrir " + path.string());
          std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
          if (bytes.empty())
            throw std::runtime_error("Falha ao ler imagem " + path.string());
          cv::Mat image = ImageProcessor::decodeAndNormalizeImage(bytes, opt.image_size);
          auto chw = ImageProcessor::convertHWCtoCHW(image);
          batchTensor.insert(batchTensor.end(), chw.begin(), chw.end());
        };

        load_image(pair.left);
        load_image(pair.right);
      }

      auto t0 = std::chrono::high_resolution_clock::now();
      auto embs = engine.runInference(batchTensor, imagesInBatch);
      auto t1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      timings_ms.push_back(ms);

      for (int b = 0; b < currentPairs; ++b)
      {
        const auto &pair = pairs[idx + b];
        const auto &embA = embs[b * 2];
        const auto &embB = embs[b * 2 + 1];
        float dist = l2_distance(embA, embB);
        bool pred_match = dist <= opt.threshold;
        correct += (pred_match == pair.is_match);
      }

      idx += currentPairs;
    }

    double acc = static_cast<double>(correct) / pairs.size();
    double mean = timings_ms.empty() ? 0.0 : std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0) / timings_ms.size();

    std::cout << "Total de pares: " << pairs.size() << "\n";
    std::cout << "Acurácia: " << acc * 100.0 << "%\n";
    std::cout << "Latência runInference (ms): mean=" << mean
              << ", p50=" << percentile(timings_ms, 50)
              << ", p90=" << percentile(timings_ms, 90)
              << ", p95=" << percentile(timings_ms, 95)
              << ", p99=" << percentile(timings_ms, 99)
              << "\n";
  }
  catch (const std::exception &ex)
  {
    std::cerr << "Erro: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
