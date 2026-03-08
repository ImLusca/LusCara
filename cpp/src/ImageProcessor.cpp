#include <ImageProcessor.hpp>

cv::Mat ImageProcessor::decodeAndNormalizeImage(const std::vector<uint8_t> &imageBytes, int imageSize)
{
  cv::Mat image = cv::imdecode(imageBytes, cv::IMREAD_COLOR);
  if (image.empty())
    throw std::runtime_error("Falha ao decodificar imagem");

  cv::resize(image, image, cv::Size(imageSize, imageSize));
  image.convertTo(image, CV_32F, 1.0 / 255.0);

  image = (image - 0.5f) / 0.5f;
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  return image;
}

std::vector<float> ImageProcessor::convertHWCtoCHW(const cv::Mat &image)
{
  const int channels = image.channels();
  const int rows = image.rows;
  const int cols = image.cols;

  std::vector<float> tensor(static_cast<size_t>(channels) * rows * cols);
  const size_t planeSize = static_cast<size_t>(rows) * cols;

  // Usa ponteiros diretos para reduzir overhead de at<>
  for (int c = 0; c < channels; ++c)
  {
    float *dst = tensor.data() + c * planeSize;
    for (int y = 0; y < rows; ++y)
    {
      const cv::Vec3f *row = image.ptr<cv::Vec3f>(y);
      for (int x = 0; x < cols; ++x)
      {
        dst[y * cols + x] = row[x][c];
      }
    }
  }

  return tensor;
}

cv::Mat ImageProcessor::makeSquare(const cv::Mat &face)
{
  int size = std::max(face.rows, face.cols);
  cv::Mat square = cv::Mat::zeros(size, size, face.type());

  int y_offset = (size - face.rows) / 2;
  int x_offset = (size - face.cols) / 2;

  face.copyTo(square(cv::Rect(x_offset, y_offset, face.cols, face.rows)));
  return square;
}
