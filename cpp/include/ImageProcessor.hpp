#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class ImageProcessor
{
public:
  static cv::Mat decodeAndNormalizeImage(const std::vector<uint8_t> &imageBytes, int imageSize);
  static std::vector<float> convertHWCtoCHW(const cv::Mat &image);

private:
  static cv::Mat makeSquare(const cv::Mat &face);
};
