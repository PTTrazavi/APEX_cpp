/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#include <iostream>

#include "tiny_dnn/tiny_dnn.h"

// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
  Activation a(1);
  return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

void convert_image(const std::string &imagefilename,
                   double minv,
                   double maxv,
                   int w,
                   int h,
                   tiny_dnn::vec_t &data) {
  tiny_dnn::image<> img(imagefilename, tiny_dnn::image_type::grayscale);
  tiny_dnn::image<> resized = resize_image(img, w, h);

  // mnist dataset is "white on black", so negate required
  //std::transform(
  //  resized.begin(), resized.end(), std::back_inserter(data),
  //  [=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
  data.resize(resized.width() * resized.height() * resized.depth());
  for (size_t c = 0; c < resized.depth(); ++c) {
    for (size_t y = 0; y < resized.height(); ++y) {
      for (size_t x = 0; x < resized.width(); ++x) {
        data[c * resized.width() * resized.height() + y * resized.width() + x] =
          (maxv - minv) * (resized[y * resized.width() + x + c]) / 255.0 + minv;
      }
    }
  }
}

void recognize(const std::string &dictionary, const std::string &src_filename) {
  tiny_dnn::network<tiny_dnn::sequential> nn;

  nn.load(dictionary);

  // convert imagefile to vec_t
  tiny_dnn::vec_t data;
  convert_image(src_filename, -1.0, 1.0, 32, 32, data);

  // recognize
  auto res = nn.predict(data);
  std::vector<std::pair<double, int>> scores;

  // sort & print top-3
  for (int i = 0; i < 3; i++)
    std::cout << i << "," << res[i] << std::endl;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "please specify image file" << std::endl;
    return 0;
  }
  recognize("LeNet-model", argv[1]);
}
