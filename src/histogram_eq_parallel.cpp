//
// Created by herve on 13-04-2024.
//

#include "histogram_eq_parallel.h"

namespace cp {

	constexpr auto HISTOGRAM_LENGTH = 256;

	static float inline prob(const int x, const int size) {
		return (float) x / (float) size;
	}

	static unsigned char inline clamp(unsigned char x) {
		return std::min(std::max(x, static_cast<unsigned char>(0)), static_cast<unsigned char>(255));
	}

	static unsigned char inline correct_color(float cdf_val, float cdf_min) {
		return clamp(static_cast<unsigned char>(255 * (cdf_val - cdf_min) / (1 - cdf_min)));
	}

	static void percentageTo255(const float *input_image_data, const std::shared_ptr<unsigned char[]> &uchar_image,
								int size_channels) {
		#pragma omp parallel for
		for (int i = 0; i < size_channels; i++)
			uchar_image[i] = (unsigned char) (255 * input_image_data[i]);
	}

	static void grayScale(int height, int width, const std::shared_ptr<unsigned char[]> &uchar_image,
						  const std::shared_ptr<unsigned char[]> &gray_image) {
		#pragma omp parallel for
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				auto idx = i * width + j;
				auto r = uchar_image[3 * idx];
				auto g = uchar_image[3 * idx + 1];
				auto b = uchar_image[3 * idx + 2];
				gray_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
			}
		}
	}

	static void computeHistogram(const int size, int (&histogram)[HISTOGRAM_LENGTH],
								 const std::shared_ptr<unsigned char[]> &gray_image) {
		#pragma omp parallel for
		for (int i = 0; i < size; i++)
			histogram[gray_image[i]]++;
	}

	static float computeCDF(float (&cdf)[HISTOGRAM_LENGTH], int (&histogram)[HISTOGRAM_LENGTH], const int size) {
		cdf[0] = prob(histogram[0], size);
		float cdf_min = cdf[0];
		for (int i = 1; i < HISTOGRAM_LENGTH; i++) {
			cdf[i] = cdf[i - 1] + prob(histogram[i], size);
			cdf_min = std::min(cdf_min, cdf[i]);
		}
		return cdf_min;
	}

	static void computeOutputImage(float *output_image_data, float (&cdf)[HISTOGRAM_LENGTH],
								   const std::shared_ptr<unsigned char[]> &uchar_image, float cdf_min,
								   int size_channels) {
		#pragma omp parallel for
		for (int i = 0; i < size_channels; i++) {
			output_image_data[i] = static_cast<float>(correct_color(cdf[uchar_image[i]], cdf_min)) / 255.0f;
		}
	}

	static void histogram_equalization(const int width, const int height,
									   const float *input_image_data,
									   float *output_image_data,
									   const std::shared_ptr<unsigned char[]> &uchar_image,
									   const std::shared_ptr<unsigned char[]> &gray_image,
									   int (&histogram)[HISTOGRAM_LENGTH],
									   float (&cdf)[HISTOGRAM_LENGTH]) {

		constexpr auto channels = 3;
		const int size = width * height;
		const int size_channels = size * channels;

		percentageTo255(input_image_data, uchar_image, size_channels);

		grayScale(height, width, uchar_image, gray_image);

		std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);

		computeHistogram(size, histogram, gray_image);

		float cdf_min = computeCDF(cdf, histogram, size);

		computeOutputImage(output_image_data, cdf, uchar_image, cdf_min, size_channels);
	}

	wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations) {
		const auto width = wbImage_getWidth(input_image);
		const auto height = wbImage_getHeight(input_image);
		constexpr auto channels = 3;
		const auto size = width * height;
		const auto size_channels = size * channels;

		wbImage_t output_image = wbImage_new(width, height, channels);
		float *input_image_data = wbImage_getData(input_image);
		float *output_image_data = wbImage_getData(output_image);

		std::shared_ptr<unsigned char[]> uchar_image(new unsigned char[size_channels]);
		std::shared_ptr<unsigned char[]> gray_image(new unsigned char[size]);

		int histogram[HISTOGRAM_LENGTH];
		float cdf[HISTOGRAM_LENGTH];

		for (int i = 0; i < iterations; i++) {
			histogram_equalization(width, height,
								   input_image_data, output_image_data,
								   uchar_image, gray_image,
								   histogram, cdf);

			input_image_data = output_image_data;
		}

		return output_image;
	}
}