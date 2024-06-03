//
// Created by herve on 13-04-2024.
//

#include "histogram_eq.h"

#include <chrono>

using namespace std::chrono;

namespace cp {
	constexpr auto HISTOGRAM_LENGTH = 256;

	int FUNCTION_TIMES[5] = {0, 0, 0, 0, 0};

	static float inline prob(const int x, const int size) {
		return (float) x / (float) size;
	}

	static unsigned char inline clamp(unsigned char x) {
		return std::min(std::max(x, static_cast<unsigned char>(0)), static_cast<unsigned char>(255));
	}

	static unsigned char inline correct_color(float cdf_val, float cdf_min) {
		return clamp(static_cast<unsigned char>(255 * (cdf_val - cdf_min) / (1 - cdf_min)));
	}

	static void histogram_equalization(const int width, const int height,
									   const float *input_image_data,
									   float *output_image_data,
									   const std::shared_ptr<unsigned char[]> &uchar_image,
									   const std::shared_ptr<unsigned char[]> &gray_image,
									   int (&histogram)[HISTOGRAM_LENGTH],
									   float (&cdf)[HISTOGRAM_LENGTH]) {

		constexpr auto channels = 3;
		const auto size = width * height;
		const auto size_channels = size * channels;

		auto start = high_resolution_clock::now();

		for (int i = 0; i < size_channels; i++)
			uchar_image[i] = (unsigned char) (255 * input_image_data[i]);

		auto end = high_resolution_clock::now();
		FUNCTION_TIMES[0] += duration_cast<milliseconds>(end - start).count();

		start = high_resolution_clock::now();

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				auto idx = i * width + j;
				auto r = uchar_image[3 * idx];
				auto g = uchar_image[3 * idx + 1];
				auto b = uchar_image[3 * idx + 2];
				gray_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
			}

		end = high_resolution_clock::now();
		FUNCTION_TIMES[1] += duration_cast<milliseconds>(end - start).count();

		start = high_resolution_clock::now();

		std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);
		for (int i = 0; i < size; i++)
			histogram[gray_image[i]]++;

		end = high_resolution_clock::now();
		FUNCTION_TIMES[2] += duration_cast<milliseconds>(end - start).count();

		start = high_resolution_clock::now();

		cdf[0] = prob(histogram[0], size);
		for (int i = 1; i < HISTOGRAM_LENGTH; i++)
			cdf[i] = cdf[i - 1] + prob(histogram[i], size);

		auto cdf_min = cdf[0];
		for (int i = 1; i < HISTOGRAM_LENGTH; i++)
			cdf_min = std::min(cdf_min, cdf[i]);

		end = high_resolution_clock::now();
		FUNCTION_TIMES[3] += duration_cast<milliseconds>(end - start).count();

		start = high_resolution_clock::now();

		for (int i = 0; i < size_channels; i++)
			uchar_image[i] = correct_color(cdf[uchar_image[i]], cdf_min);

		for (int i = 0; i < size_channels; i++)
			output_image_data[i] = static_cast<float>(uchar_image[i]) / 255.0f;

		end = high_resolution_clock::now();
		FUNCTION_TIMES[4] += duration_cast<milliseconds>(end - start).count();
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

		for (int i = 0; i < 5; ++i) {
			std::cout << FUNCTION_TIMES[i] << "\n";
		}

		return output_image;
	}
}