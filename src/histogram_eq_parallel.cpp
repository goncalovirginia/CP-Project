#include "histogram_eq.h"
#include <chrono>
#include <omp.h>

using namespace std::chrono;

namespace cp {

	constexpr int HISTOGRAM_LENGTH = 256;

	int FUNCTION_TIMES[5] = {0, 0, 0, 0, 0};

	static float inline prob(const int x, const int size) {
		return (float) x / (float) size;
	}

	static float inline clamp(float x) {
		return std::min(std::max(x, static_cast<float>(0.0)), static_cast<float>(1.0));
	}

	static float inline correct_color(float cdf_val, float cdf_min) {
		return clamp((cdf_val - cdf_min) / (1 - cdf_min));
	}

	static void percentageTo255(const float *input_image_data, const std::shared_ptr<unsigned char[]> &uchar_image,
								int size_channels) {
		#pragma omp parallel for
		for (int i = 0; i < size_channels; ++i)
			uchar_image[i] = (unsigned char) (255 * input_image_data[i]);
	}

	static void grayScale(int height, int width, const std::shared_ptr<unsigned char[]> &uchar_image,
						  const std::shared_ptr<unsigned char[]> &gray_image) {
		#pragma omp parallel for
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				int idx = i * width + j;
				unsigned char r = uchar_image[3 * idx];
				unsigned char g = uchar_image[3 * idx + 1];
				unsigned char b = uchar_image[3 * idx + 2];
				gray_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
			}
		}
	}

	static void computeHistogram(const int size, int (&histogram)[HISTOGRAM_LENGTH], const std::shared_ptr<unsigned char[]> &gray_image) {
		std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);

		#pragma omp parallel for reduction(+:histogram)
		for (int i = 0; i < size; ++i) {
			++histogram[gray_image[i]];
		}
	}

	static float computeCDF(float (&cdf)[HISTOGRAM_LENGTH], int (&histogram)[HISTOGRAM_LENGTH], const int size) {
		#pragma omp parallel for schedule(static, 64)
		for (int i = 0; i < HISTOGRAM_LENGTH; ++i) {
			cdf[i] = prob(histogram[i], size);
		}

		float cdf_min = cdf[0];

		for (int i = 1; i < HISTOGRAM_LENGTH; ++i) {
			cdf[i] += cdf[i-1];
			cdf_min = std::min(cdf_min, cdf[i]);
		}

		return cdf_min;
	}

	static void computeOutputImage(float *output_image_data, float (&cdf)[HISTOGRAM_LENGTH],
								   const std::shared_ptr<unsigned char[]> &uchar_image, float cdf_min,
								   int size_channels) {
		#pragma omp parallel for
		for (int i = 0; i < size_channels; ++i) {
			output_image_data[i] = correct_color(cdf[uchar_image[i]], cdf_min);
		}
	}

	static void histogram_equalization(const int width, const int height, const int size, const int size_channels,
									   const float *input_image_data,
									   float *output_image_data,
									   const std::shared_ptr<unsigned char[]> &uchar_image,
									   const std::shared_ptr<unsigned char[]> &gray_image,
									   int (&histogram)[HISTOGRAM_LENGTH],
									   float (&cdf)[HISTOGRAM_LENGTH]) {

		auto start = high_resolution_clock::now();

		percentageTo255(input_image_data, uchar_image, size_channels);

		auto end = high_resolution_clock::now();
		FUNCTION_TIMES[0] += duration_cast<milliseconds>(end - start).count();

		start = high_resolution_clock::now();

		grayScale(height, width, uchar_image, gray_image);

		end = high_resolution_clock::now();
		FUNCTION_TIMES[1] += duration_cast<milliseconds>(end - start).count();

		start = high_resolution_clock::now();

		computeHistogram(size, histogram, gray_image);

		end = high_resolution_clock::now();
		FUNCTION_TIMES[2] += duration_cast<milliseconds>(end - start).count();

		start = high_resolution_clock::now();

		float cdf_min = computeCDF(cdf, histogram, size);

		end = high_resolution_clock::now();
		FUNCTION_TIMES[3] += duration_cast<milliseconds>(end - start).count();

		start = high_resolution_clock::now();

		computeOutputImage(output_image_data, cdf, uchar_image, cdf_min, size_channels);

		end = high_resolution_clock::now();
		FUNCTION_TIMES[4] += duration_cast<milliseconds>(end - start).count();
	}

	wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations) {
		const int width = wbImage_getWidth(input_image);
		const int height = wbImage_getHeight(input_image);
		const int channels = 3;
		const int size = width * height;
		const int size_channels = size * channels;

		wbImage_t output_image = wbImage_new(width, height, channels);
		float *input_image_data = wbImage_getData(input_image);
		float *output_image_data = wbImage_getData(output_image);

		std::shared_ptr<unsigned char[]> uchar_image(new unsigned char[size_channels]);
		std::shared_ptr<unsigned char[]> gray_image(new unsigned char[size]);

		int histogram[HISTOGRAM_LENGTH];
		float cdf[HISTOGRAM_LENGTH];

		//omp_set_num_threads(8);

		for (int i = 0; i < iterations; ++i) {
			histogram_equalization(width, height, size, size_channels,
								   input_image_data, output_image_data,
								   uchar_image, gray_image,histogram, cdf);

			input_image_data = output_image_data;
		}

		for (int i = 0; i < 5; ++i) {
			std::cout << FUNCTION_TIMES[i] << "\n";
		}

		return output_image;
	}
}