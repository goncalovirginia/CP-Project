#include <cuda_runtime.h>
#include "histogram_eq.h"
#include <iostream>
#include <memory>

namespace cp {

	constexpr int HISTOGRAM_LENGTH = 256;
	constexpr int THREADS_PER_BLOCK = 2048;

	static float inline prob(const int x, const int size) {
		return static_cast<float>(x) / static_cast<float>(size);
	}

	__device__ static unsigned char inline gpu_clamp(unsigned char x) {
		return min(max(x, static_cast<unsigned char>(0)), static_cast<unsigned char>(255));
	}

	__device__ static unsigned char inline gpu_correct_color(float cdf_val, float cdf_min) {
		return gpu_clamp(static_cast<unsigned char>(255 * (cdf_val - cdf_min) / (1 - cdf_min)));
	}

	__global__ void correct_color(unsigned char *uchar_image, float *cdf, float cdf_min, int size_channels) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < size_channels) {
			uchar_image[idx] = gpu_correct_color(cdf[uchar_image[idx]], cdf_min);
		}
	}

	__global__ void percentageTo255(const float *input, unsigned char *output, int size) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < size) {
			output[idx] = static_cast<unsigned char>(255 * input[idx]);
		}
	}

	__global__ void grayScale(const unsigned char *rgb_image, unsigned char *gray_image, int width, int height) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < width && y < height) {
			int idx = y * width + x;
			int rgb_idx = 3 * idx;
			unsigned char r = rgb_image[rgb_idx];
			unsigned char g = rgb_image[rgb_idx + 1];
			unsigned char b = rgb_image[rgb_idx + 2];
			gray_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
		}
	}

	__global__ void computeLocalHistogram(const unsigned char *gray_image, int *local_histograms, int width, int height) {
		extern __shared__ int local_hist[];
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		int idx = threadIdx.x;

		if (idx < HISTOGRAM_LENGTH) {
			local_hist[idx] = 0;
		}
		__syncthreads();

		while (tid < width * height) {
			atomicAdd(&local_hist[gray_image[tid]], 1);
			tid += blockDim.x * gridDim.x;
		}
		__syncthreads();

		if (idx < HISTOGRAM_LENGTH) {
			atomicAdd(&local_histograms[blockIdx.x * HISTOGRAM_LENGTH + idx], local_hist[idx]);
		}
	}

	__global__ void mergeLocalHistograms(int *local_histograms, int *global_histogram, int num_blocks) {
		int idx = threadIdx.x;
		int sum = 0;

		for (int i = 0; i < num_blocks; i++) {
			sum += local_histograms[i * HISTOGRAM_LENGTH + idx];
		}
		global_histogram[idx] = sum;
	}

	__global__ void computeOutputImage(const unsigned char *uchar_image, float *output_image_data, int size_channels) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < size_channels) {
			output_image_data[idx] = static_cast<float>(uchar_image[idx]) / 255.0f;
		}
	}
/*
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
*/
	static void histogram_equalization(const int width, const int height, const int size, const int size_channels,
									   const float *input_image_data,
									   float *output_image_data,
									   const std::shared_ptr<unsigned char[]> &uchar_image,
									   const std::shared_ptr<unsigned char[]> &gray_image,
									   int (&histogram)[HISTOGRAM_LENGTH],
									   float (&cdf)[HISTOGRAM_LENGTH],
									   float *d_input, unsigned char *d_uchar_image, unsigned char *d_rgb_image,
									   unsigned char *d_gray_image, int *d_local_histograms, int *d_global_histogram,
									   float *d_cdf, float *d_output_image_data) {

		percentageTo255<<<(size_channels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_input,d_uchar_image,size_channels);
		cudaDeviceSynchronize();

		// Compute grid and block dimensions for grayscale conversion
		int blockWidth = static_cast<int>(sqrt(THREADS_PER_BLOCK));
		while (THREADS_PER_BLOCK % blockWidth != 0) {
			blockWidth--;
		}
		int blockHeight = THREADS_PER_BLOCK / blockWidth;
		dim3 blockDim(blockWidth, blockHeight);
		dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

		// Convert RGB to grayscale
		grayScale<<<gridDim, blockDim>>>(d_uchar_image, d_gray_image, width, height);
		cudaDeviceSynchronize();

		// Calculate number of blocks for the histogram computation
		int numBlocks = (width * height + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

		// Compute histogram
		std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);
		cudaMemset(d_local_histograms, 0, numBlocks * HISTOGRAM_LENGTH * sizeof(int));
		cudaMemset(d_global_histogram, 0, HISTOGRAM_LENGTH * sizeof(int));
		computeLocalHistogram<<<numBlocks, THREADS_PER_BLOCK, HISTOGRAM_LENGTH * sizeof(int)>>>(d_gray_image, d_local_histograms, width, height);
		cudaDeviceSynchronize();
		mergeLocalHistograms<<<1, HISTOGRAM_LENGTH>>>(d_local_histograms, d_global_histogram, numBlocks);
		cudaDeviceSynchronize();
		cudaMemcpy(histogram, d_global_histogram, HISTOGRAM_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);

		// Compute CDF
		cdf[0] = prob(histogram[0], size);
		for (int i = 1; i < HISTOGRAM_LENGTH; i++)
			cdf[i] = cdf[i - 1] + prob(histogram[i], size);

		auto cdf_min = cdf[0];

		// Correct colors
		cudaMemcpy(d_cdf, cdf, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
		correct_color<<<(size_channels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_uchar_image,d_cdf, cdf_min, size_channels);
		cudaDeviceSynchronize();

		// Generate equalized image
		computeOutputImage<<<(size_channels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_uchar_image, d_output_image_data, size_channels);
		cudaDeviceSynchronize();
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

		float *gpu_input;
		unsigned char *gpu_uchar_image, *gpu_rgb_image, *gpu_gray_image;
		int *gpu_local_histograms, *gpu_global_histogram;
		float *gpu_cdf, *gpu_output_image_data;

		int numBlocks = (width * height + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

		cudaMalloc(&gpu_input, size_channels * sizeof(float));
		cudaMalloc(&gpu_uchar_image, size_channels * sizeof(unsigned char));
		cudaMalloc(&gpu_rgb_image, size_channels * sizeof(unsigned char));
		cudaMalloc(&gpu_gray_image, size * sizeof(unsigned char));
		cudaMalloc(&gpu_local_histograms, numBlocks * HISTOGRAM_LENGTH * sizeof(int));
		cudaMalloc(&gpu_global_histogram, HISTOGRAM_LENGTH * sizeof(int));
		cudaMalloc(&gpu_cdf, HISTOGRAM_LENGTH * sizeof(float));
		cudaMalloc(&gpu_output_image_data, size_channels * sizeof(float));

		cudaMemcpy(gpu_input, input_image_data, size_channels * sizeof(float), cudaMemcpyHostToDevice);

		for (int i = 0; i < iterations; i++) {
			histogram_equalization(width, height, size, size_channels,
								   input_image_data, output_image_data,
								   uchar_image, gray_image, histogram, cdf,
								   gpu_input, gpu_uchar_image, gpu_rgb_image,
								   gpu_gray_image, gpu_local_histograms,
								   gpu_global_histogram,gpu_cdf, gpu_output_image_data);

			input_image_data = output_image_data;
		}

		cudaMemcpy(output_image_data, gpu_output_image_data, size_channels * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(gpu_input);
		cudaFree(gpu_uchar_image);
		cudaFree(gpu_rgb_image);
		cudaFree(gpu_gray_image);
		cudaFree(gpu_local_histograms);
		cudaFree(gpu_global_histogram);
		cudaFree(gpu_cdf);
		cudaFree(gpu_output_image_data);

		return output_image;
	}
}