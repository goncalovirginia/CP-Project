#include <filesystem>
#include "gtest/gtest.h"
#include "histogram_eq.h"

using namespace cp;

#define DATASET_FOLDER "../../dataset/"

TEST(HistogramEq, Input01_4) {
	wbImage_t inputImage = wbImport(DATASET_FOLDER "borabora_1.ppm");

	auto seqArray = wbImage_getData(wbImport(DATASET_FOLDER "borabora_1_out_test.ppm"));
	auto parArray = wbImage_getData(iterative_histogram_equalization(inputImage, 10));

	for (int i = 0; i < inputImage->height * inputImage->width; ++i) {
		if (parArray[i] != seqArray[i]) {
			std:: cout << "Images are not equal.";
			return;
		}
	}

	std::cout << "Images are equal.";
}