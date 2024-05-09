
#include "histogram_eq.h"
#include <cstdlib>

int main(int argc, char **argv) {

    if (argc != 4) {
        std::cout << "usage" << argv[0] << " input_image.ppm n_iterations output_image.ppm\n";
        return 1;
    }

    wbImage_t inputImage = wbImport(argv[1]);
    int n_iterations = static_cast<int>(std::strtol(argv[2], nullptr, 10));
    wbImage_t outputImage = cp::iterative_histogram_equalization(inputImage, n_iterations);
    wbExport(argv[3], outputImage);

    return 0;
}