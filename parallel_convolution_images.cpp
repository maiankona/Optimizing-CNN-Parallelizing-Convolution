#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <ctime>

// Parallelized `im2col` Function using OpenMP
void parallel_im2col(const std::vector<std::vector<double>>& input, 
                     std::vector<std::vector<double>>& im2col_matrix,
                     int input_h, int input_w, int filter_h, int filter_w, 
                     int stride, int padding) {
    int output_h = (input_h - filter_h + 2 * padding) / stride + 1;
    int output_w = (input_w - filter_w + 2 * padding) / stride + 1;

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < output_h; ++y) {
        for (int x = 0; x < output_w; ++x) {
            for (int fy = 0; fy < filter_h; ++fy) {
                for (int fx = 0; fx < filter_w; ++fx) {
                    int in_y = y * stride + fy - padding;
                    int in_x = x * stride + fx;
                    im2col_matrix[y * output_w + x][fy * filter_w + fx] =
                        (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) 
                        ? input[in_y][in_x]
                        : 0.0;
                }
            }
        }
    }
}

// Parallel Convolution Function using `im2col` and OpenMP
void parallel_convolution(const std::vector<std::vector<double>>& im2col_matrix,
                          const std::vector<std::vector<double>>& filter,
                          std::vector<std::vector<double>>& output, 
                          int output_h, int output_w, int filter_h, int filter_w) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < output_h; ++y) {
        for (int x = 0; x < output_w; ++x) {
            double sum = 0.0;
            for (int fy = 0; fy < filter_h; ++fy) {
                for (int fx = 0; fx < filter_w; ++fx) {
                    sum += im2col_matrix[y * output_w + x][fy * filter_w + fx] * filter[fy][fx];
                }
            }
            output[y][x] = sum;
        }
    }
}

int main() {
    // Fixed filter dimensions
    int filter_h = 3, filter_w = 3;

    // List of image sizes
    std::vector<int> image_sizes = {128, 512, 1024, 2048, 4096, 8192};

    for (int input_size : image_sizes) {
        int input_h = input_size, input_w = input_size;
        int stride = 1, padding = 0;

        // Calculate output dimensions
        int output_h = (input_h - filter_h + 2 * padding) / stride + 1;
        int output_w = (input_w - filter_w + 2 * padding) / stride + 1;

        // Initialize input, filter, and output
        std::vector<std::vector<double>> input(input_h, std::vector<double>(input_w));
        std::vector<std::vector<double>> filter(filter_h, std::vector<double>(filter_w));
        std::vector<std::vector<double>> output(output_h, std::vector<double>(output_w, 0.0));

        // Fill input and filter with random values
        for (int i = 0; i < input_h; ++i) {
            for (int j = 0; j < input_w; ++j) {
                input[i][j] = static_cast<double>(rand()) / RAND_MAX;
            }
        }
        for (int i = 0; i < filter_h; ++i) {
            for (int j = 0; j < filter_w; ++j) {
                filter[i][j] = static_cast<double>(rand()) / RAND_MAX;
            }
        }

        // Initialize im2col matrix
        std::vector<std::vector<double>> im2col_matrix(output_h * output_w, std::vector<double>(filter_h * filter_w));

        // Calculate FLOPs
        long long flops = 2LL * output_h * output_w * filter_h * filter_w;

        // Start timing
        double start_time = omp_get_wtime();

        // Perform `im2col` and convolution
        parallel_im2col(input, im2col_matrix, input_h, input_w, filter_h, filter_w, stride, padding);
        parallel_convolution(im2col_matrix, filter, output, output_h, output_w, filter_h, filter_w);

        // Stop timing
        double elapsed_time = omp_get_wtime() - start_time;

        // Output results
        std::cout << "Image Size: " << input_size << "x" << input_size << std::endl;
        std::cout << "Parallel Convolution Execution Time: " << elapsed_time << " seconds" << std::endl;
        std::cout << "Total Floating Point Ops: " << flops << std::endl;
        std::cout << "FLOPS: " << flops / elapsed_time << " FLOPS" << std::endl;
        std::cout << "--------------------------------------" << std::endl;
    }

    return 0;
}
