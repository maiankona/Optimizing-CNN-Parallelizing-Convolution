#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>

// Function to perform `im2col` transformation
void im2col(const std::vector<std::vector<double>>& input, 
            std::vector<std::vector<double>>& im2col_matrix,
            int input_h, int input_w, int filter_h, int filter_w, 
            int stride, int padding) {
    int output_h = (input_h - filter_h + 2 * padding) / stride + 1;
    int output_w = (input_w - filter_w + 2 * padding) / stride + 1;

    int col_idx = 0;
    for (int y = 0; y < output_h; ++y) {
        for (int x = 0; x < output_w; ++x) {
            for (int fy = 0; fy < filter_h; ++fy) {
                for (int fx = 0; fx < filter_w; ++fx) {
                    int in_y = y * stride + fy - padding;
                    int in_x = x * stride + fx - padding;
                    im2col_matrix[col_idx][fy * filter_w + fx] =
                        (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) 
                        ? input[in_y][in_x]
                        : 0.0;
                }
            }
            ++col_idx;
        }
    }
}

// Serial Convolution Function using `im2col`
void serial_convolution(const std::vector<std::vector<double>>& im2col_matrix,
                        const std::vector<std::vector<double>>& filter,
                        std::vector<std::vector<double>>& output, 
                        int output_h, int output_w, int filter_h, int filter_w) {
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
    // Image and filter dimensions
    int input_h = 4096, input_w = 4096;
    int filter_h = 3, filter_w = 3;
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

    // Start timing
    clock_t start = clock();

    // Perform `im2col` and convolution
    im2col(input, im2col_matrix, input_h, input_w, filter_h, filter_w, stride, padding);
    serial_convolution(im2col_matrix, filter, output, output_h, output_w, filter_h, filter_w);

    // Stop timing
    clock_t end = clock();
    double elapsed_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    // Output execution time
    std::cout << "Serial Convolution Execution Time: " << elapsed_time << " seconds" << std::endl;

    return 0;
}
