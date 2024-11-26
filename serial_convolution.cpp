#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <omp.h>

// Define volume_t and conv_layer_t structures
typedef struct {
    double *weights;
    int width;
    int height;
    int depth;
} volume_t;

typedef struct {
    volume_t **filters;
    double *biases;
    int filter_width;
    int stride;
    int pad;
    int output_depth;
    int output_width;
    int output_height;
} conv_layer_t;

// Optimized Convolution Forward Function with OpenMP and Batch Processing
void conv_forward(conv_layer_t *l, volume_t **inputs, volume_t **outputs, int batch_size) {
    int stride = l->stride;
    int pad = l->pad;
    int filter_width = l->filter_width;
    int output_depth = l->output_depth;
    int output_height = l->output_height;
    int output_width = l->output_width;

    for (int i = 0; i < batch_size; i++) {
        #pragma omp parallel for
        for (int f = 0; f < output_depth; f++) {
            for (int out_y = 0; out_y < output_height; out_y++) {
                for (int out_x = 0; out_x < output_width; out_x++) {
                    for (int fd = 0; fd < inputs[i]->depth; fd++) {
                        double sum = 0.0;
                        int y_start = out_y * stride - pad;
                        int x_start = out_x * stride - pad;
                        for (int fy = 0; fy < filter_width; fy++) {
                            int in_y = y_start + fy;
                            if (in_y >= 0 && in_y < inputs[i]->height) {
                                for (int fx = 0; fx < filter_width; fx++) {
                                    int in_x = x_start + fx;
                                    if (in_x >= 0 && in_x < inputs[i]->width) {
                                        double filter_value = l->filters[f]->weights[((fy * filter_width) + fx) * inputs[i]->depth + fd];
                                        double input_value = inputs[i]->weights[((in_y * inputs[i]->width) + in_x) * inputs[i]->depth + fd];
                                        sum += filter_value * input_value;
                                    }
                                }
                            }
                        }
                        sum += l->biases[f];
                        outputs[i]->weights[((out_y * output_width) + out_x) * output_depth + f] = sum;
                    }
                }
            }
        }
    }
}

int main() {
    // Increased batch size and decreased input dimensions
    int batch_size = 1;  // Increased batch size
    int input_h = 4096;    // Decreased input height
    int input_w = 4096;    // Decreased input width
    int input_depth = 1;  // Example depth for RGB images

    // Allocate and initialize input data, weights, etc.
    volume_t **test_data = (volume_t **)malloc(batch_size * sizeof(volume_t *));
    conv_layer_t *conv1 = (conv_layer_t *)malloc(sizeof(conv_layer_t));
    conv1->filters = (volume_t **)malloc(6 * sizeof(volume_t *));
    conv1->biases = (double *)malloc(6 * sizeof(double));
    conv1->filter_width = 5;
    conv1->stride = 1;
    conv1->pad = 2;  // Adjusted padding to maintain output dimensions
    conv1->output_depth = 6;
    conv1->output_width = (input_w - conv1->filter_width + 2 * conv1->pad) / conv1->stride + 1;
    conv1->output_height = (input_h - conv1->filter_width + 2 * conv1->pad) / conv1->stride + 1;

    volume_t **conv1_outputs = (volume_t **)malloc(batch_size * sizeof(volume_t *));

    // Initialize input data
    for (int i = 0; i < batch_size; ++i) {
        test_data[i] = (volume_t *)malloc(sizeof(volume_t));
        test_data[i]->weights = (double *)malloc(input_h * input_w * input_depth * sizeof(double));
        test_data[i]->width = input_w;
        test_data[i]->height = input_h;
        test_data[i]->depth = input_depth;
        for (int j = 0; j < input_h * input_w * input_depth; ++j) {
            test_data[i]->weights[j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }

    // Initialize filters and biases
    for (int i = 0; i < conv1->output_depth; ++i) {
        conv1->filters[i] = (volume_t *)malloc(sizeof(volume_t));
        conv1->filters[i]->weights = (double *)malloc(input_depth * conv1->filter_width * conv1->filter_width * sizeof(double));
        conv1->filters[i]->width = conv1->filter_width;
        conv1->filters[i]->height = conv1->filter_width;
        conv1->filters[i]->depth = input_depth;
        for (int j = 0; j < input_depth * conv1->filter_width * conv1->filter_width; ++j) {
            conv1->filters[i]->weights[j] = static_cast<double>(rand()) / RAND_MAX;
        }
        conv1->biases[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // Initialize output volumes
    for (int i = 0; i < batch_size; ++i) {
        conv1_outputs[i] = (volume_t *)malloc(sizeof(volume_t));
        conv1_outputs[i]->weights = (double *)malloc(conv1->output_height * conv1->output_width * conv1->output_depth * sizeof(double));
        conv1_outputs[i]->width = conv1->output_width;
        conv1_outputs[i]->height = conv1->output_height;
        conv1_outputs[i]->depth = conv1->output_depth;
    }

    struct timespec start, stop;
    double time;

    // Start timing for execution time measurement
    if (clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime"); }

    // Convolution with OpenMP
    conv_forward(conv1, test_data, conv1_outputs, batch_size);

    // Stop timing and calculate execution time
    if (clock_gettime(CLOCK_REALTIME, &stop) == -1) { perror("clock gettime"); }
    time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / 1e9;

    std::cout << "Serial Convolution Execution Time: " << time << " sec" << std::endl;

    // Release memory
    for (int i = 0; i < batch_size; ++i) {
        free(test_data[i]->weights);
        free(test_data[i]);
        free(conv1_outputs[i]->weights);
        free(conv1_outputs[i]);
    }
    for (int i = 0; i < conv1->output_depth; ++i) {
        free(conv1->filters[i]->weights);
        free(conv1->filters[i]);
    }
    free(conv1->filters);
    free(conv1->biases);
    free(conv1);
    free(test_data);
    free(conv1_outputs);

    return 0;
}
