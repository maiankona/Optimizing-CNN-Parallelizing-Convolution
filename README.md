# Optimizing Convolutional Neural Networks: Parallelizing Convolution
- Files related to parallel basic convolution: serial_convolution_images.cpp (varying image size), serial_convolution_filters.cpp (varying filter size)
- Files related to parallel im2col convolution: parallel_convolution_images.cpp (varying image size), parallel_convolution_filters.cpp (varying filter size)
- To execute with static image size and filter size simply choose one of the files and change the list of images/filter sizes to a single integer
- Python file used to create visuals (original visuals from report included)
- To compile and run one of the cpp files:
  - export OMP_NUM_THREADS="number"
  - taskset --cpu-list "cores" ./"program_name"
  - no quotes just write the number/name as is
