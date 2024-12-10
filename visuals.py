import matplotlib.pyplot as plt
import numpy as np

# Data for Graph 1: Execution Time vs. Filter Size
filter_sizes = [1, 3, 5, 25, 50, 75]
cores = [1, 4, 16, 32, 64]

# Execution time for Serial and Parallel implementations for each core count
serial_execution_times = {
    1: [0.141389, 0.51364, 1.24926, 26.7942, 99.3127, 213.071],
    4: [0.144781, 0.507465, 1.22217, 26.3645, 97.5243, 209.188],
    16: [0.142129, 0.506018, 1.22557, 26.3668, 97.6832, 209.708],
    32: [0.142704, 0.508135, 1.22306, 26.6313, 97.9147, 209.614],
    64: [0.141153, 0.503057, 1.22896, 26.2545, 97.6247, 209.548]
}

parallel_execution_times = {
    1: [0.0264269, 0.177266, 0.475334, 12.1404, 45.1612, 97.4953],
    4: [0.00850974, 0.0473002, 0.122467, 3.63366, 13.1105, 29.525],
    16: [0.0302975, 0.024009, 0.0358103, 0.998181, 3.83531, 8.11946],
    32: [0.0381019, 0.016835, 0.0200853, 0.723312, 2.47739, 5.45564],
    64: [0.0414745, 0.0217307, 0.0141928, 0.652707, 2.63562, 5.08741]
}

# Plot Graph 1: Serial vs. Parallel Execution Times (Filter Size)
#plt.figure(figsize=(14, 6))
plt.figure(figsize=(8, 6))
for core_count in cores:
    plt.plot(filter_sizes, serial_execution_times[core_count], marker='o', linestyle='-', label=f"Parallel Basic Convolution ({core_count} cores)")
    plt.xlabel("Filter Size")
    plt.ylabel("Execution Time (s)")
    plt.title("Parallel Basic Convolution Execution Time vs. Filter Size")
    plt.legend(loc='best')
    plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()

plt.figure(figsize=(8, 6))
for core_count in cores:
    plt.plot(filter_sizes, parallel_execution_times[core_count], marker='x', linestyle='--', label=f"Parallel Im2Col Convolution ({core_count} cores)")
    plt.xlabel("Filter Size")
    plt.ylabel("Execution Time (s)")
    plt.title("Parallel Im2Col Convolution Execution Time vs. Filter Size")
    plt.legend(loc='best')
    plt.grid(True, linestyle="--", linewidth=0.5)

    #plt.tight_layout()
plt.show()

# Data for Graph 2: Execution Time vs. Image Size
image_sizes = [128, 512, 1024, 2048, 4096, 8192]

# Execution time for Serial and Parallel implementations at fixed filter size (3x3)
serial_execution_times_by_image = {
    1: [0.0178221, 0.125678, 2.10048, 8.39556, 33.1221, None],
    4: [0.0183116, 0.125337, 2.10619, 8.36355, 32.9934, None],
    16: [0.0136307, 0.126099, 2.19871, 8.37512, 33.0805, None],
    32: [0.0132253, 0.125285, 2.07458, 8.72185, 33.1377, None],
    64: [0.0139586, 0.12655, 2.07213, 8.37353, 32.5295, None]
}

parallel_execution_times_by_image = {
    1: [0.00715488, 0.043871, 0.714993, 3.06578, 12.3763, None],
    4: [0.00178698, 0.0136796, 0.206177, 0.76491, 3.64282, None],
    16: [0.0257152, 0.0198248, 0.0526687, 0.240517, 0.953058, None],
    32: [0.0516898, 0.0228377, 0.0585449, 0.197917, 0.717082, None],
    64: [0.0388859, 0.0175855, 0.0305529, 0.149131, 0.863424, None]
}

# Plot Graph 2: Serial vs. Parallel Execution Times (Image Size)
#plt.figure(figsize=(14, 6))
plt.figure(figsize=(8, 6))
for core_count in cores:
    plt.plot(image_sizes[:len(serial_execution_times_by_image[core_count])], serial_execution_times_by_image[core_count], marker='o', linestyle='-', label=f"Parallel Basic Convolution ({core_count} cores)")
    plt.xlabel("Image Size")
    plt.ylabel("Execution Time (s)")
    plt.title("Parallel Basic Convolution Execution Time vs. Image Size")
    plt.legend(loc='best')
    plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()

plt.figure(figsize=(8, 6))
for core_count in cores:
    plt.plot(image_sizes[:len(parallel_execution_times_by_image[core_count])], parallel_execution_times_by_image[core_count], marker='x', linestyle='--', label=f"Parallel Im2Col Convolution ({core_count} cores)")
    plt.xlabel("Image Size")
    plt.ylabel("Execution Time (s)")
    plt.title("Parallel Im2Col Convolution Execution Time vs. Image Size")
    plt.legend(loc='best')
    plt.grid(True, linestyle="--", linewidth=0.5)

    #plt.tight_layout()
plt.show()

# Graph 3: Execution Time for Fixed Image Size (1024x1024) and Fixed Filter Size (3x3) across Cores
fixed_filter_size = 3
fixed_image_size = 1024

serial_fixed_times = [0.51364, 0.487465, 0.466018, 0.438135, 0.413057]
parallel_fixed_times = [0.177266, 0.0473002, 0.024009, 0.016835, 0.0127307]

plt.figure(figsize=(8, 6))
plt.plot(cores, serial_fixed_times, marker='o', linestyle='-', label="Parallel Basic Convolution Execution Time")
plt.plot(cores, parallel_fixed_times, marker='x', linestyle='--', label="Parallel Im2Col Convolution Execution Time")
plt.xlabel("Number of Cores")
plt.ylabel("Execution Time (s)")
plt.title(f"Execution Time for Fixed Image Size ({fixed_image_size}x{fixed_image_size}) and Filter Size {fixed_filter_size}x{fixed_filter_size}")
plt.legend(loc='best')
plt.grid(True, linestyle="--", linewidth=0.5)
plt.show()

