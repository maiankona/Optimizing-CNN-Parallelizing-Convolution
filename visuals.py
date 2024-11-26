import matplotlib.pyplot as plt

# Data for AMD EPYC server
amd_epyc_cores = [1, 2, 4, 8, 16, 32, 64, 128]
amd_epyc_serial_times = [20.4469, 10.9835, 8.54488, 6.38514, 3.71425, 3.69849, 3.69846, 3.64171]
amd_epyc_parallel_times = [3.68936, 1.77482, 0.866098, 0.515158, 0.404307, 0.132018, 0.160907, 0.207677]

# Calculate speedup for parallel execution
amd_epyc_parallel_speedup = [amd_epyc_parallel_times[0] / t for t in amd_epyc_parallel_times]
amd_epyc_serial_speedup = [amd_epyc_serial_times[0] / t for t in amd_epyc_serial_times]

# Function to add non-overlapping data labels
def add_labels(x, y, offset=0.02):
    for i in range(len(x)):
        plt.text(
            x[i], y[i] + offset, f'{y[i]:.2f}', ha='center', fontsize=10, color="black"
        )

# Visualization: Execution Times
plt.figure(figsize=(12, 7))
plt.plot(amd_epyc_cores, amd_epyc_serial_times, label="AMD Parallel Baseline Times", marker='o', linestyle='-', linewidth=2)
plt.plot(amd_epyc_cores, amd_epyc_parallel_times, label="AMD Parallel Im2Col Times", marker='s', linestyle='--', linewidth=2)

# Add labels to points with slight offsets
add_labels(amd_epyc_cores, amd_epyc_serial_times, offset=0.5)
add_labels(amd_epyc_cores, amd_epyc_parallel_times, offset=0.05)

# Add labels and improve visualization
plt.xlabel("Number of Threads", fontsize=14, fontweight='bold')
plt.ylabel("Execution Time (seconds)", fontsize=14, fontweight='bold')
plt.title("Execution Time vs. Number of Threads (AMD EPYC Server)", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("amd_epyc_execution_times_normal.png")
plt.show()

# Visualization: Speedup
plt.figure(figsize=(12, 7))
plt.plot(amd_epyc_cores, amd_epyc_serial_speedup, label="AMD EPYC Parallel Baseline Speedup", marker='o', linestyle='-', linewidth=2)
plt.plot(amd_epyc_cores, amd_epyc_parallel_speedup, label="AMD EPYC Parallel Im2Col Speedup", marker='s', linestyle='--', linewidth=2)
plt.axhline(y=1, color='r', linestyle='--', label="No Speedup")

# Add labels to points with slight offsets
add_labels(amd_epyc_cores, amd_epyc_serial_speedup, offset=0.1)
add_labels(amd_epyc_cores, amd_epyc_parallel_speedup, offset=0.1)

# Add labels and improve visualization
plt.xlabel("Number of Threads", fontsize=14, fontweight='bold')
plt.ylabel("Speedup", fontsize=14, fontweight='bold')
plt.title("Speedup vs. Number of Threads (AMD EPYC Server)", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("amd_epyc_speedup_normal.png")
plt.show()
