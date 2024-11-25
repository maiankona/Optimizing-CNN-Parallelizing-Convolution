import matplotlib.pyplot as plt

# Data for CARC server (64 cores)
carc_cores = [1, 2, 4, 16, 64]
carc_parallel_times = [3.22922, 1.49847, 0.759854, 0.407283, 0.315415]
carc_serial_time = 1.47952

# Data for AMD EPYC 7763 64-Core Processor (256 cores)
epyc_cores = [1, 2, 4, 16, 64, 128, 256]
epyc_parallel_times = [3.68936, 1.77482, 0.866098, 0.404307, 0.160907, 0.207677, 0.196967]
epyc_serial_time = 4.00527

# Calculate speedup
carc_speedup = [carc_serial_time / t for t in carc_parallel_times]
epyc_speedup = [epyc_serial_time / t for t in epyc_parallel_times]

# Function to add non-overlapping data labels
def add_labels(x, y, offset=0.02):
    for i in range(len(x)):
        plt.text(
            x[i], y[i] + offset, f'{y[i]:.2f}', ha='center', fontsize=10, color="black"
        )

# Visualization: Execution Times
plt.figure(figsize=(12, 7))
plt.plot(carc_cores, carc_parallel_times, label="CARC Parallel Times", marker='o', linestyle='-', linewidth=2)
plt.axhline(y=carc_serial_time, color='r', linestyle='--', label="CARC Serial Time")
plt.plot(epyc_cores, epyc_parallel_times, label="AMD EPYC Parallel Times", marker='s', linestyle='--', linewidth=2)
plt.axhline(y=epyc_serial_time, color='g', linestyle='--', label="AMD EPYC Serial Time")

# Add shaded region for optimal scaling range
plt.axvspan(4, 64, color='purple', alpha=0.2, label="Optimal Scaling Range (4-64 cores)")

# Add labels to points with slight offsets
add_labels(carc_cores, carc_parallel_times, offset=0.05)
add_labels(epyc_cores, epyc_parallel_times, offset=0.05)

# Add labels and improve visualization
plt.xlabel("Number of Cores", fontsize=14, fontweight='bold')
plt.ylabel("Execution Time (seconds)", fontsize=14, fontweight='bold')
plt.title("Execution Time vs. Number of Cores (CARC vs AMD EPYC)", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.yscale("log")  # Logarithmic scale for better visualization
plt.tight_layout()
plt.savefig("enhanced_execution_times.png")
plt.show()

# Visualization: Speedup
plt.figure(figsize=(12, 7))
plt.plot(carc_cores, carc_speedup, label="CARC Speedup", marker='o', linestyle='-', linewidth=2)
plt.plot(epyc_cores, epyc_speedup, label="AMD EPYC Speedup", marker='s', linestyle='--', linewidth=2)
plt.axhline(y=1, color='r', linestyle='--', label="No Speedup")

# Add shaded region for optimal scaling range
plt.axvspan(4, 64, color='purple', alpha=0.2, label="Optimal Scaling Range (4-64 cores)")

# Add labels to points with slight offsets
add_labels(carc_cores, carc_speedup, offset=0.1)
add_labels(epyc_cores, epyc_speedup, offset=0.1)

# Add labels and improve visualization
plt.xlabel("Number of Cores", fontsize=14, fontweight='bold')
plt.ylabel("Speedup", fontsize=14, fontweight='bold')
plt.title("Speedup vs. Number of Cores (CARC vs AMD EPYC)", fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("enhanced_speedup.png")
plt.show()
