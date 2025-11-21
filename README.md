# NVIDIA GPM Metrics Reader & Monitor

A lightweight C tool for reading GPU Performance Metrics (GPM) on NVIDIA GPUs, with support for MIG (Multi-Instance GPU) slices. Includes a Python wrapper for continuous monitoring and CSV export.

## Prerequisites

- NVIDIA GPU with GPM support (Hopper or newer: H100, GH200, etc.)
- NVIDIA driver 525.60.13 or newer
- CUDA Toolkit (for NVML headers)
- GCC compiler
- Python 3.6+ (only for monitoring wrapper)

## Installation

### 1. Clone or download the project

```bash
git clone <repository>
cd gpm-metrics-reader
```

### 2. Compile the C binary

```bash
gcc -o gpm_metrics_reader gpm_metrics_reader.c -lnvidia-ml -I/usr/local/cuda/include -L/usr/local/cuda/lib64
```
## Usage

### Basic Usage

Run once to see current metrics:

```bash
./gpm_metrics_reader
```


### Continuous Monitoring

Use the Python wrapper for continuous sampling:

```bash
# Monitor continuously, save to CSV
python gpm_monitor.py --output metrics.csv

# Sample every 500ms
python gpm_monitor.py --output metrics.csv --interval 500

# Run for 60 seconds
python gpm_monitor.py --output metrics.csv --duration 60

# Custom binary path
python gpm_monitor.py --output metrics.csv --binary /path/to/gpm_metrics_reader
```
**Arguments:**
- `--output FILE` (required): Output CSV file path
- `--interval MSEC` (default: 1000): Sampling interval in milliseconds
- `--duration SEC` (default: infinite): Total duration in seconds
- `--binary PATH` (default: ./gpm_metrics_reader): Path to binary

**CSV Output Format:**
```csv
timestamp,device_id,device_name,gpu_instance_id,compute_instance_id,metric_id,metric_name,value,unit
2025-11-21T10:30:45.123,0,NVIDIA GH200 480GB,,,1,GRAPHICS_UTIL,0.01,%
2025-11-21T10:30:45.123,0,NVIDIA GH200 480GB,,,2,SM_UTIL,0.00,%
```

## Configuration

### Customizing Metrics

Edit the `METRICS_TO_QUERY` array in `gpm_metrics_reader.c` to select which metrics to monitor:

```c
// List of metrics to query - add/remove them here
static const nvmlGpmMetricId_t METRICS_TO_QUERY[] = {
    NVML_GPM_METRIC_SM_UTIL,           // ID 2
    NVML_GPM_METRIC_DRAM_BW_UTIL,      // ID 10
    NVML_GPM_METRIC_FP16_UTIL,         // ID 13
};
```

Recompile after making changes:
```bash
gcc -o gpm_metrics_reader gpm_metrics_reader.c -lnvidia-ml -I/usr/local/cuda/include -L/usr/local/cuda/lib64
```

### Available Metrics
See [NVIDIA NVML documentation](https://docs.nvidia.com/deploy/nvml-api/group__nvmlGpmEnums.html) for the complete list.

## References

- [NVIDIA NVML API Documentation](https://docs.nvidia.com/deploy/nvml-api/)
- [GPM Functions Reference](https://docs.nvidia.com/deploy/nvml-api/group__nvmlGpmFunctions.html)
- [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
