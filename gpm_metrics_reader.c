/*
 * NVIDIA GPM Metrics Reader for MIG Slices
 * 
 * Reads GPU Performance Metrics on NVIDIA MIG instances using GPM NVML API
 * 
 * Compile:
 *   gcc -o gpm_metrics_reader gpm_metrics_reader.c -lnvidia-ml -I/usr/local/cuda/include -L/usr/local/cuda/lib64
 * 
 * Usage:
 *   ./gpm_metrics_reader
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <nvml.h>

#define MAX_MIG_DEVICES 64
#define SAMPLE_INTERVAL_MS 150

// List of metric to query - add/remove them here
// https://docs.nvidia.com/deploy/nvml-api/group__nvmlGpmEnums.html
static const nvmlGpmMetricId_t METRICS_TO_QUERY[] = {
    NVML_GPM_METRIC_GRAPHICS_UTIL,
    NVML_GPM_METRIC_SM_UTIL,
    NVML_GPM_METRIC_SM_OCCUPANCY,
    NVML_GPM_METRIC_INTEGER_UTIL,
    NVML_GPM_METRIC_ANY_TENSOR_UTIL,
    NVML_GPM_METRIC_DFMA_TENSOR_UTIL,
    NVML_GPM_METRIC_HMMA_TENSOR_UTIL,
    NVML_GPM_METRIC_IMMA_TENSOR_UTIL,
    NVML_GPM_METRIC_DRAM_BW_UTIL,
    NVML_GPM_METRIC_FP64_UTIL,
    NVML_GPM_METRIC_FP32_UTIL,
    NVML_GPM_METRIC_FP16_UTIL,
    NVML_GPM_METRIC_PCIE_TX_PER_SEC,
    NVML_GPM_METRIC_PCIE_RX_PER_SEC,
};

typedef struct {
    nvmlDevice_t device;
    nvmlGpuInstance_t gpuInstance;
    nvmlComputeInstance_t computeInstance;
    unsigned int deviceIdx;
    unsigned int gpuInstanceId;
    unsigned int computeInstanceId;
} MigDeviceInfo;

void print_separator(void) {
    printf("\n");
    for (int i = 0; i < 70; i++) printf("=");
    printf("\n");
}

void print_nvml_error(const char *func, nvmlReturn_t result) {
    fprintf(stderr, "✗ %s failed: %s\n", func, nvmlErrorString(result));
}

void print_device_info(nvmlDevice_t device, unsigned int deviceIdx) {
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    char uuid[NVML_DEVICE_UUID_BUFFER_SIZE];
    nvmlReturn_t result;
    
    print_separator();
    
    result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
    if (result != NVML_SUCCESS) {
        print_nvml_error("nvmlDeviceGetName", result);
        snprintf(name, sizeof(name), "Unknown");
    }
    
    result = nvmlDeviceGetUUID(device, uuid, NVML_DEVICE_UUID_BUFFER_SIZE);
    if (result != NVML_SUCCESS) {
        print_nvml_error("nvmlDeviceGetUUID", result);
        snprintf(uuid, sizeof(uuid), "Unknown");
    }
    
    printf("GPU %u: %s\n", deviceIdx, name);
    printf("UUID: %s\n", uuid);
}

void print_mig_info(unsigned int gpuInstanceId, unsigned int computeInstanceId) {
    printf("MIG Slice - GI: %u, CI: %u\n", gpuInstanceId, computeInstanceId);
}

void print_metrics(nvmlGpmMetricsGet_t *metricsGet) {
    printf("\n  %-5s %-35s %12s %10s %8s\n", "ID", "Name", "Value", "Unit", "Status");
    printf("  %-5s %-35s %12s %10s %8s\n", "-----", "-----------------------------------", 
           "------------", "----------", "--------");
    
    // Print metrics in the order they were requested
    for (size_t i = 0; i < sizeof(METRICS_TO_QUERY) / sizeof(METRICS_TO_QUERY[0]); i++) {
        nvmlGpmMetricId_t requestedId = METRICS_TO_QUERY[i];
        
        // Find this metric in the results
        int found = 0;
        for (unsigned int j = 0; j < metricsGet->numMetrics; j++) {
            nvmlGpmMetric_t *metric = &metricsGet->metrics[j];
            
            if (metric->metricId == requestedId) {
                found = 1;
                const char *status = (metric->nvmlReturn == NVML_SUCCESS) ? "OK" : "FAIL";
                
                if (metric->nvmlReturn == NVML_SUCCESS) {
                    const char *name = metric->metricInfo.longName ? metric->metricInfo.longName : "Unknown";
                    const char *unit = metric->metricInfo.unit ? metric->metricInfo.unit : "";
                    
                    printf("  %-5u %-35s %12.2f %10s %8s\n", 
                           metric->metricId,
                           name,
                           metric->value,
                           unit,
                           status);
                } else {
                    printf("  %-5u %-35s %12s %10s %8s\n", 
                           requestedId,
                           "N/A",
                           "N/A",
                           "",
                           status);
                }
                break;
            }
        }
        
        if (!found) {
            printf("  %-5u %-35s %12s %10s %8s\n", 
                   requestedId,
                   "N/A",
                   "N/A",
                   "",
                   "MISS");
        }
    }
}

int query_gpm_metrics(nvmlDevice_t device, int isMig, unsigned int gpuInstanceId) {
    nvmlReturn_t result;
    nvmlGpmSupport_t gpmSupport;
    nvmlGpmSample_t sample1 = NULL;
    nvmlGpmSample_t sample2 = NULL;
    
    gpmSupport.version = NVML_GPM_SUPPORT_VERSION;

    // Check if GPM is supported
    result = nvmlGpmQueryDeviceSupport(device, &gpmSupport);
    if (result != NVML_SUCCESS) {
        printf("  GPM not supported on this device\n");
        return -1;
    }
    
    if (!gpmSupport.isSupportedDevice) {
        printf("  GPM support not available\n");
        return -1;
    }
    
    // Allocate sample buffers
    result = nvmlGpmSampleAlloc(&sample1);
    if (result != NVML_SUCCESS) {
        print_nvml_error("nvmlGpmSampleAlloc (sample1)", result);
        return -1;
    }
    
    result = nvmlGpmSampleAlloc(&sample2);
    if (result != NVML_SUCCESS) {
        print_nvml_error("nvmlGpmSampleAlloc (sample2)", result);
        nvmlGpmSampleFree(sample1);
        return -1;
    }
    
    // Get first sample
    if (isMig) {
        result = nvmlGpmMigSampleGet(device, gpuInstanceId, sample1);
        if (result != NVML_SUCCESS) {
            print_nvml_error("nvmlGpmMigSampleGet (sample1)", result);
            nvmlGpmSampleFree(sample1);
            nvmlGpmSampleFree(sample2);
            return -1;
        }
    } else {
        result = nvmlGpmSampleGet(device, sample1);
        if (result != NVML_SUCCESS) {
            print_nvml_error("nvmlGpmSampleGet (sample1)", result);
            nvmlGpmSampleFree(sample1);
            nvmlGpmSampleFree(sample2);
            return -1;
        }
    }
    
    // Wait for sample interval (must be >100ms)
    usleep(SAMPLE_INTERVAL_MS * 1000);
    
    // Get second sample
    if (isMig) {
        result = nvmlGpmMigSampleGet(device, gpuInstanceId, sample2);
        if (result != NVML_SUCCESS) {
            print_nvml_error("nvmlGpmMigSampleGet (sample2)", result);
            nvmlGpmSampleFree(sample1);
            nvmlGpmSampleFree(sample2);
            return -1;
        }
    } else {
        result = nvmlGpmSampleGet(device, sample2);
        if (result != NVML_SUCCESS) {
            print_nvml_error("nvmlGpmSampleGet (sample2)", result);
            nvmlGpmSampleFree(sample1);
            nvmlGpmSampleFree(sample2);
            return -1;
        }
    }
    
    // Prepare metrics structure
    nvmlGpmMetricsGet_t metricsGet;
    memset(&metricsGet, 0, sizeof(metricsGet));
    metricsGet.version = NVML_GPM_METRICS_GET_VERSION;
    metricsGet.sample1 = sample1;
    metricsGet.sample2 = sample2;
    metricsGet.numMetrics = sizeof(METRICS_TO_QUERY) / sizeof(METRICS_TO_QUERY[0]);
    
    // Set up metric IDs
    for (size_t i = 0; i < sizeof(METRICS_TO_QUERY) / sizeof(METRICS_TO_QUERY[0]); i++) {
        metricsGet.metrics[i].metricId = METRICS_TO_QUERY[i];
        metricsGet.metrics[i].nvmlReturn = NVML_ERROR_UNKNOWN;
    }
    
    // Query all metrics
    result = nvmlGpmMetricsGet(&metricsGet);
    
    if (result == NVML_SUCCESS) {
        print_metrics(&metricsGet);
    } else {
        print_nvml_error("nvmlGpmMetricsGet", result);
    }
    
    // Free sample buffers
    nvmlGpmSampleFree(sample1);
    nvmlGpmSampleFree(sample2);
    
    return (result == NVML_SUCCESS) ? 0 : -1;
}

int get_mig_devices(MigDeviceInfo *migDevices, int maxDevices) {
    nvmlReturn_t result;
    unsigned int deviceCount;
    int migCount = 0;
    
    result = nvmlDeviceGetCount(&deviceCount);
    if (result != NVML_SUCCESS) {
        print_nvml_error("nvmlDeviceGetCount", result);
        return -1;
    }
    
    for (unsigned int i = 0; i < deviceCount && migCount < maxDevices; i++) {
        nvmlDevice_t device;
        
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (result != NVML_SUCCESS) {
            continue;
        }
        
        // Check if MIG mode is enabled
        unsigned int currentMode, pendingMode;
        result = nvmlDeviceGetMigMode(device, &currentMode, &pendingMode);
        
        if (result == NVML_SUCCESS && currentMode == NVML_DEVICE_MIG_ENABLE) {
            // Enumerate GPU instances by trying each ID
            for (unsigned int gi = 0; gi < 16 && migCount < maxDevices; gi++) {
                nvmlGpuInstance_t gpuInstance;
                
                result = nvmlDeviceGetGpuInstanceById(device, gi, &gpuInstance);
                if (result != NVML_SUCCESS) {
                    continue;
                }
                
                // Enumerate compute instances
                for (unsigned int ci = 0; ci < 8 && migCount < maxDevices; ci++) {
                    nvmlComputeInstance_t computeInstance;
                    
                    result = nvmlGpuInstanceGetComputeInstanceById(gpuInstance, ci, &computeInstance);
                    if (result != NVML_SUCCESS) {
                        continue;
                    }
                    
                    // Found a valid MIG device
                    migDevices[migCount].device = device;
                    migDevices[migCount].gpuInstance = gpuInstance;
                    migDevices[migCount].computeInstance = computeInstance;
                    migDevices[migCount].deviceIdx = i;
                    migDevices[migCount].gpuInstanceId = gi;
                    migDevices[migCount].computeInstanceId = ci;
                    migCount++;
                }
            }
        }
    }
    
    return migCount;
}

int main(int argc, char *argv[]) {
    nvmlReturn_t result;
    
    // Initialize NVML
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        print_nvml_error("nvmlInit", result);
        return 1;
    }
    
    printf("✓ NVML initialized successfully\n");
    
    // Get MIG devices
    MigDeviceInfo migDevices[MAX_MIG_DEVICES];
    int migCount = get_mig_devices(migDevices, MAX_MIG_DEVICES);
    
    if (migCount < 0) {
        fprintf(stderr, "Failed to enumerate MIG devices\n");
        nvmlShutdown();
        return 1;
    }
    
    if (migCount == 0) {
        printf("\n⚠ No MIG devices found. Checking regular GPUs...\n");
        
        // Fall back to regular GPU monitoring
        unsigned int deviceCount;
        result = nvmlDeviceGetCount(&deviceCount);
        
        if (result == NVML_SUCCESS) {
            for (unsigned int i = 0; i < deviceCount; i++) {
                nvmlDevice_t device;
                result = nvmlDeviceGetHandleByIndex(i, &device);
                
                if (result == NVML_SUCCESS) {
                    print_device_info(device, i);
                    query_gpm_metrics(device, 0, 0);
                }
            }
        }
    } else {
        printf("\n✓ Found %d MIG device(s)\n", migCount);
        
        // Query metrics for each MIG device
        for (int i = 0; i < migCount; i++) {
            print_device_info(migDevices[i].device, migDevices[i].deviceIdx);
            print_mig_info(migDevices[i].gpuInstanceId, migDevices[i].computeInstanceId);
            query_gpm_metrics(migDevices[i].device, 1, migDevices[i].gpuInstanceId);
        }
    }
    
    print_separator();
    printf("\n");
    
    // Cleanup
    nvmlShutdown();
    
    return 0;
}