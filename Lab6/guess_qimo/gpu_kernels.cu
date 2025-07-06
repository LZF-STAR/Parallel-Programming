#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>
#include "unified_framework.h"

// GPU常量内存
__constant__ char d_charset[256];
__constant__ int d_segment_offsets[32];

// 任务描述符（GPU友好格式）
struct GPUTaskDescriptor {
    int segment_count;
    int segment_types[8];     // 最多8个segment
    int segment_lengths[8];
    int value_counts[8];
    int value_offsets[8];     // 在值数组中的偏移
    int total_combinations;
};

// GPU MD5相关定义
#define GPU_F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define GPU_G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define GPU_H(x, y, z) ((x) ^ (y) ^ (z))
#define GPU_I(x, y, z) ((y) ^ ((x) | (~z)))

__device__ inline uint32_t rotateLeft(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// 统一内核 - 处理所有类型的PT任务
__global__ void unifiedKernel(
    GPUTaskDescriptor* tasks,
    char* segment_values,       // 所有segment的可能值
    char* output_passwords,     // 输出的密码
    uint32_t* output_hashes,    // 输出的MD5哈希
    int* output_offsets,        // 每个线程的输出偏移
    int num_tasks,
    int max_output_per_thread
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // 共享内存用于协作
    extern __shared__ char shared_mem[];
    char* local_buffer = shared_mem + threadIdx.x * 64; // 每线程64字节
    
    // 任务级并行
    for (int task_id = tid; task_id < num_tasks; task_id += total_threads) {
        GPUTaskDescriptor& task = tasks[task_id];
        
        // 计算该线程处理的组合范围
        int combinations_per_thread = (task.total_combinations + total_threads - 1) / total_threads;
        int start_combo = tid * combinations_per_thread;
        int end_combo = min(start_combo + combinations_per_thread, task.total_combinations);
        
        // 数据级并行 - 生成密码组合
        for (int combo_id = start_combo; combo_id < end_combo; combo_id++) {
            int remaining = combo_id;
            int password_len = 0;
            
            // 构建密码
            for (int seg = 0; seg < task.segment_count; seg++) {
                int value_count = task.value_counts[seg];
                int value_idx = remaining % value_count;
                remaining /= value_count;
                
                // 从全局内存复制segment值
                int value_offset = task.value_offsets[seg] + value_idx * task.segment_lengths[seg];
                for (int i = 0; i < task.segment_lengths[seg]; i++) {
                    local_buffer[password_len++] = segment_values[value_offset + i];
                }
            }
            
            // 计算输出位置
            int output_idx = atomicAdd(&output_offsets[tid], 1);
            if (output_idx >= max_output_per_thread) break;
            
            // 写入密码
            int global_offset = tid * max_output_per_thread * 32 + output_idx * 32;
            for (int i = 0; i < password_len; i++) {
                output_passwords[global_offset + i] = local_buffer[i];
            }
            output_passwords[global_offset + password_len] = '\0';
            
            // 简化的MD5计算（实际应该完整实现）
            // 这里只是示例
            output_hashes[tid * max_output_per_thread + output_idx] = combo_id;
        }
    }
}

// 优化的多segment内核
__global__ void multiSegmentKernel(
    GPUTaskDescriptor* task,
    char* segment_values,
    char* output_passwords,
    int* output_count,
    int max_outputs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // 使用寄存器存储索引
    int indices[8];
    for (int i = 0; i < 8; i++) indices[i] = 0;
    
    // 分配工作
    for (int combo_id = tid; combo_id < task->total_combinations; combo_id += stride) {
        // 计算当前组合的索引
        int temp = combo_id;
        for (int i = 0; i < task->segment_count; i++) {
            indices[i] = temp % task->value_counts[i];
            temp /= task->value_counts[i];
        }
        
        // 构建密码
        char password[64];
        int pos = 0;
        
        for (int seg = 0; seg < task->segment_count; seg++) {
            int offset = task->value_offsets[seg] + indices[seg] * task->segment_lengths[seg];
            for (int i = 0; i < task->segment_lengths[seg]; i++) {
                password[pos++] = segment_values[offset + i];
            }
        }
        password[pos] = '\0';
        
        // 原子操作获取输出位置
        int out_idx = atomicAdd(output_count, 1);
        if (out_idx < max_outputs) {
            // 合并写入以提高带宽利用率
            for (int i = 0; i <= pos; i++) {
                output_passwords[out_idx * 64 + i] = password[i];
            }
        }
    }
}

// GPU内存管理类
class GPUMemoryManager {
private:
    struct MemoryPool {
        void* device_ptr;
        size_t size;
        size_t used;
        cudaStream_t stream;
    };
    
    static const int NUM_POOLS = 4;
    MemoryPool pools[NUM_POOLS];
    int current_pool = 0;
    
public:
    GPUMemoryManager(size_t pool_size = 256 * 1024 * 1024) { // 256MB per pool
        for (int i = 0; i < NUM_POOLS; i++) {
            cudaMalloc(&pools[i].device_ptr, pool_size);
            pools[i].size = pool_size;
            pools[i].used = 0;
            cudaStreamCreate(&pools[i].stream);
        }
    }
    
    ~GPUMemoryManager() {
        for (int i = 0; i < NUM_POOLS; i++) {
            cudaFree(pools[i].device_ptr);
            cudaStreamDestroy(pools[i].stream);
        }
    }
    
    void* allocate(size_t size, cudaStream_t& stream) {
        // 轮转池分配
        MemoryPool& pool = pools[current_pool];
        if (pool.used + size > pool.size) {
            current_pool = (current_pool + 1) % NUM_POOLS;
            pools[current_pool].used = 0;
        }
        
        void* ptr = (char*)pools[current_pool].device_ptr + pools[current_pool].used;
        pools[current_pool].used += size;
        stream = pools[current_pool].stream;
        return ptr;
    }
    
    void reset() {
        for (int i = 0; i < NUM_POOLS; i++) {
            pools[i].used = 0;
        }
        current_pool = 0;
    }
};

// GPU批处理执行器
class GPUBatchExecutor {
private:
    GPUMemoryManager memory_manager;
    
    // 多流并发
    static const int NUM_STREAMS = 3;
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t events[NUM_STREAMS];
    
    // 设备属性
    cudaDeviceProp device_prop;
    int max_threads_per_block;
    int max_blocks;
    
public:
    GPUBatchExecutor() {
        cudaGetDeviceProperties(&device_prop, 0);
        max_threads_per_block = device_prop.maxThreadsPerBlock;
        max_blocks = device_prop.multiProcessorCount * 32;
        
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamCreate(&streams[i]);
            cudaEventCreate(&events[i]);
        }
    }
    
    ~GPUBatchExecutor() {
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
            cudaEventDestroy(events[i]);
        }
    }
    
    void executeBatch(PTBatch& batch, StringBatch& results) {
        int stream_id = 0;
        
        for (size_t i = 0; i < batch.size(); i++) {
            cudaStream_t stream = streams[stream_id];
            stream_id = (stream_id + 1) % NUM_STREAMS;
            
            // 准备GPU任务描述符
            GPUTaskDescriptor host_task;
            prepareTaskDescriptor(batch.data()[i], host_task);
            
            // 分配设备内存
            GPUTaskDescriptor* d_task;
            char* d_values;
            char* d_output;
            int* d_count;
            
            size_t values_size = calculateValuesSize(batch.data()[i]);
            cudaMallocAsync(&d_task, sizeof(GPUTaskDescriptor), stream);
            cudaMallocAsync(&d_values, values_size, stream);
            cudaMallocAsync(&d_output, 1024 * 1024, stream); // 1MB输出缓冲
            cudaMallocAsync(&d_count, sizeof(int), stream);
            
            // 复制数据到设备
            cudaMemcpyAsync(d_task, &host_task, sizeof(GPUTaskDescriptor), 
                           cudaMemcpyHostToDevice, stream);
            copySegmentValues(batch.data()[i], d_values, stream);
            cudaMemsetAsync(d_count, 0, sizeof(int), stream);
            
            // 计算网格和块大小
            int block_size = 256;
            int grid_size = min((host_task.total_combinations + block_size - 1) / block_size, 
                               max_blocks);
            
            // 启动内核
            multiSegmentKernel<<<grid_size, block_size, 0, stream>>>(
                d_task, d_values, d_output, d_count, 16384
            );
            
            // 异步复制结果
            cudaEventRecord(events[stream_id], stream);
            
            // 清理（延迟到流完成）
            cudaFreeAsync(d_task, stream);
            cudaFreeAsync(d_values, stream);
            cudaFreeAsync(d_output, stream);
            cudaFreeAsync(d_count, stream);
        }
        
        // 等待所有流完成
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaEventSynchronize(events[i]);
        }
    }
    
private:
    void prepareTaskDescriptor(const PT& pt, GPUTaskDescriptor& desc) {
        desc.segment_count = pt.content.size();
        desc.total_combinations = 1;
        
        for (int i = 0; i < desc.segment_count; i++) {
            desc.segment_types[i] = pt.content[i].type;
            desc.segment_lengths[i] = pt.content[i].length;
            desc.value_counts[i] = pt.content[i].ordered_values.size();
            desc.total_combinations *= desc.value_counts[i];
        }
    }
    
    size_t calculateValuesSize(const PT& pt) {
        size_t total = 0;
        for (const auto& seg : pt.content) {
            total += seg.ordered_values.size() * seg.length;
        }
        return total;
    }
    
    void copySegmentValues(const PT& pt, char* d_values, cudaStream_t stream) {
        // 实际实现需要正确复制所有segment值
        // 这里简化处理
    }
};

// GPU自适应批处理器
class GPUAdaptiveBatcher {
private:
    struct BatchStats {
        double throughput;
        double gpu_utilization;
        size_t batch_size;
        double execution_time;
    };
    
    std::vector<BatchStats> history;
    size_t current_batch_size = 10000;
    size_t min_batch_size = 1000;
    size_t max_batch_size = 1000000;
    
public:
    size_t getOptimalBatchSize() {
        if (history.size() < 3) {
            return current_batch_size;
        }
        
        // 使用二次回归找最优批大小
        double best_throughput = 0;
        size_t best_size = current_batch_size;
        
        for (const auto& stat : history) {
            if (stat.throughput > best_throughput) {
                best_throughput = stat.throughput;
                best_size = stat.batch_size;
            }
        }
        
        // 渐进式调整
        if (best_size > current_batch_size) {
            current_batch_size = min(current_batch_size * 1.2, max_batch_size);
        } else if (best_size < current_batch_size) {
            current_batch_size = max(current_batch_size * 0.8, min_batch_size);
        }
        
        return current_batch_size;
    }
    
    void recordExecution(size_t batch_size, double time, double throughput) {
        BatchStats stat;
        stat.batch_size = batch_size;
        stat.execution_time = time;
        stat.throughput = throughput;
        
        // 获取GPU利用率
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        // 实际应该使用NVML获取真实利用率
        stat.gpu_utilization = min(1.0, throughput / (prop.clockRate * 1000.0));
        
        history.push_back(stat);
        
        // 保持历史窗口
        if (history.size() > 10) {
            history.erase(history.begin());
        }
    }
};

// 导出的C++接口
extern "C" {
    void* createGPUExecutor() {
        return new GPUBatchExecutor();
    }
    
    void destroyGPUExecutor(void* executor) {
        delete static_cast<GPUBatchExecutor*>(executor);
    }
    
    void executeGPUBatch(void* executor, void* batch, void* results) {
        auto* gpu_executor = static_cast<GPUBatchExecutor*>(executor);
        auto* pt_batch = static_cast<PTBatch*>(batch);
        auto* string_batch = static_cast<StringBatch*>(results);
        
        gpu_executor->executeBatch(*pt_batch, *string_batch);
    }
}