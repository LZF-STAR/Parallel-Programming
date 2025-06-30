#include "guessing_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// 常量定义
#define MAX_STRING_LENGTH 256
#define BLOCK_SIZE 256

// GPU上的字符串结构
struct GPUString {
    char data[MAX_STRING_LENGTH];
    int length;
    
    __device__ __host__ GPUString() : length(0) {
        data[0] = '\0';
    }
    
    __device__ __host__ void set(const char* str) {
        length = 0;
        while (str[length] && length < MAX_STRING_LENGTH - 1) {
            data[length] = str[length];
            length++;
        }
        data[length] = '\0';
    }
    
    __device__ __host__ void append(const char* str) {
        int str_len = 0;
        while (str[str_len]) str_len++; // 计算字符串长度
        
        if (length + str_len < MAX_STRING_LENGTH - 1) {
            for (int i = 0; i < str_len; i++) {
                data[length + i] = str[i];
            }
            length += str_len;
            data[length] = '\0';
        }
    }
    
    __device__ __host__ const char* c_str() const {
        return data;
    }
};

// CUDA内核：处理单个segment的情况
__global__ void generate_single_segment_kernel(
    GPUString* segment_values,
    int num_values,
    GPUString* output_guesses,
    int* output_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_values) {
        output_guesses[idx] = segment_values[idx];
        atomicAdd(output_count, 1);
    }
}

// CUDA内核：处理多个segment的情况
__global__ void generate_multi_segment_kernel(
    const char* base_guess,
    GPUString* segment_values,
    int num_values,
    GPUString* output_guesses,
    int* output_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_values) {
        // 复制基础猜测
        GPUString result;
        result.set(base_guess);
        
        // 追加当前segment的值
        result.append(segment_values[idx].c_str());
        
        output_guesses[idx] = result;
        atomicAdd(output_count, 1);
    }
}

// GPUPriorityQueue 类的实现

// 构造函数
GPUPriorityQueue::GPUPriorityQueue() : PriorityQueue(), gpu_initialized(false) {
    d_segment_values = nullptr;
    d_output_guesses = nullptr;
    d_output_count = nullptr;
    d_base_guess = nullptr;
    initializeGPU();
}

// 析构函数
GPUPriorityQueue::~GPUPriorityQueue() {
    cleanupGPU();
}

void GPUPriorityQueue::initializeGPU() {
    if (gpu_initialized) return;
    
    // 分配GPU内存
    CUDA_CHECK(cudaMalloc(&d_segment_values, MAX_SEGMENT_VALUES * sizeof(GPUString)));
    CUDA_CHECK(cudaMalloc(&d_output_guesses, MAX_OUTPUT_GUESSES * sizeof(GPUString)));
    CUDA_CHECK(cudaMalloc(&d_output_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_base_guess, MAX_STRING_LENGTH));
    
    gpu_initialized = true;
    printf("GPU memory initialized successfully!\n");
}

void GPUPriorityQueue::cleanupGPU() {
    if (!gpu_initialized) return;
    
    if (d_segment_values) cudaFree(d_segment_values);
    if (d_output_guesses) cudaFree(d_output_guesses);
    if (d_output_count) cudaFree(d_output_count);
    if (d_base_guess) cudaFree(d_base_guess);
    
    gpu_initialized = false;
}

// GPU优化的Generate函数
void GPUPriorityQueue::GenerateGPU(PT pt) {
    CalProb(pt);
    
    if (pt.content.size() == 1) {
        generateSingleSegmentGPU(pt);
    } else {
        generateMultiSegmentGPU(pt);
    }
}

void GPUPriorityQueue::generateSingleSegmentGPU(PT pt) {
    // 获取segment指针
    segment* a = getSegmentPointer(pt.content[0]);
    if (!a) return;
    
    int num_values = pt.max_indices[0];
    
    if (num_values > MAX_SEGMENT_VALUES) {
        // 如果数据量太大，分批处理
        processBatches(pt, num_values);
        return;
    }
    
    // 准备主机端数据 - 使用C风格数组避免STL
    GPUString* h_segment_values = new GPUString[num_values];
    for (int i = 0; i < num_values; i++) {
        h_segment_values[i].set(a->ordered_values[i].c_str());
    }
    
    // 复制到GPU
    CUDA_CHECK(cudaMemcpy(d_segment_values, h_segment_values, 
                         num_values * sizeof(GPUString), cudaMemcpyHostToDevice));
    
    // 重置输出计数
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_output_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    
    // 启动CUDA内核
    int grid_size = (num_values + BLOCK_SIZE - 1) / BLOCK_SIZE;
    generate_single_segment_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_segment_values, num_values, d_output_guesses, d_output_count
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 复制结果回主机
    copyResultsToHost(num_values);
    
    // 清理内存
    delete[] h_segment_values;
}

void GPUPriorityQueue::generateMultiSegmentGPU(PT pt) {
    // 构建基础猜测字符串
    string base_guess = buildBaseGuess(pt);
    
    // 获取最后一个segment
    segment* a = getSegmentPointer(pt.content[pt.content.size() - 1]);
    if (!a) return;
    
    int num_values = pt.max_indices[pt.content.size() - 1];
    
    if (num_values > MAX_SEGMENT_VALUES) {
        // 如果数据量太大，分批处理
        processBatches(pt, num_values);
        return;
    }
    
    // 准备segment值 - 使用C风格数组
    GPUString* h_segment_values = new GPUString[num_values];
    for (int i = 0; i < num_values; i++) {
        h_segment_values[i].set(a->ordered_values[i].c_str());
    }
    
    // 复制数据到GPU
    CUDA_CHECK(cudaMemcpy(d_segment_values, h_segment_values, 
                         num_values * sizeof(GPUString), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_base_guess, base_guess.c_str(), 
                         base_guess.length() + 1, cudaMemcpyHostToDevice));
    
    // 重置输出计数
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_output_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    
    // 启动CUDA内核
    int grid_size = (num_values + BLOCK_SIZE - 1) / BLOCK_SIZE;
    generate_multi_segment_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_base_guess, d_segment_values, num_values, d_output_guesses, d_output_count
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 复制结果回主机
    copyResultsToHost(num_values);
    
    // 清理内存
    delete[] h_segment_values;
}

segment* GPUPriorityQueue::getSegmentPointer(const segment& seg) {
    if (seg.type == 1) {
        return &m.letters[m.FindLetter(seg)];
    } else if (seg.type == 2) {
        return &m.digits[m.FindDigit(seg)];
    } else if (seg.type == 3) {
        return &m.symbols[m.FindSymbol(seg)];
    }
    return nullptr;
}

string GPUPriorityQueue::buildBaseGuess(const PT& pt) {
    string base_guess;
    int seg_idx = 0;
    
    for (int idx : pt.curr_indices) {
        if (seg_idx == pt.content.size() - 1) break;
        
        if (pt.content[seg_idx].type == 1) {
            base_guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
        } else if (pt.content[seg_idx].type == 2) {
            base_guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
        } else if (pt.content[seg_idx].type == 3) {
            base_guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
        }
        seg_idx++;
    }
    
    return base_guess;
}

void GPUPriorityQueue::copyResultsToHost(int num_values) {
    // 复制结果回主机 - 使用C风格数组
    GPUString* h_output_guesses = new GPUString[num_values];
    CUDA_CHECK(cudaMemcpy(h_output_guesses, d_output_guesses, 
                         num_values * sizeof(GPUString), cudaMemcpyDeviceToHost));
    
    // 转换为std::string并添加到guesses
    for (int i = 0; i < num_values; i++) {
        guesses.push_back(string(h_output_guesses[i].c_str()));
        total_guesses++;
    }
    
    // 清理内存
    delete[] h_output_guesses;
}

void GPUPriorityQueue::processBatches(const PT& pt, int total_values) {
    // 对于超大数据集，分批处理
    const int batch_size = MAX_SEGMENT_VALUES;
    
    for (int start = 0; start < total_values; start += batch_size) {
        int end = (start + batch_size < total_values) ? start + batch_size : total_values;
        int current_batch_size = end - start;
        
        // 创建当前批次的PT副本
        PT batch_pt = pt;
        if (pt.content.size() == 1) {
            batch_pt.max_indices[0] = current_batch_size;
            generateSingleSegmentBatch(batch_pt, start);
        } else {
            batch_pt.max_indices[pt.content.size() - 1] = current_batch_size;
            generateMultiSegmentBatch(batch_pt, start);
        }
    }
}

void GPUPriorityQueue::generateSingleSegmentBatch(const PT& pt, int start_idx) {
    segment* a = getSegmentPointer(pt.content[0]);
    if (!a) return;
    
    int batch_size = pt.max_indices[0];
    
    GPUString* h_segment_values = new GPUString[batch_size];
    for (int i = 0; i < batch_size; i++) {
        h_segment_values[i].set(a->ordered_values[start_idx + i].c_str());
    }
    
    CUDA_CHECK(cudaMemcpy(d_segment_values, h_segment_values, 
                         batch_size * sizeof(GPUString), cudaMemcpyHostToDevice));
    
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_output_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    
    int grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    generate_single_segment_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_segment_values, batch_size, d_output_guesses, d_output_count
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    copyResultsToHost(batch_size);
    
    delete[] h_segment_values;
}

void GPUPriorityQueue::generateMultiSegmentBatch(const PT& pt, int start_idx) {
    string base_guess = buildBaseGuess(pt);
    segment* a = getSegmentPointer(pt.content[pt.content.size() - 1]);
    if (!a) return;
    
    int batch_size = pt.max_indices[pt.content.size() - 1];
    
    GPUString* h_segment_values = new GPUString[batch_size];
    for (int i = 0; i < batch_size; i++) {
        h_segment_values[i].set(a->ordered_values[start_idx + i].c_str());
    }
    
    CUDA_CHECK(cudaMemcpy(d_segment_values, h_segment_values, 
                         batch_size * sizeof(GPUString), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_base_guess, base_guess.c_str(), 
                         base_guess.length() + 1, cudaMemcpyHostToDevice));
    
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_output_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    
    int grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    generate_multi_segment_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_base_guess, d_segment_values, batch_size, d_output_guesses, d_output_count
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    copyResultsToHost(batch_size);
    
    delete[] h_segment_values;
}

// 更新的PopNext函数，使用GPU优化
void GPUPriorityQueue::PopNext() {
    // 使用GPU优化的Generate函数
    GenerateGPU(priority.front());
    
    // 生成新的PT（这部分保持原样，因为它主要是逻辑操作）
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts) {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++) {
            if (iter != priority.end() - 1 && iter != priority.begin()) {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) {
                    priority.insert(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1) {
                priority.push_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob) {
                priority.insert(iter, pt);
                break;
            }
        }
    }
    
    priority.erase(priority.begin());
}