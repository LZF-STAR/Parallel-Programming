//guessing_gpu.cu
// 优化后的GPU猜测生成代码

#include "PCFG.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <cstring>

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

// 优化的常量定义
#define MAX_STRING_LENGTH 64
#define BLOCK_SIZE 256
#define MIN_GPU_SIZE 100000    // 只有足够大的数据才用GPU
#define WARP_SIZE 32

// 优化的GPU字符串操作内核
__global__ void single_segment_kernel_optimized(
    char* segment_data,     // 紧凑存储的字符串数据
    int* segment_offsets,   // 每个字符串的起始位置
    int* segment_lengths,   // 每个字符串的长度
    int num_segments,
    char* output_data,      // 输出字符串数据
    int* output_offsets     // 输出字符串偏移
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_segments) {
        int input_offset = segment_offsets[idx];
        int output_offset = output_offsets[idx];
        int length = segment_lengths[idx];
        
        // 直接内存复制，效率更高
        for (int i = 0; i < length; i++) {
            output_data[output_offset + i] = segment_data[input_offset + i];
        }
        output_data[output_offset + length] = '\0';
    }
}

__global__ void multi_segment_kernel_optimized(
    char* base_data,        // 基础字符串
    int base_length,
    char* segment_data,     // segment字符串数据
    int* segment_offsets,   // segment偏移
    int* segment_lengths,   // segment长度
    int num_segments,
    char* output_data,      // 输出数据
    int* output_offsets     // 输出偏移
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_segments) {
        int seg_offset = segment_offsets[idx];
        int out_offset = output_offsets[idx];
        int seg_length = segment_lengths[idx];
        
        // 复制基础字符串
        for (int i = 0; i < base_length; i++) {
            output_data[out_offset + i] = base_data[i];
        }
        
        // 追加segment字符串
        for (int i = 0; i < seg_length; i++) {
            output_data[out_offset + base_length + i] = segment_data[seg_offset + i];
        }
        
        output_data[out_offset + base_length + seg_length] = '\0';
    }
}

// GPU内存管理类
class GPUMemoryManager {
private:
    // 预分配的GPU内存
    char* d_input_buffer;
    char* d_output_buffer;
    char* d_base_buffer;
    int* d_offsets;
    int* d_lengths;
    int* d_output_offsets;
    
    // 主机端缓冲区
    std::vector<char> h_input_buffer;
    std::vector<char> h_output_buffer;
    std::vector<int> h_offsets;
    std::vector<int> h_lengths;
    std::vector<int> h_output_offsets;
    
    static const size_t MAX_BUFFER_SIZE = 256 * 1024 * 1024; // 256MB
    static const int MAX_SEGMENTS = 1000000;
    
    bool initialized;
    
public:
    GPUMemoryManager() : initialized(false) {
        initializeMemory();
    }
    
    ~GPUMemoryManager() {
        cleanup();
    }
    
    void initializeMemory() {
        if (initialized) return;
        
        // 分配GPU内存
        CUDA_CHECK(cudaMalloc(&d_input_buffer, MAX_BUFFER_SIZE));
        CUDA_CHECK(cudaMalloc(&d_output_buffer, MAX_BUFFER_SIZE));
        CUDA_CHECK(cudaMalloc(&d_base_buffer, MAX_STRING_LENGTH));
        CUDA_CHECK(cudaMalloc(&d_offsets, MAX_SEGMENTS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_lengths, MAX_SEGMENTS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_output_offsets, MAX_SEGMENTS * sizeof(int)));
        
        // 预分配主机内存
        h_input_buffer.reserve(MAX_BUFFER_SIZE);
        h_output_buffer.reserve(MAX_BUFFER_SIZE);
        h_offsets.reserve(MAX_SEGMENTS);
        h_lengths.reserve(MAX_SEGMENTS);
        h_output_offsets.reserve(MAX_SEGMENTS);
        
        initialized = true;
        printf("GPU Memory Manager initialized: 256MB allocated\n");
    }
    
    void cleanup() {
        if (initialized) {
            cudaFree(d_input_buffer);
            cudaFree(d_output_buffer);
            cudaFree(d_base_buffer);
            cudaFree(d_offsets);
            cudaFree(d_lengths);
            cudaFree(d_output_offsets);
            initialized = false;
        }
    }
    
    // 处理单segment情况
    bool processSingleSegment(const std::vector<std::string>& values, 
                             std::vector<std::string>& results) {
        if (!initialized || values.size() < MIN_GPU_SIZE) {
            return false; // 使用CPU处理
        }
        
        // 清空缓冲区
        h_input_buffer.clear();
        h_offsets.clear();
        h_lengths.clear();
        h_output_offsets.clear();
        
        // 准备输入数据
        int current_offset = 0;
        int output_offset = 0;
        
        for (const auto& value : values) {
            h_offsets.push_back(current_offset);
            h_lengths.push_back(value.length());
            h_output_offsets.push_back(output_offset);
            
            // 复制字符串数据
            for (char c : value) {
                h_input_buffer.push_back(c);
            }
            
            current_offset += value.length();
            output_offset += value.length() + 1; // +1 for null terminator
        }
        
        // 检查缓冲区大小
        if (h_input_buffer.size() > MAX_BUFFER_SIZE || output_offset > MAX_BUFFER_SIZE) {
            return false; // 数据太大，使用CPU
        }
        
        // 传输到GPU
        CUDA_CHECK(cudaMemcpy(d_input_buffer, h_input_buffer.data(), 
                             h_input_buffer.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), 
                             h_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lengths, h_lengths.data(), 
                             h_lengths.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_output_offsets, h_output_offsets.data(), 
                             h_output_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
        
        // 启动内核
        int num_blocks = (values.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        single_segment_kernel_optimized<<<num_blocks, BLOCK_SIZE>>>(
            d_input_buffer, d_offsets, d_lengths, values.size(),
            d_output_buffer, d_output_offsets
        );
        
        // 检查内核错误
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 复制结果
        h_output_buffer.resize(output_offset);
        CUDA_CHECK(cudaMemcpy(h_output_buffer.data(), d_output_buffer,
                             output_offset, cudaMemcpyDeviceToHost));
        
        // 解析结果
        results.clear();
        results.reserve(values.size());
        for (size_t i = 0; i < values.size(); i++) {
            char* str_start = h_output_buffer.data() + h_output_offsets[i];
            results.emplace_back(str_start);
        }
        
        return true;
    }
    
    // 处理多segment情况
    bool processMultiSegment(const std::string& base_guess,
                            const std::vector<std::string>& values,
                            std::vector<std::string>& results) {
        if (!initialized || values.size() < MIN_GPU_SIZE) {
            return false; // 使用CPU处理
        }
        
        // 清空缓冲区
        h_input_buffer.clear();
        h_offsets.clear();
        h_lengths.clear();
        h_output_offsets.clear();
        
        // 准备输入数据
        int current_offset = 0;
        int output_offset = 0;
        int base_len = base_guess.length();
        
        for (const auto& value : values) {
            h_offsets.push_back(current_offset);
            h_lengths.push_back(value.length());
            h_output_offsets.push_back(output_offset);
            
            // 复制segment数据
            for (char c : value) {
                h_input_buffer.push_back(c);
            }
            
            current_offset += value.length();
            output_offset += base_len + value.length() + 1;
        }
        
        // 检查缓冲区大小
        if (h_input_buffer.size() > MAX_BUFFER_SIZE || output_offset > MAX_BUFFER_SIZE) {
            return false;
        }
        
        // 传输到GPU
        CUDA_CHECK(cudaMemcpy(d_base_buffer, base_guess.c_str(), 
                             base_len, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_input_buffer, h_input_buffer.data(), 
                             h_input_buffer.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), 
                             h_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lengths, h_lengths.data(), 
                             h_lengths.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_output_offsets, h_output_offsets.data(), 
                             h_output_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
        
        // 启动内核
        int num_blocks = (values.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        multi_segment_kernel_optimized<<<num_blocks, BLOCK_SIZE>>>(
            d_base_buffer, base_len,
            d_input_buffer, d_offsets, d_lengths, values.size(),
            d_output_buffer, d_output_offsets
        );
        
        // 检查内核错误
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 复制结果
        h_output_buffer.resize(output_offset);
        CUDA_CHECK(cudaMemcpy(h_output_buffer.data(), d_output_buffer,
                             output_offset, cudaMemcpyDeviceToHost));
        
        // 解析结果
        results.clear();
        results.reserve(values.size());
        for (size_t i = 0; i < values.size(); i++) {
            char* str_start = h_output_buffer.data() + h_output_offsets[i];
            results.emplace_back(str_start);
        }
        
        return true;
    }
};

// 全局GPU内存管理器
static GPUMemoryManager* g_gpu_manager = nullptr;

// GPU系统初始化
void initializeGPUSystem() {
    if (!g_gpu_manager) {
        g_gpu_manager = new GPUMemoryManager();
        atexit([]() {
            if (g_gpu_manager) {
                delete g_gpu_manager;
                g_gpu_manager = nullptr;
            }
        });
    }
}

// 优化后的Generate函数
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        // 获取segment数据
        segment *a;
        if (pt.content[0].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[0])];
        } else if (pt.content[0].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[0])];
        } else {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        int num_values = pt.max_indices[0];
        
        // 尝试使用GPU处理
        if (g_gpu_manager && num_values >= MIN_GPU_SIZE) {
            std::vector<std::string> gpu_results;
            if (g_gpu_manager->processSingleSegment(a->ordered_values, gpu_results)) {
                // GPU处理成功
                for (const auto& result : gpu_results) {
                    guesses.emplace_back(result);
                }
                total_guesses += gpu_results.size();
                return;
            }
        }
        
        // GPU处理失败或数据量太小，使用CPU
        for (int i = 0; i < num_values; i++) {
            guesses.emplace_back(a->ordered_values[i]);
            total_guesses++;
        }
    }
    else
    {
        // 构建基础猜测字符串
        std::string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1) {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 2) {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 3) {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx++;
            if (seg_idx == pt.content.size() - 1) break;
        }

        // 获取最后一个segment
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        } else if (pt.content[pt.content.size() - 1].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        } else {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }

        int num_values = pt.max_indices[pt.content.size() - 1];
        
        // 尝试使用GPU处理
        if (g_gpu_manager && num_values >= MIN_GPU_SIZE) {
            std::vector<std::string> gpu_results;
            if (g_gpu_manager->processMultiSegment(guess, a->ordered_values, gpu_results)) {
                // GPU处理成功
                for (const auto& result : gpu_results) {
                    guesses.emplace_back(result);
                }
                total_guesses += gpu_results.size();
                return;
            }
        }
        
        // GPU处理失败或数据量太小，使用CPU
        for (int i = 0; i < num_values; i++) {
            std::string temp = guess + a->ordered_values[i];
            guesses.emplace_back(temp);
            total_guesses++;
        }
    }
}