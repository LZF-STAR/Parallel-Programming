#include "guessing_cuda_multi_pt.h"
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

// GPU字符串结构
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
        while (str[str_len]) str_len++;
        
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

// 多PT处理的统一CUDA内核
__global__ void generate_multi_pt_batch_kernel(
    GPUString* segment_values,      // 所有PT的segment值
    char* base_guesses,             // 所有PT的基础猜测（连续存储）
    int* pt_offsets,                // 每个PT的segment偏移
    int* pt_sizes,                  // 每个PT的segment数量
    int* pt_types,                  // PT类型标记
    int* output_offsets,            // 每个PT的输出偏移
    int num_pts,                    // PT数量
    GPUString* output_guesses,      // 输出结果
    int* output_count               // 输出计数
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 确定当前线程处理哪个PT的哪个segment值
    int pt_id = -1;
    int local_segment_idx = -1;
    
    // 通过偏移量确定线程对应的PT和segment
    int accumulated_segments = 0;
    for (int i = 0; i < num_pts; i++) {
        int pt_segment_count = pt_sizes[i];
        if (global_idx >= accumulated_segments && 
            global_idx < accumulated_segments + pt_segment_count) {
            pt_id = i;
            local_segment_idx = global_idx - accumulated_segments;
            break;
        }
        accumulated_segments += pt_segment_count;
    }
    
    // 如果找不到对应的PT，退出
    if (pt_id == -1 || local_segment_idx == -1) return;
    
    // 计算输出位置
    int output_idx = output_offsets[pt_id] + local_segment_idx;
    
    // 获取segment值
    int segment_global_idx = pt_offsets[pt_id] + local_segment_idx;
    GPUString segment_value = segment_values[segment_global_idx];
    
    GPUString result;
    
    if (pt_types[pt_id] == 0) {
        // 单segment情况：直接复制
        result = segment_value;
    } else {
        // 多segment情况：拼接基础猜测和segment值
        // 找到对应的基础猜测（基础猜测按PT顺序存储，每个最多64字符）
        char* base_guess_ptr = base_guesses + pt_id * 64;
        result.set(base_guess_ptr);
        result.append(segment_value.c_str());
    }
    
    // 写入结果
    output_guesses[output_idx] = result;
    atomicAdd(output_count, 1);
}

// 构造函数
GPUMultiPTPriorityQueue::GPUMultiPTPriorityQueue() : PriorityQueue(), gpu_initialized(false) {
    d_segment_values = nullptr;
    d_output_guesses = nullptr;
    d_output_count = nullptr;
    d_base_guesses = nullptr;
    d_pt_offsets = nullptr;
    d_pt_sizes = nullptr;
    d_pt_types = nullptr;
    initializeMultiPTGPU();
}

// 析构函数
GPUMultiPTPriorityQueue::~GPUMultiPTPriorityQueue() {
    cleanupMultiPTGPU();
}

void GPUMultiPTPriorityQueue::initializeMultiPTGPU() {
    if (gpu_initialized) return;
    
    // 分配GPU内存
    CUDA_CHECK(cudaMalloc(&d_segment_values, MAX_TOTAL_SEGMENTS * sizeof(GPUString)));
    CUDA_CHECK(cudaMalloc(&d_output_guesses, MAX_TOTAL_OUTPUTS * sizeof(GPUString)));
    CUDA_CHECK(cudaMalloc(&d_output_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_base_guesses, MAX_BATCH_PTS * MAX_BASE_GUESS_LENGTH));
    CUDA_CHECK(cudaMalloc(&d_pt_offsets, MAX_BATCH_PTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pt_sizes, MAX_BATCH_PTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pt_types, MAX_BATCH_PTS * sizeof(int)));
    
    gpu_initialized = true;
    printf("Multi-PT GPU memory initialized successfully!\n");
}

void GPUMultiPTPriorityQueue::cleanupMultiPTGPU() {
    if (!gpu_initialized) return;
    
    if (d_segment_values) cudaFree(d_segment_values);
    if (d_output_guesses) cudaFree(d_output_guesses);
    if (d_output_count) cudaFree(d_output_count);
    if (d_base_guesses) cudaFree(d_base_guesses);
    if (d_pt_offsets) cudaFree(d_pt_offsets);
    if (d_pt_sizes) cudaFree(d_pt_sizes);
    if (d_pt_types) cudaFree(d_pt_types);
    
    gpu_initialized = false;
}

// 多PT批处理主函数
void GPUMultiPTPriorityQueue::PopNext(int batch_pt_num) {
    if (priority.empty()) return;
    
    // 1. 确定实际批处理的PT数量
    int actual_batch = min(batch_pt_num, static_cast<int>(priority.size()));
    actual_batch = min(actual_batch, MAX_BATCH_PTS);
    
    if (actual_batch <= 0) return;
    
    // 2. 提取批处理的PT
    vector<PT> batch_pts(priority.begin(), priority.begin() + actual_batch);
    
    // 3. GPU批处理生成密码
    generateMultiPTBatchGPU(batch_pts);
    
    // 4. 生成新PT并更新队列
    vector<PT> new_pts;
    for (auto& pt : batch_pts) {
        vector<PT> pts = pt.NewPTs();
        for (PT& npt : pts) {
            CalProb(npt);
            new_pts.emplace_back(move(npt));
        }
    }
    
    // 5. 从队列中删除已处理的PT
    priority.erase(priority.begin(), priority.begin() + actual_batch);
    
    // 6. 将新PT按概率插入到优先队列中
    for (PT& pt : new_pts) {
        bool inserted = false;
        for (auto iter = priority.begin(); iter != priority.end(); iter++) {
            if (iter != priority.end() - 1 && iter != priority.begin()) {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) {
                    priority.emplace(iter + 1, pt);
                    inserted = true;
                    break;
                }
            }
            if (iter == priority.end() - 1) {
                priority.emplace_back(pt);
                inserted = true;
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob) {
                priority.emplace(iter, pt);
                inserted = true;
                break;
            }
        }
        if (!inserted && priority.empty()) {
            priority.emplace_back(pt);
        }
    }
}

void GPUMultiPTPriorityQueue::generateMultiPTBatchGPU(const vector<PT>& batch_pts) {
    // 1. 准备批处理数据
    MultiPTBatch batch_data;
    prepareBatchData(batch_pts, batch_data);
    
    // 2. 检查数据量是否超限
    if (batch_data.total_segments > MAX_TOTAL_SEGMENTS) {
        printf("Warning: Total segments (%d) exceeds limit (%d), processing in sub-batches\n", 
               batch_data.total_segments, MAX_TOTAL_SEGMENTS);
        // 这里可以实现子批处理逻辑
        return;
    }
    
    // 3. 传输数据到GPU
    CUDA_CHECK(cudaMemcpy(d_segment_values, batch_data.all_segment_values.data(),
                         batch_data.total_segments * sizeof(GPUString), cudaMemcpyHostToDevice));
    
    // 传输基础猜测（固定长度格式）
    vector<char> base_guess_buffer(batch_pts.size() * MAX_BASE_GUESS_LENGTH, 0);
    for (size_t i = 0; i < batch_pts.size(); i++) {
        if (i < batch_data.all_base_guesses.size()) {
            const string& base = batch_data.all_base_guesses[i];
            strncpy(&base_guess_buffer[i * MAX_BASE_GUESS_LENGTH], base.c_str(), 
                   min(static_cast<int>(base.length()), MAX_BASE_GUESS_LENGTH - 1));
        }
    }
    CUDA_CHECK(cudaMemcpy(d_base_guesses, base_guess_buffer.data(),
                         batch_pts.size() * MAX_BASE_GUESS_LENGTH, cudaMemcpyHostToDevice));
    
    // 传输PT元数据
    CUDA_CHECK(cudaMemcpy(d_pt_offsets, batch_data.pt_offsets.data(),
                         batch_pts.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pt_sizes, batch_data.pt_sizes.data(),
                         batch_pts.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pt_types, batch_data.pt_types.data(),
                         batch_pts.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    // 传输输出偏移
    vector<int> output_offsets(batch_pts.size());
    CUDA_CHECK(cudaMemcpy(d_output_count, batch_data.output_offsets.data(),
                         batch_pts.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    // 重置输出计数
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_output_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    
    // 4. 启动CUDA内核
    int grid_size = (batch_data.total_segments + BLOCK_SIZE - 1) / BLOCK_SIZE;
    generate_multi_pt_batch_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_segment_values,
        d_base_guesses,
        d_pt_offsets,
        d_pt_sizes,
        d_pt_types,
        batch_data.output_offsets.data(),
        batch_pts.size(),
        d_output_guesses,
        d_output_count
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 5. 复制结果回主机
    copyBatchResultsToHost(batch_data);
}

void GPUMultiPTPriorityQueue::prepareBatchData(const vector<PT>& batch_pts, MultiPTBatch& batch_data) {
    batch_data.total_segments = 0;
    batch_data.total_expected_outputs = 0;
    
    for (size_t i = 0; i < batch_pts.size(); i++) {
        const PT& pt = batch_pts[i];
        
        // 记录PT偏移和类型
        batch_data.pt_offsets.push_back(batch_data.total_segments);
        batch_data.pt_types.push_back(pt.content.size() == 1 ? 0 : 1);
        batch_data.output_offsets.push_back(batch_data.total_expected_outputs);
        
        if (pt.content.size() == 1) {
            // 单segment情况
            segment* a = nullptr;
            if (pt.content[0].type == 1) {
                a = &m.letters[m.FindLetter(pt.content[0])];
            } else if (pt.content[0].type == 2) {
                a = &m.digits[m.FindDigit(pt.content[0])];
            } else if (pt.content[0].type == 3) {
                a = &m.symbols[m.FindSymbol(pt.content[0])];
            }
            
            if (a) {
                int num_values = pt.max_indices[0];
                batch_data.pt_sizes.push_back(num_values);
                
                // 添加所有segment值
                for (int j = 0; j < num_values; j++) {
                    GPUString gpu_str;
                    gpu_str.set(a->ordered_values[j].c_str());
                    batch_data.all_segment_values.push_back(gpu_str);
                }
                
                batch_data.total_segments += num_values;
                batch_data.total_expected_outputs += num_values;
                batch_data.all_base_guesses.push_back("");  // 单segment不需要基础猜测
            }
        } else {
            // 多segment情况
            // 构建基础猜测
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
            batch_data.all_base_guesses.push_back(base_guess);
            
            // 获取最后一个segment
            segment* a = nullptr;
            if (pt.content[pt.content.size() - 1].type == 1) {
                a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
            } else if (pt.content[pt.content.size() - 1].type == 2) {
                a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
            } else if (pt.content[pt.content.size() - 1].type == 3) {
                a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
            }
            
            if (a) {
                int num_values = pt.max_indices[pt.content.size() - 1];
                batch_data.pt_sizes.push_back(num_values);
                
                // 添加最后segment的所有值
                for (int j = 0; j < num_values; j++) {
                    GPUString gpu_str;
                    gpu_str.set(a->ordered_values[j].c_str());
                    batch_data.all_segment_values.push_back(gpu_str);
                }
                
                batch_data.total_segments += num_values;
                batch_data.total_expected_outputs += num_values;
            }
        }
    }
}

void GPUMultiPTPriorityQueue::copyBatchResultsToHost(const MultiPTBatch& batch_data) {
    // 复制结果回主机
    vector<GPUString> h_output_guesses(batch_data.total_expected_outputs);
    CUDA_CHECK(cudaMemcpy(h_output_guesses.data(), d_output_guesses,
                         batch_data.total_expected_outputs * sizeof(GPUString), 
                         cudaMemcpyDeviceToHost));
    
    // 转换为std::string并添加到guesses
    for (int i = 0; i < batch_data.total_expected_outputs; i++) {
        guesses.push_back(string(h_output_guesses[i].c_str()));
        total_guesses++;
    }
}