#include "PCFG_PT.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

using namespace std;

// CUDA内核：批量生成密码
__global__ void batch_generation_kernel(
    char* segment_data,          // 所有segment字符串数据
    int* segment_offsets,        // segment偏移数组
    int* segment_lengths,        // segment长度数组
    GPUBatchTask* batch_tasks,   // 批处理任务数组
    int num_tasks,              // 任务数量
    char* output_data,          // 输出字符串数据
    int* output_offsets         // 输出偏移数组
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // 每个线程处理多个任务，确保所有任务都被处理
    for (int task_idx = 0; task_idx < num_tasks; task_idx++) {
        GPUBatchTask& task = batch_tasks[task_idx];
        
        // 计算当前线程在这个任务中应该处理的segment索引
        int segments_per_thread = (task.segment_count + total_threads - 1) / total_threads;
        int start_seg = global_idx * segments_per_thread;
        int end_seg = min(start_seg + segments_per_thread, task.segment_count);
        
        for (int seg_idx = start_seg; seg_idx < end_seg; seg_idx++) {
            int global_seg_idx = task.segment_start_idx + seg_idx;
            int output_idx = task.output_start_idx + seg_idx;
            
            // 获取segment数据
            int seg_offset = segment_offsets[global_seg_idx];
            int seg_length = segment_lengths[global_seg_idx];
            int out_offset = output_offsets[output_idx];
            
            if (task.task_type == 0) {
                // 单segment情况：直接复制
                for (int i = 0; i < seg_length; i++) {
                    output_data[out_offset + i] = segment_data[seg_offset + i];
                }
                output_data[out_offset + seg_length] = '\0';
            } else {
                // 多segment情况：复制基础字符串 + segment
                // 复制基础字符串
                for (int i = 0; i < task.base_length; i++) {
                    output_data[out_offset + i] = task.base_guess[i];
                }
                // 追加segment字符串
                for (int i = 0; i < seg_length; i++) {
                    output_data[out_offset + task.base_length + i] = segment_data[seg_offset + i];
                }
                output_data[out_offset + task.base_length + seg_length] = '\0';
            }
        }
    }
}

// C包装函数
extern "C" void launch_batch_generation_kernel(
    char* d_segment_data,
    int* d_segment_offsets,
    int* d_segment_lengths,
    GPUBatchTask* d_batch_tasks,
    int num_tasks,
    char* d_output_data,
    int* d_output_offsets,
    cudaStream_t stream
) {
    int num_blocks = 64;  // 使用固定的block数量
    int block_size = BLOCK_SIZE;
    
    batch_generation_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_segment_data, d_segment_offsets, d_segment_lengths,
        d_batch_tasks, num_tasks, d_output_data, d_output_offsets
    );
}

// GPU内存管理器实现
GPUMemoryManager::GPUMemoryManager() : initialized(false) {
    initializeMemory();
}

GPUMemoryManager::~GPUMemoryManager() {
    cleanup();
}

void GPUMemoryManager::initializeMemory() {
    if (initialized) return;
    
    // 分配GPU内存
    CUDA_CHECK(cudaMalloc(&d_segment_data, MAX_BUFFER_SIZE));
    CUDA_CHECK(cudaMalloc(&d_segment_offsets, MAX_SEGMENTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_segment_lengths, MAX_SEGMENTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_batch_tasks, MAX_BATCH_PTS * sizeof(GPUBatchTask)));
    CUDA_CHECK(cudaMalloc(&d_output_data, MAX_BUFFER_SIZE));
    CUDA_CHECK(cudaMalloc(&d_output_offsets, MAX_SEGMENTS * sizeof(int)));
    
    // 预分配主机内存
    h_segment_buffer.reserve(MAX_BUFFER_SIZE);
    h_segment_offsets.reserve(MAX_SEGMENTS);
    h_segment_lengths.reserve(MAX_SEGMENTS);
    h_batch_tasks.reserve(MAX_BATCH_PTS);
    h_output_buffer.reserve(MAX_BUFFER_SIZE);
    h_output_offsets.reserve(MAX_SEGMENTS);
    
    initialized = true;
    printf("GPU Memory Manager initialized: 512MB allocated\n");
}

void GPUMemoryManager::cleanup() {
    if (initialized) {
        cudaFree(d_segment_data);
        cudaFree(d_segment_offsets);
        cudaFree(d_segment_lengths);
        cudaFree(d_batch_tasks);
        cudaFree(d_output_data);
        cudaFree(d_output_offsets);
        initialized = false;
    }
}

bool GPUMemoryManager::processBatchPTs(const vector<PT>& batch_pts, 
                                      model& m,
                                      vector<string>& results) {
    if (!initialized || batch_pts.empty()) {
        return false;
    }
    
    // 估算总工作量
    int total_segments = 0;
    for (const auto& pt : batch_pts) {
        if (pt.content.size() == 1) {
            total_segments += pt.max_indices[0];
        } else {
            total_segments += pt.max_indices[pt.content.size() - 1];
        }
    }
    
    // 如果工作量太小，使用CPU
    if (total_segments < MIN_GPU_SIZE) {
        return false;
    }
    
    try {
        // 准备批处理数据
        prepareBatchData(batch_pts, m);
        
        // 执行GPU内核
        executeBatchKernel();
        
        // 收集结果
        collectResults(results);
        
        return true;
    } catch (...) {
        return false; // 出错时返回false，让CPU处理
    }
}

void GPUMemoryManager::prepareBatchData(const vector<PT>& batch_pts, model& m) {
    // 清空缓冲区
    h_segment_buffer.clear();
    h_segment_offsets.clear();
    h_segment_lengths.clear();
    h_batch_tasks.clear();
    h_output_offsets.clear();
    
    int current_segment_offset = 0;
    int current_output_offset = 0;
    int global_segment_idx = 0;
    
    for (int pt_idx = 0; pt_idx < batch_pts.size(); pt_idx++) {
        const PT& pt = batch_pts[pt_idx];
        GPUBatchTask task;
        task.pt_id = pt_idx;
        task.segment_start_idx = global_segment_idx;
        task.output_start_idx = h_output_offsets.size();
        
        segment* target_segment = nullptr;
        int segment_count = 0;
        
        if (pt.content.size() == 1) {
            // 单segment情况
            task.task_type = 0;
            task.base_length = 0;
            
            // 获取segment数据
            if (pt.content[0].type == 1) {
                target_segment = &m.letters[m.FindLetter(pt.content[0])];
            } else if (pt.content[0].type == 2) {
                target_segment = &m.digits[m.FindDigit(pt.content[0])];
            } else {
                target_segment = &m.symbols[m.FindSymbol(pt.content[0])];
            }
            segment_count = pt.max_indices[0];
        } else {
            // 多segment情况
            task.task_type = 1;
            
            // 构建基础字符串
            string base_guess;
            int seg_idx = 0;
            for (int idx : pt.curr_indices) {
                if (pt.content[seg_idx].type == 1) {
                    base_guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
                } else if (pt.content[seg_idx].type == 2) {
                    base_guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
                } else if (pt.content[seg_idx].type == 3) {
                    base_guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
                }
                seg_idx++;
                if (seg_idx == pt.content.size() - 1) break;
            }
            
            strncpy(task.base_guess, base_guess.c_str(), MAX_STRING_LENGTH - 1);
            task.base_guess[MAX_STRING_LENGTH - 1] = '\0';
            task.base_length = base_guess.length();
            
            // 获取最后一个segment
            if (pt.content[pt.content.size() - 1].type == 1) {
                target_segment = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
            } else if (pt.content[pt.content.size() - 1].type == 2) {
                target_segment = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
            } else {
                target_segment = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
            }
            segment_count = pt.max_indices[pt.content.size() - 1];
        }
        
        task.segment_count = segment_count;
        
        // 复制segment数据
        for (int i = 0; i < segment_count; i++) {
            const string& value = target_segment->ordered_values[i];
            h_segment_offsets.push_back(current_segment_offset);
            h_segment_lengths.push_back(value.length());
            
            // 复制字符串数据
            for (char c : value) {
                h_segment_buffer.push_back(c);
            }
            current_segment_offset += value.length();
            
            // 计算输出偏移
            int output_length = (task.task_type == 0) ? 
                value.length() + 1 : task.base_length + value.length() + 1;
            h_output_offsets.push_back(current_output_offset);
            current_output_offset += output_length;
        }
        
        global_segment_idx += segment_count;
        h_batch_tasks.push_back(task);
    }
    
    // 检查缓冲区大小
    if (h_segment_buffer.size() > MAX_BUFFER_SIZE || 
        current_output_offset > MAX_BUFFER_SIZE ||
        global_segment_idx > MAX_SEGMENTS) {
        throw runtime_error("Buffer overflow");
    }
}

void GPUMemoryManager::executeBatchKernel() {
    // 传输数据到GPU
    CUDA_CHECK(cudaMemcpy(d_segment_data, h_segment_buffer.data(), 
                         h_segment_buffer.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_segment_offsets, h_segment_offsets.data(), 
                         h_segment_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_segment_lengths, h_segment_lengths.data(), 
                         h_segment_lengths.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_batch_tasks, h_batch_tasks.data(), 
                         h_batch_tasks.size() * sizeof(GPUBatchTask), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_offsets, h_output_offsets.data(), 
                         h_output_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    // 启动内核
    launch_batch_generation_kernel(
        d_segment_data, d_segment_offsets, d_segment_lengths,
        d_batch_tasks, h_batch_tasks.size(),
        d_output_data, d_output_offsets
    );
    
    // 检查内核错误
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void GPUMemoryManager::collectResults(vector<string>& results) {
    // 计算总输出大小
    int total_output_size = 0;
    if (!h_output_offsets.empty()) {
        total_output_size = h_output_offsets.back();
        // 添加最后一个字符串的大小
        for (const auto& task : h_batch_tasks) {
            if (task.output_start_idx + task.segment_count - 1 == h_output_offsets.size() - 1) {
                int last_seg_idx = task.segment_start_idx + task.segment_count - 1;
                int last_seg_len = h_segment_lengths[last_seg_idx];
                total_output_size += (task.task_type == 0) ? 
                    last_seg_len + 1 : task.base_length + last_seg_len + 1;
                break;
            }
        }
    }
    
    // 复制结果数据
    h_output_buffer.resize(total_output_size);
    CUDA_CHECK(cudaMemcpy(h_output_buffer.data(), d_output_data,
                         total_output_size, cudaMemcpyDeviceToHost));
    
    // 解析结果字符串
    results.clear();
    results.reserve(h_output_offsets.size());
    
    for (size_t i = 0; i < h_output_offsets.size(); i++) {
        char* str_start = h_output_buffer.data() + h_output_offsets[i];
        results.emplace_back(str_start);
    }
}

// PriorityQueue实现
PriorityQueue::PriorityQueue() {
    gpu_manager = new GPUMemoryManager();
}

PriorityQueue::~PriorityQueue() {
    delete gpu_manager;
}

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext(int batch_pt_num)
{
    if (priority.empty()) return;
    
    // 确定实际批次大小
    int actual_batch = min(batch_pt_num, static_cast<int>(priority.size()));
    vector<PT> batch_pts(priority.begin(), priority.begin() + actual_batch);
    
    // 尝试GPU批量处理
    vector<string> gpu_results;
    bool gpu_success = gpu_manager->processBatchPTs(batch_pts, m, gpu_results);
    
    if (gpu_success) {
        // GPU处理成功
        for (const auto& result : gpu_results) {
            guesses.emplace_back(result);
        }
        total_guesses += gpu_results.size();
    } else {
        // GPU处理失败，使用CPU逐个处理
        for (const auto& pt : batch_pts) {
            Generate(pt);
        }
    }
    
    // 生成新PT并放回队列
    vector<PT> new_pts;
    for (auto &pt : batch_pts) {
        vector<PT> pts = pt.NewPTs();
        for (PT &npt : pts) {
            CalProb(npt);
            new_pts.emplace_back(move(npt));
        }
    }
    // 从队列中删除已处理的PT
    priority.erase(priority.begin(), priority.begin() + actual_batch);
    
    // 将新PT按概率插入到优先队列中
    for (PT &pt : new_pts) {
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

void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[0])];
        } else if (pt.content[0].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[0])];
        } else {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        for (int i = 0; i < pt.max_indices[0]; i += 1) {
            guesses.emplace_back(a->ordered_values[i]);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
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

        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        } else if (pt.content[pt.content.size() - 1].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        } else {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }

        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1) {
            string temp = guess + a->ordered_values[i];
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;

    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        int init_pivot = pivot;

        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            curr_indices[i] += 1;

            if (curr_indices[i] < max_indices[i])
            {
                pivot = i;
                res.emplace_back(*this);
            }

            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}