#ifndef GUESSING_CUDA_MULTI_PT_H
#define GUESSING_CUDA_MULTI_PT_H

#include "PCFG.h"

// 前向声明
struct GPUString;
struct MultiPTBatch;

// GPU多PT并行优化的PriorityQueue类
class GPUMultiPTPriorityQueue : public PriorityQueue {
private:
    // GPU内存指针 - 支持多PT批处理
    GPUString* d_segment_values;     // 所有PT的segment值
    GPUString* d_output_guesses;     // 所有输出结果
    int* d_output_count;             // 总输出计数
    char* d_base_guesses;            // 多个基础猜测字符串
    int* d_pt_offsets;               // 每个PT在数组中的偏移量
    int* d_pt_sizes;                 // 每个PT的segment数量
    int* d_pt_types;                 // PT类型标记（单/多segment）
    
    // 批处理参数
    static const int MAX_BATCH_PTS = 16;           // 最大批处理PT数量
    static const int MAX_TOTAL_SEGMENTS = 5000000; // 总segment值上限
    static const int MAX_TOTAL_OUTPUTS = 5000000;  // 总输出上限
    static const int MAX_BASE_GUESS_LENGTH = 64;   // 基础猜测最大长度
    
    bool gpu_initialized;
    
    // 私有方法
    void initializeMultiPTGPU();
    void cleanupMultiPTGPU();
    void generateMultiPTBatchGPU(const vector<PT>& batch_pts);
    void prepareBatchData(const vector<PT>& batch_pts, MultiPTBatch& batch_data);
    void copyBatchResultsToHost(const MultiPTBatch& batch_data);
    
public:
    GPUMultiPTPriorityQueue();
    ~GPUMultiPTPriorityQueue();
    
    // GPU优化的多PT批处理函数
    void PopNext(int batch_pt_num = 4);
};

// 多PT批处理数据结构
struct MultiPTBatch {
    vector<GPUString> all_segment_values;     // 所有PT的segment值
    vector<string> all_base_guesses;          // 所有PT的基础猜测
    vector<int> pt_offsets;                   // 每个PT在segment数组中的起始位置
    vector<int> pt_sizes;                     // 每个PT的segment数量
    vector<int> pt_types;                     // PT类型（0=单segment, 1=多segment）
    vector<int> output_offsets;               // 每个PT的输出起始位置
    int total_segments;                       // 总segment数量
    int total_expected_outputs;               // 预期总输出数量
};

#endif // GUESSING_CUDA_MULTI_PT_H