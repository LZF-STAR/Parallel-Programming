#ifndef GUESSING_CUDA_H
#define GUESSING_CUDA_H
#include "PCFG.h"
// 前向声明CUDA相关结构
struct GPUString;

// GPU优化的PriorityQueue类声明
class GPUPriorityQueue : public PriorityQueue {
private:
    // GPU内存指针
    GPUString* d_segment_values;
    GPUString* d_output_guesses;
    int* d_output_count;
    char* d_base_guess;
    
    // 最大分配大小
    static const int MAX_SEGMENT_VALUES = 1000000;
    static const int MAX_OUTPUT_GUESSES = 1000000;
    
    bool gpu_initialized;
    
    // 私有方法声明
    void initializeGPU();
    void cleanupGPU();
    void generateSingleSegmentGPU(PT pt);
    void generateMultiSegmentGPU(PT pt);
    segment* getSegmentPointer(const segment& seg);
    std::string buildBaseGuess(const PT& pt);
    void copyResultsToHost(int num_values);
    void processBatches(const PT& pt, int total_values);
    void generateSingleSegmentBatch(const PT& pt, int start_idx);
    void generateMultiSegmentBatch(const PT& pt, int start_idx);
    
public:
    GPUPriorityQueue();
    ~GPUPriorityQueue();
    
    // GPU优化的Generate函数
    void GenerateGPU(PT pt);
    
    // 重写PopNext函数
    void PopNext();
};

#endif // GUESSING_CUDA_H