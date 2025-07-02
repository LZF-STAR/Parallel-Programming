#include <string>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

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

// GPU常量定义
#define MAX_STRING_LENGTH 64
#define BLOCK_SIZE 256
#define MIN_GPU_SIZE 1000000    // 只有足够大的数据才用GPU
#define MAX_BATCH_PTS 32      // 一次最多处理的PT数量

class segment
{
public:
    int type; // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    int length; // 长度，例如S6的长度就是6
    segment(int type, int length)
    {
        this->type = type;
        this->length = length;
    };

    // 打印相关信息
    void PrintSeg();

    // 按照概率降序排列的value。例如，123是D3的一个具体value，其概率在D3的所有value中排名第三，那么其位置就是ordered_values[2]
    vector<string> ordered_values;

    // 按照概率降序排列的频数（概率）
    vector<int> ordered_freqs;

    // total_freq作为分母，用于计算每个value的概率
    int total_freq = 0;

    // 未排序的value，其中int就是对应的id
    unordered_map<string, int> values;

    // 根据id，在freqs中查找/修改一个value的频数
    unordered_map<int, int> freqs;

    void insert(string value);
    void order();
    void PrintValues();
};

class PT
{
public:
    // 例如，L6D1的content大小为2，content[0]为L6，content[1]为D1
    vector<segment> content;

    // pivot值，参见PCFG的原理
    int pivot = 0;
    void insert(segment seg);
    void PrintPT();

    // 导出新的PT
    vector<PT> NewPTs();

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的下标
    vector<int> curr_indices;

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的最大下标（即最大可以是max_indices[x]-1）
    vector<int> max_indices;
    
    float preterm_prob;
    float prob;
};

class model
{
public:
    // 对于PT/LDS而言，序号是递增的
    // 训练时每遇到一个新的PT/LDS，就获取一个新的序号，并且当前序号递增1
    int preterm_id = -1;
    int letters_id = -1;
    int digits_id = -1;
    int symbols_id = -1;
    int GetNextPretermID()
    {
        preterm_id++;
        return preterm_id;
    };
    int GetNextLettersID()
    {
        letters_id++;
        return letters_id;
    };
    int GetNextDigitsID()
    {
        digits_id++;
        return digits_id;
    };
    int GetNextSymbolsID()
    {
        symbols_id++;
        return symbols_id;
    };

    // C++上机和数据结构实验中，一般不允许使用stl
    // 这就导致大家对stl不甚熟悉。现在是时候体会stl的便捷之处了
    // unordered_map: 无序映射
    int total_preterm = 0;
    vector<PT> preterminals;
    int FindPT(PT pt);

    vector<segment> letters;
    vector<segment> digits;
    vector<segment> symbols;
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);

    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq;
    unordered_map<int, int> digits_freq;
    unordered_map<int, int> symbols_freq;

    vector<PT> ordered_pts;

    // 给定一个训练集，对模型进行训练
    void train(string train_path);

    // 对已经训练的模型进行保存
    void store(string store_path);

    // 从现有的模型文件中加载模型
    void load(string load_path);

    // 对一个给定的口令进行切分
    void parse(string pw);

    void order();

    // 打印模型
    void print();
};

// GPU批处理任务结构
struct GPUBatchTask {
    int pt_id;                    // PT在批次中的ID
    int task_type;               // 0: single segment, 1: multi segment
    char base_guess[MAX_STRING_LENGTH];  // 基础字符串（多segment时使用）
    int base_length;             // 基础字符串长度
    int segment_start_idx;       // segment在全局数据中的起始位置
    int segment_count;          // 该PT需要处理的segment值数量
    int output_start_idx;       // 输出在全局数组中的起始位置
};

// GPU内存管理器
class GPUMemoryManager {
private:
    // GPU内存
    char* d_segment_data;        // 所有segment字符串数据
    int* d_segment_offsets;      // segment偏移数组
    int* d_segment_lengths;      // segment长度数组
    GPUBatchTask* d_batch_tasks; // 批处理任务数组
    char* d_output_data;         // 输出字符串数据
    int* d_output_offsets;       // 输出偏移数组
    
    // 主机内存
    vector<char> h_segment_buffer;
    vector<int> h_segment_offsets;
    vector<int> h_segment_lengths;
    vector<GPUBatchTask> h_batch_tasks;
    vector<char> h_output_buffer;
    vector<int> h_output_offsets;
    
    static const size_t MAX_BUFFER_SIZE = 512 * 1024 * 1024; // 512MB
    static const int MAX_SEGMENTS = 2000000;
    
    bool initialized;
    
public:
    GPUMemoryManager();
    ~GPUMemoryManager();
    
    void initializeMemory();
    void cleanup();
    
    // 批量处理多个PT
    bool processBatchPTs(const vector<PT>& batch_pts, 
                        model& m,
                        vector<string>& results);
    
private:
    void prepareBatchData(const vector<PT>& batch_pts, model& m);
    void executeBatchKernel();
    void collectResults(vector<string>& results);
};

// 优先队列，用于按照概率降序生成口令猜测
// 实际上，这个class负责队列维护、口令生成、结果存储的全部过程
class PriorityQueue
{
public:
    // 用vector实现的priority queue
    vector<PT> priority;

    // 模型作为成员，辅助猜测生成
    model m;

    // GPU内存管理器
    GPUMemoryManager* gpu_manager;

    // 计算一个pt的概率
    void CalProb(PT &pt);

    // 优先队列的初始化
    void init();

    // 对优先队列的一个PT，生成所有guesses
    void Generate(PT pt);

    // 批量处理多个PT (GPU版本)
    void PopNext(int batch_pt_num = 8);

    int total_guesses = 0;
    vector<string> guesses;
    
    // 构造函数和析构函数
    PriorityQueue();
    ~PriorityQueue();
};

// GPU内核函数声明
extern "C" {
    void launch_batch_generation_kernel(
        char* d_segment_data,
        int* d_segment_offsets,
        int* d_segment_lengths,
        GPUBatchTask* d_batch_tasks,
        int num_tasks,
        char* d_output_data,
        int* d_output_offsets,
        cudaStream_t stream = 0
    );
}