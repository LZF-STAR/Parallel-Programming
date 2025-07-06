#ifndef UNIFIED_FRAMEWORK_H
#define UNIFIED_FRAMEWORK_H

#include <iostream>
#include <vector>
#include <queue>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <cmath>    // 添加这个，用于 std::log
#include <thread>   // 添加这个，用于 std::thread
#include <list>     // 添加这个，用于 std::list
#include "PCFG.h"

// 前向声明
class TaskScheduler;
class UnifiedMemoryPool;
struct TaskFeatures;

// ========== 层间标准接口 ==========
template<typename Input, typename Output>
class IComputeLayer {
public:
    virtual ~IComputeLayer() = default;
    virtual bool canProcess(const Input& in) = 0;
    virtual void process(const Input& in, Output& out) = 0;
    virtual double getPerformanceMetrics() = 0;
    virtual void updateMetrics(double time, size_t data_size) = 0;
};

// ========== 性能监控 ==========
class PerformanceMonitor {
    private:
        struct LayerMetrics {
            std::atomic<double> total_time{0};
            std::atomic<size_t> total_calls{0};
            std::atomic<size_t> total_data{0};
            std::atomic<double> utilization{0};
        };
        
        std::unordered_map<std::string, LayerMetrics> metrics;
        mutable std::mutex metrics_mutex;  // 添加 mutable
        
    public:
        void recordMetric(const std::string& layer, double time, size_t data_size);
        void updateUtilization(const std::string& layer, double util);
        void printReport();
        double getLayerEfficiency(const std::string& layer) const;  // 添加 const
    };

// ========== 统一内存池管理 ==========
class UnifiedMemoryPool {
public:
    struct MemoryRegion {
        void* ptr;
        size_t size;
        size_t offset;
        std::atomic<int> ref_count{1};
        
        MemoryRegion(void* p, size_t s) : ptr(p), size(s), offset(0) {}
    };
    
    // 在 MemoryView 类中添加默认构造函数
class MemoryView {
    private:
        std::shared_ptr<MemoryRegion> region;
        size_t view_offset;
        size_t view_size;
        
    public:
        MemoryView() : view_offset(0), view_size(0) {}  // 添加默认构造函数
        
        MemoryView(std::shared_ptr<MemoryRegion> r, size_t off, size_t sz)
            : region(r), view_offset(off), view_size(sz) {}
        
        template<typename T>
        T* as() { 
            if (!region) return nullptr;  // 添加空指针检查
            return reinterpret_cast<T*>(static_cast<char*>(region->ptr) + view_offset); 
        }
        
        MemoryView slice(size_t start, size_t len);
        size_t size() const { return view_size; }
    };

private:
    // 三级内存池
    static constexpr size_t L1_SIZE = 64 * 1024;      // 64KB per thread
    static constexpr size_t L2_SIZE = 16 * 1024 * 1024; // 16MB per NUMA node
    static constexpr size_t L3_SIZE = 1024 * 1024 * 1024; // 1GB global
    
    // L1: 线程本地池
    thread_local static char* tls_pool;
    thread_local static size_t tls_offset;
    
    // L2: NUMA节点池
    struct NUMAPool {
        std::unique_ptr<char[]> memory;
        std::atomic<size_t> offset{0};
        std::mutex alloc_mutex;
        
        // 添加移动构造函数和移动赋值运算符
        NUMAPool() = default;
        NUMAPool(NUMAPool&& other) noexcept 
            : memory(std::move(other.memory)), 
            offset(other.offset.load()) {}
        
        NUMAPool& operator=(NUMAPool&& other) noexcept {
            if (this != &other) {
                memory = std::move(other.memory);
                offset = other.offset.load();
            }
            return *this;
        }
    
    // 删除拷贝构造和拷贝赋值
    NUMAPool(const NUMAPool&) = delete;
    NUMAPool& operator=(const NUMAPool&) = delete;
};
    std::vector<NUMAPool> numa_pools;
    
    // L3: 全局池
    std::unique_ptr<char[]> global_pool;
    std::atomic<size_t> global_offset{0};
    std::mutex global_mutex;
    
    // 内存区域管理
    std::vector<std::shared_ptr<MemoryRegion>> regions;
    std::mutex regions_mutex;

public:
    UnifiedMemoryPool();
    ~UnifiedMemoryPool();
    
    // 分配接口
    MemoryView allocate(size_t size, int numa_hint = -1);
    void* allocateRaw(size_t size, size_t alignment = 16);
    
    // 零拷贝数据传递
    MemoryView createView(void* ptr, size_t size);
    void releaseView(MemoryView& view);
    
    // 统计信息
    size_t getTotalAllocated() const;
    double getFragmentation() const;
    
private:
    void* allocateFromL1(size_t size);
    void* allocateFromL2(size_t size, int numa_node);
    void* allocateFromL3(size_t size);
};

// ========== 任务特征分析 ==========
struct TaskFeatures {
    size_t segment_count;        // segment数量
    size_t max_values;          // 最大可能值数
    size_t total_combinations;   // 总组合数
    int task_type;              // 0: single, 1: multi
    double estimated_time;       // 预估执行时间
    bool gpu_suitable;          // 是否适合GPU
    int memory_pattern;         // 0: sequential, 1: random
    
    // 计算任务复杂度评分
    double getComplexityScore() const {
        return std::log(total_combinations) * segment_count;
    }
};

// ========== 智能任务调度器 ==========
class TaskScheduler {
public:
    enum LayerType {
        LAYER_SIMD = 0,
        LAYER_THREAD = 1,
        LAYER_GPU = 2,
        LAYER_MPI = 3
    };
    
    struct SchedulingDecision {
        LayerType primary_layer;
        LayerType secondary_layer;
        size_t batch_size;
        int priority;
    };

private:
    // 任务队列
    struct TaskWrapper {
        PT task;
        TaskFeatures features;
        int priority;
        std::chrono::time_point<std::chrono::steady_clock> enqueue_time;
        
        bool operator<(const TaskWrapper& other) const {
            return priority < other.priority;
        }
    };
    
    std::priority_queue<TaskWrapper> task_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    
    // 性能模型参数
    struct PerformanceModel {
        double cpu_throughput = 1000000;  // passwords/sec
        double gpu_throughput = 10000000; // passwords/sec
        double simd_speedup = 4.0;
        double thread_efficiency = 0.8;
        double gpu_transfer_bandwidth = 10e9; // bytes/sec
        double gpu_latency = 0.001; // seconds
    };
    PerformanceModel perf_model;
    
    // 自适应阈值
    std::atomic<size_t> gpu_threshold{10000};
    std::atomic<size_t> thread_threshold{100};
    std::atomic<double> load_balance_factor{0.8};
    
    // 负载统计
    std::atomic<int> active_cpu_tasks{0};
    std::atomic<int> active_gpu_tasks{0};
    std::atomic<double> cpu_utilization{0};
    std::atomic<double> gpu_utilization{0};
    
    // 性能监控
    PerformanceMonitor monitor;

public:
    TaskScheduler();
    ~TaskScheduler();
    
    // 任务提交与获取
    void submitTask(const PT& task);
    bool getNextTask(PT& task, LayerType preferred_layer);
    
    // 任务特征分析
    TaskFeatures analyzeTask(const PT& task);
    
    // 调度决策
    SchedulingDecision makeDecision(const TaskFeatures& features);
    
    // 动态调整
    void updateThresholds(const PerformanceMonitor& monitor);
    void reportTaskCompletion(LayerType layer, double time, size_t passwords);
    
    // 负载均衡
    bool shouldMigrateTask(LayerType from, LayerType to);
    void migrateTask(const PT& task, LayerType from, LayerType to);
    
private:
    // 性能预测
    double predictExecutionTime(const TaskFeatures& features, LayerType layer);
    double calculateTransferCost(size_t data_size, LayerType from, LayerType to);
    
    // 调度算法
    LayerType selectOptimalLayer(const TaskFeatures& features);
    size_t calculateOptimalBatchSize(const TaskFeatures& features, LayerType layer);
};

// ========== 数据批处理接口 ==========
template<typename T>
class DataBatch {
public:
    virtual ~DataBatch() = default;
    virtual size_t size() const = 0;
    virtual T* data() = 0;
    virtual const T* data() const = 0;
    virtual void resize(size_t new_size) = 0;
    virtual bool isGPUAccessible() const = 0;
    virtual UnifiedMemoryPool::MemoryView getMemoryView() = 0;
};

// PT任务批次
class PTBatch : public DataBatch<PT> {
private:
    std::vector<PT> pts;
    UnifiedMemoryPool::MemoryView memory_view;
    bool gpu_ready = false;
    
public:
    PTBatch() = default;
    explicit PTBatch(const std::vector<PT>& tasks) : pts(tasks) {}
    
    size_t size() const override { return pts.size(); }
    PT* data() override { return pts.data(); }
    const PT* data() const override { return pts.data(); }
    void resize(size_t new_size) override { pts.resize(new_size); }
    bool isGPUAccessible() const override { return gpu_ready; }
    UnifiedMemoryPool::MemoryView getMemoryView() override { return memory_view; }
    
    void prepareForGPU();
    void addTask(const PT& task) { pts.push_back(task); }
};

// 字符串批次（零拷贝）
class StringBatch : public DataBatch<std::string_view> {
private:
    std::vector<std::string_view> views;
    UnifiedMemoryPool::MemoryView memory_view;
    
public:
    StringBatch() = default;
    
    size_t size() const override { return views.size(); }
    std::string_view* data() override { return views.data(); }
    const std::string_view* data() const override { return views.data(); }
    void resize(size_t new_size) override { views.resize(new_size); }
    bool isGPUAccessible() const override { return false; }
    UnifiedMemoryPool::MemoryView getMemoryView() override { return memory_view; }
    
    void addString(const char* str, size_t len) {
        views.emplace_back(str, len);
    }
};

// ========== 全局框架管理器 ==========
class UnifiedFramework {
private:
    static UnifiedFramework* instance;
    
    std::unique_ptr<UnifiedMemoryPool> memory_pool;
    std::unique_ptr<TaskScheduler> scheduler;
    PerformanceMonitor global_monitor;
    
    // 层实例管理
    std::unordered_map<std::string, std::unique_ptr<IComputeLayer<PT, StringBatch>>> layers;
    
    // 配置参数
    struct Config {
        int num_threads = 8;
        bool use_gpu = true;
        bool use_mpi = false;
        size_t memory_pool_size = 1024 * 1024 * 1024; // 1GB
    } config;
    
    UnifiedFramework() = default;

public:
    static UnifiedFramework* getInstance() {
        if (!instance) {
            instance = new UnifiedFramework();
        }
        return instance;
    }
    
    // 初始化
    void initialize(const Config& cfg);
    void shutdown();
    
    // 获取组件
    UnifiedMemoryPool* getMemoryPool() { return memory_pool.get(); }
    TaskScheduler* getScheduler() { return scheduler.get(); }
    PerformanceMonitor* getMonitor() { return &global_monitor; }
    
    // 层管理
    void registerLayer(const std::string& name, 
                      std::unique_ptr<IComputeLayer<PT, StringBatch>> layer);
    IComputeLayer<PT, StringBatch>* getLayer(const std::string& name);
    
    // 运行框架
    void run(PriorityQueue& queue, size_t max_passwords);
    
private:
    void processTaskBatch(PTBatch& batch);
    void collectResults(StringBatch& results);
};

// ========== 辅助函数 ==========
namespace FrameworkUtils {
    // NUMA相关
    int getCurrentNUMANode();
    void bindToNUMANode(int node);
    
    // 内存对齐
    size_t getOptimalAlignment(TaskScheduler::LayerType layer);
    void* alignPointer(void* ptr, size_t alignment);
    
    // 性能预测
    double estimateTaskSize(const PT& task);
    int detectMemoryPattern(const PT& task);
}

#endif // UNIFIED_FRAMEWORK_H