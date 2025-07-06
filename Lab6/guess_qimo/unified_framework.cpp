#include "unified_framework.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numa.h>
#include <sched.h>
#include <thread> 
#include <sys/mman.h>

using namespace std;
using namespace chrono;

// 静态成员初始化
UnifiedFramework* UnifiedFramework::instance = nullptr;
thread_local char* UnifiedMemoryPool::tls_pool = nullptr;
thread_local size_t UnifiedMemoryPool::tls_offset = 0;

void PerformanceMonitor::recordMetric(const string& layer, double time, size_t data_size) {
    lock_guard<mutex> lock(metrics_mutex);
    auto& m = metrics[layer];
    
    // 修复 atomic<double> 的操作
    double current = m.total_time.load();
    m.total_time.store(current + time);
    
    m.total_calls++;
    m.total_data += data_size;
}

void PerformanceMonitor::updateUtilization(const string& layer, double util) {
    lock_guard<mutex> lock(metrics_mutex);
    metrics[layer].utilization = util;
}

void PerformanceMonitor::printReport() {
    lock_guard<mutex> lock(metrics_mutex);
    cout << "\n=== Performance Report ===" << endl;
    for (const auto& [layer, m] : metrics) {
        double avg_time = m.total_calls > 0 ? m.total_time / m.total_calls : 0;
        double throughput = m.total_time > 0 ? m.total_data / m.total_time : 0;
        cout << layer << ":" << endl;
        cout << "  Total Time: " << m.total_time << " s" << endl;
        cout << "  Total Calls: " << m.total_calls << endl;
        cout << "  Avg Time/Call: " << avg_time * 1000 << " ms" << endl;
        cout << "  Throughput: " << throughput / 1e6 << " MB/s" << endl;
        cout << "  Utilization: " << m.utilization * 100 << "%" << endl;
    }
}

double PerformanceMonitor::getLayerEfficiency(const string& layer) {
    lock_guard<mutex> lock(metrics_mutex);
    auto it = metrics.find(layer);
    if (it != metrics.end()) {
        return it->second.utilization;
    }
    return 0.0;
}

// ========== 统一内存池实现 ==========
UnifiedMemoryPool::UnifiedMemoryPool() {
    // 初始化NUMA池
    int num_nodes = numa_max_node() + 1;
    numa_pools.resize(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        numa_pools[i].memory = make_unique<char[]>(L2_SIZE);
    }
    
    // 初始化全局池
    global_pool = make_unique<char[]>(L3_SIZE);
    
    // 初始化线程本地池
    if (!tls_pool) {
        tls_pool = new char[L1_SIZE];
        tls_offset = 0;
    }
}

UnifiedMemoryPool::~UnifiedMemoryPool() {
    // 清理将由智能指针自动处理
}

UnifiedMemoryPool::MemoryView UnifiedMemoryPool::allocate(size_t size, int numa_hint) {
    void* ptr = nullptr;
    
    // 根据大小选择分配策略
    if (size < L1_SIZE / 4) {
        ptr = allocateFromL1(size);
    } else if (size < L2_SIZE / 4) {
        int node = numa_hint >= 0 ? numa_hint : numa_node_of_cpu(sched_getcpu());
        ptr = allocateFromL2(size, node);
    } else {
        ptr = allocateFromL3(size);
    }
    
    // 创建内存区域
    auto region = make_shared<MemoryRegion>(ptr, size);
    {
        lock_guard<mutex> lock(regions_mutex);
        regions.push_back(region);
    }
    
    return MemoryView(region, 0, size);
}

void* UnifiedMemoryPool::allocateFromL1(size_t size) {
    // 对齐到16字节
    size = (size + 15) & ~15;
    
    if (tls_offset + size > L1_SIZE) {
        // L1池已满，回退到L2
        return allocateFromL2(size, numa_node_of_cpu(sched_getcpu()));
    }
    
    void* ptr = tls_pool + tls_offset;
    tls_offset += size;
    return ptr;
}

void* UnifiedMemoryPool::allocateFromL2(size_t size, int numa_node) {
    if (numa_node < 0 || numa_node >= numa_pools.size()) {
        numa_node = 0;
    }
    
    auto& pool = numa_pools[numa_node];
    
    // 使用CAS操作进行无锁分配
    size_t old_offset, new_offset;
    do {
        old_offset = pool.offset.load();
        new_offset = old_offset + size;
        if (new_offset > L2_SIZE) {
            // L2池已满，回退到L3
            return allocateFromL3(size);
        }
    } while (!pool.offset.compare_exchange_weak(old_offset, new_offset));
    
    return pool.memory.get() + old_offset;
}

void* UnifiedMemoryPool::allocateFromL3(size_t size) {
    lock_guard<mutex> lock(global_mutex);
    
    size_t offset = global_offset.fetch_add(size);
    if (offset + size > L3_SIZE) {
        throw runtime_error("Global memory pool exhausted");
    }
    
    return global_pool.get() + offset;
}

UnifiedMemoryPool::MemoryView UnifiedMemoryPool::MemoryView::slice(size_t start, size_t len) {
    if (start + len > view_size) {
        throw out_of_range("Slice exceeds view bounds");
    }
    return MemoryView(region, view_offset + start, len);
}

// ========== 任务调度器实现 ==========
TaskScheduler::TaskScheduler() {
    // 初始化性能模型参数（可以从配置文件读取）
}

TaskScheduler::~TaskScheduler() {
    // 清理资源
}

void TaskScheduler::submitTask(const PT& task) {
    TaskWrapper wrapper;
    wrapper.task = task;
    wrapper.features = analyzeTask(task);
    wrapper.priority = wrapper.features.getComplexityScore();
    wrapper.enqueue_time = steady_clock::now();
    
    {
        lock_guard<mutex> lock(queue_mutex);
        task_queue.push(wrapper);
    }
    queue_cv.notify_one();
}

bool TaskScheduler::getNextTask(PT& task, LayerType preferred_layer) {
    unique_lock<mutex> lock(queue_mutex);
    
    if (task_queue.empty()) {
        return false;
    }
    
    // 获取最高优先级任务
    TaskWrapper wrapper = task_queue.top();
    task_queue.pop();
    
    // 检查是否适合请求的层
    auto decision = makeDecision(wrapper.features);
    if (decision.primary_layer != preferred_layer && 
        decision.secondary_layer != preferred_layer) {
        // 任务不适合该层，重新入队
        task_queue.push(wrapper);
        return false;
    }
    
    task = wrapper.task;
    return true;
}

TaskFeatures TaskScheduler::analyzeTask(const PT& task) {
    TaskFeatures features;
    features.segment_count = task.content.size();
    features.total_combinations = 1;
    features.max_values = 0;
    
    // 分析每个segment
    for (const auto& seg : task.content) {
        size_t values = 0;
        if (seg.type == 1) { // Letter
            values = seg.ordered_values.size();
        } else if (seg.type == 2) { // Digit
            values = seg.ordered_values.size();
        } else if (seg.type == 3) { // Symbol
            values = seg.ordered_values.size();
        }
        
        features.total_combinations *= values;
        features.max_values = max(features.max_values, values);
    }
    
    // 判断任务类型
    features.task_type = (features.segment_count == 1) ? 0 : 1;
    
    // 分析内存访问模式
    features.memory_pattern = (features.segment_count > 3) ? 1 : 0;
    
    // GPU适应性分析
    features.gpu_suitable = (features.total_combinations > gpu_threshold) && 
                           (features.memory_pattern == 0);
    
    // 预测执行时间
    features.estimated_time = predictExecutionTime(features, LAYER_THREAD);
    
    return features;
}

TaskScheduler::SchedulingDecision TaskScheduler::makeDecision(const TaskFeatures& features) {
    SchedulingDecision decision;
    
    // 选择主层
    decision.primary_layer = selectOptimalLayer(features);
    
    // 选择备用层
    if (decision.primary_layer == LAYER_GPU) {
        decision.secondary_layer = LAYER_THREAD;
    } else if (decision.primary_layer == LAYER_THREAD) {
        decision.secondary_layer = LAYER_SIMD;
    } else {
        decision.secondary_layer = decision.primary_layer;
    }
    
    // 计算批大小
    decision.batch_size = calculateOptimalBatchSize(features, decision.primary_layer);
    
    // 设置优先级
    decision.priority = features.getComplexityScore();
    
    return decision;
}

double TaskScheduler::predictExecutionTime(const TaskFeatures& features, LayerType layer) {
    double base_time = features.total_combinations / perf_model.cpu_throughput;
    
    switch (layer) {
        case LAYER_SIMD:
            return base_time / perf_model.simd_speedup;
            
        case LAYER_THREAD:
            return base_time * perf_model.thread_efficiency / 8; // 假设8线程
            
        case LAYER_GPU:
            {
                double compute_time = features.total_combinations / perf_model.gpu_throughput;
                double transfer_time = 2 * features.total_combinations * 20 / 
                                     perf_model.gpu_transfer_bandwidth;
                return perf_model.gpu_latency + compute_time + transfer_time;
            }
            
        case LAYER_MPI:
            return base_time; // 简化处理
            
        default:
            return base_time;
    }
}

TaskScheduler::LayerType TaskScheduler::selectOptimalLayer(const TaskFeatures& features) {
    // 基于任务特征的启发式选择
    if (features.total_combinations < thread_threshold) {
        return LAYER_SIMD;
    } else if (features.total_combinations < gpu_threshold || !features.gpu_suitable) {
        return LAYER_THREAD;
    } else {
        // 检查GPU利用率
        if (gpu_utilization < 0.8) {
            return LAYER_GPU;
        } else if (cpu_utilization < 0.8) {
            return LAYER_THREAD;
        } else {
            // 负载均衡
            return (active_gpu_tasks < active_cpu_tasks) ? LAYER_GPU : LAYER_THREAD;
        }
    }
}

size_t TaskScheduler::calculateOptimalBatchSize(const TaskFeatures& features, LayerType layer) {
    switch (layer) {
        case LAYER_SIMD:
            return 4; // SSE处理4个
            
        case LAYER_THREAD:
            return min(size_t(1000), features.total_combinations / 8);
            
        case LAYER_GPU:
            // 基于带宽和延迟的优化公式
            return sqrt(2 * perf_model.gpu_transfer_bandwidth * perf_model.gpu_latency / 20);
            
        default:
            return 100;
    }
}

void TaskScheduler::updateThresholds(const PerformanceMonitor& monitor) {
    // 基于性能反馈动态调整阈值
    double cpu_eff = monitor.getLayerEfficiency("CPU");
    double gpu_eff = monitor.getLayerEfficiency("GPU");
    
    if (gpu_eff > cpu_eff * 1.5 && gpu_threshold > 5000) {
        gpu_threshold = gpu_threshold * 0.8; // 降低GPU门槛
    } else if (cpu_eff > gpu_eff * 1.2 && gpu_threshold < 50000) {
        gpu_threshold = gpu_threshold * 1.2; // 提高GPU门槛
    }
}

// ========== PT批次实现 ==========
void PTBatch::prepareForGPU() {
    // 将PT数据转换为GPU友好的格式
    // 这里简化处理，实际需要更复杂的数据重组
    gpu_ready = true;
}

// ========== 统一框架实现 ==========
void UnifiedFramework::initialize(const Config& cfg) {
    config = cfg;
    
    // 初始化内存池
    memory_pool = make_unique<UnifiedMemoryPool>();
    
    // 初始化调度器
    scheduler = make_unique<TaskScheduler>();
    
    // 初始化各层
    // 这里应该注册实际的层实现
}

void UnifiedFramework::shutdown() {
    // 清理资源
    layers.clear();
    scheduler.reset();
    memory_pool.reset();
}

void UnifiedFramework::run(PriorityQueue& queue, size_t max_passwords) {
    size_t generated = 0;
    
    while (!queue.priority.empty() && generated < max_passwords) {
        // 创建任务批次
        PTBatch batch;
        
        // 填充批次
        for (int i = 0; i < 10 && !queue.priority.empty(); i++) {
            PT pt = queue.priority.front();
            queue.priority.erase(queue.priority.begin());
            
            // 提交到调度器
            scheduler->submitTask(pt);
            batch.addTask(pt);
        }
        
        // 处理批次
        processTaskBatch(batch);
        
        generated += batch.size() * 1000; // 估算
    }
    
    // 打印最终报告
    global_monitor.printReport();
}

void UnifiedFramework::processTaskBatch(PTBatch& batch) {
    auto start = steady_clock::now();
    
    // 模拟处理
    std::this_thread::sleep_for(std::chrono::milliseconds(10));  // 添加 std::
    
    auto end = steady_clock::now();
    double elapsed = duration_cast<microseconds>(end - start).count() / 1e6;
    
    global_monitor.recordMetric("BatchProcess", elapsed, batch.size());
}

// ========== 辅助函数实现 ==========
namespace FrameworkUtils {
    int getCurrentNUMANode() {
        return numa_node_of_cpu(sched_getcpu());
    }
    
    void bindToNUMANode(int node) {
        if (numa_available() >= 0) {
            numa_bind(numa_parse_nodestring(to_string(node).c_str()));
        }
    }
    
    size_t getOptimalAlignment(TaskScheduler::LayerType layer) {
        switch (layer) {
            case TaskScheduler::LAYER_SIMD:
                return 16;  // SSE alignment
            case TaskScheduler::LAYER_GPU:
                return 128; // GPU cache line
            case TaskScheduler::LAYER_THREAD:
                return 64;  // CPU cache line
            default:
                return 16;
        }
    }
    
    void* alignPointer(void* ptr, size_t alignment) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
        return reinterpret_cast<void*>(aligned);
    }
    
    double estimateTaskSize(const PT& task) {
        double size = 1.0;
        for (const auto& seg : task.content) {
            size *= seg.ordered_values.size();
        }
        return size;
    }
    
    int detectMemoryPattern(const PT& task) {
        // 简化的内存访问模式检测
        if (task.content.size() > 3) {
            return 1; // Random access
        }
        return 0; // Sequential access
    }
}