#include <mpi.h>
#include <vector>
#include <queue>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <chrono>
#include "unified_framework.h"
#include "PCFG.h"

using namespace std;
using namespace chrono;

// MPI消息标签
enum MPITags {
    TAG_TASK_REQUEST = 1,
    TAG_TASK_RESPONSE = 2,
    TAG_RESULT_SUBMIT = 3,
    TAG_HEARTBEAT = 4,
    TAG_SHUTDOWN = 5,
    TAG_TASK_MIGRATE = 6,
    TAG_CHECKPOINT = 7,
    TAG_LOAD_BALANCE = 8
};

// 任务状态
enum TaskStatus {
    TASK_PENDING = 0,
    TASK_ASSIGNED = 1,
    TASK_COMPLETED = 2,
    TASK_FAILED = 3
};

// 序列化的PT任务
struct SerializedPT {
    int segment_count;
    int segment_types[8];
    int segment_lengths[8];
    int value_counts[8];
    double priority;
    char data[4096]; // 简化的数据存储
};

// 工作节点状态
struct WorkerStatus {
    int rank;
    double cpu_load;
    double memory_usage;
    int active_tasks;
    int completed_tasks;
    steady_clock::time_point last_heartbeat;
    bool is_alive;
};

// 分布式优先队列管理器
class DistributedPriorityQueue {
private:
    priority_queue<pair<double, PT>> global_queue;
    mutex queue_mutex;
    
    // 版本控制
    atomic<uint64_t> version{0};
    
    // 任务追踪
    unordered_map<int, TaskStatus> task_status;
    unordered_map<int, int> task_assignment; // task_id -> worker_rank
    
public:
    void addTask(const PT& task, double priority) {
        lock_guard<mutex> lock(queue_mutex);
        global_queue.push({priority, task});
        version++;
    }
    
    bool getTask(PT& task, double& priority) {
        lock_guard<mutex> lock(queue_mutex);
        if (global_queue.empty()) return false;
        
        auto top = global_queue.top();
        global_queue.pop();
        priority = top.first;
        task = top.second;
        version++;
        return true;
    }
    
    size_t size() const {
        lock_guard<mutex> lock(queue_mutex);
        return global_queue.size();
    }
    
    uint64_t getVersion() const {
        return version.load();
    }
};

// MPI协调器主类
class MPICoordinator {
private:
    int rank;
    int world_size;
    bool is_master;
    
    // 分布式队列（仅主节点使用）
    unique_ptr<DistributedPriorityQueue> dist_queue;
    
    // 工作节点状态追踪
    unordered_map<int, WorkerStatus> worker_status;
    mutex status_mutex;
    
    // 心跳检测
    thread heartbeat_thread;
    atomic<bool> running{true};
    
    // 任务缓冲
    struct TaskBuffer {
        queue<PT> high_priority;
        queue<PT> normal_priority;
        mutex buffer_mutex;
        
        size_t size() const {
            return high_priority.size() + normal_priority.size();
        }
    } local_buffer;
    
    // 容错相关
    struct Checkpoint {
        vector<PT> completed_tasks;
        vector<PT> in_progress_tasks;
        uint64_t timestamp;
    } last_checkpoint;
    
    // 性能统计
    struct Stats {
        atomic<size_t> tasks_distributed{0};
        atomic<size_t> tasks_completed{0};
        atomic<size_t> tasks_failed{0};
        atomic<double> total_time{0};
    } stats;

public:
    MPICoordinator() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        is_master = (rank == 0);
        
        if (is_master) {
            dist_queue = make_unique<DistributedPriorityQueue>();
            initializeMaster();
        } else {
            initializeWorker();
        }
    }
    
    ~MPICoordinator() {
        shutdown();
    }
    
    // 主节点运行循环
    void runMaster(PriorityQueue& pcfg_queue) {
        cout << "[Master] Starting with " << world_size << " processes" << endl;
        
        // 启动心跳监控
        heartbeat_thread = thread([this] { heartbeatMonitor(); });
        
        // 将PCFG队列导入分布式队列
        while (!pcfg_queue.priority.empty()) {
            PT task = pcfg_queue.priority.front();
            pcfg_queue.priority.erase(pcfg_queue.priority.begin());
            dist_queue->addTask(task, task.prob);
        }
        
        // 主循环
        while (running && (dist_queue->size() > 0 || hasActiveTasks())) {
            handleMasterRequests();
            checkLoadBalance();
            
            // 定期检查点
            if (shouldCheckpoint()) {
                createCheckpoint();
            }
        }
        
        // 发送关闭信号
        broadcastShutdown();
    }
    
    // 工作节点运行循环
    void runWorker() {
        cout << "[Worker " << rank << "] Started" << endl;
        
        while (running) {
            // 请求任务
            if (local_buffer.size() < 10) {
                requestTasks();
            }
            
            // 处理本地任务
            PT task;
            if (getLocalTask(task)) {
                processTask(task);
            }
            
            // 发送心跳
            sendHeartbeat();
            
            // 检查消息
            handleWorkerMessages();
        }
    }

private:
    void initializeMaster() {
        // 初始化工作节点状态
        for (int i = 1; i < world_size; i++) {
            worker_status[i] = WorkerStatus{
                i, 0.0, 0.0, 0, 0, 
                steady_clock::now(), true
            };
        }
    }
    
    void initializeWorker() {
        // 工作节点初始化
    }
    
    void handleMasterRequests() {
        MPI_Status status;
        int flag;
        
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        
        if (!flag) return;
        
        switch (status.MPI_TAG) {
            case TAG_TASK_REQUEST:
                handleTaskRequest(status.MPI_SOURCE);
                break;
                
            case TAG_RESULT_SUBMIT:
                handleResultSubmit(status.MPI_SOURCE);
                break;
                
            case TAG_HEARTBEAT:
                handleHeartbeat(status.MPI_SOURCE);
                break;
                
            case TAG_TASK_MIGRATE:
                handleTaskMigration(status.MPI_SOURCE);
                break;
        }
    }
    
    void handleTaskRequest(int source) {
        int requested_count;
        MPI_Recv(&requested_count, 1, MPI_INT, source, TAG_TASK_REQUEST, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // 准备任务批次
        vector<SerializedPT> tasks;
        for (int i = 0; i < requested_count && dist_queue->size() > 0; i++) {
            PT task;
            double priority;
            if (dist_queue->getTask(task, priority)) {
                tasks.push_back(serializeTask(task));
                stats.tasks_distributed++;
            }
        }
        
        // 发送任务
        int actual_count = tasks.size();
        MPI_Send(&actual_count, 1, MPI_INT, source, TAG_TASK_RESPONSE, MPI_COMM_WORLD);
        
        if (actual_count > 0) {
            MPI_Send(tasks.data(), actual_count * sizeof(SerializedPT), 
                    MPI_BYTE, source, TAG_TASK_RESPONSE, MPI_COMM_WORLD);
        }
        
        // 更新工作节点状态
        {
            lock_guard<mutex> lock(status_mutex);
            worker_status[source].active_tasks += actual_count;
        }
    }
    
    void requestTasks() {
        int request_count = 50; // 请求50个任务
        MPI_Send(&request_count, 1, MPI_INT, 0, TAG_TASK_REQUEST, MPI_COMM_WORLD);
        
        int received_count;
        MPI_Recv(&received_count, 1, MPI_INT, 0, TAG_TASK_RESPONSE, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        if (received_count > 0) {
            vector<SerializedPT> tasks(received_count);
            MPI_Recv(tasks.data(), received_count * sizeof(SerializedPT), 
                    MPI_BYTE, 0, TAG_TASK_RESPONSE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // 反序列化并加入本地缓冲
            for (const auto& serialized : tasks) {
                PT task = deserializeTask(serialized);
                lock_guard<mutex> lock(local_buffer.buffer_mutex);
                if (task.prob > 0.01) { // 高优先级阈值
                    local_buffer.high_priority.push(task);
                } else {
                    local_buffer.normal_priority.push(task);
                }
            }
        }
    }
    
    bool getLocalTask(PT& task) {
        lock_guard<mutex> lock(local_buffer.buffer_mutex);
        
        if (!local_buffer.high_priority.empty()) {
            task = local_buffer.high_priority.front();
            local_buffer.high_priority.pop();
            return true;
        } else if (!local_buffer.normal_priority.empty()) {
            task = local_buffer.normal_priority.front();
            local_buffer.normal_priority.pop();
            return true;
        }
        
        return false;
    }
    
    void processTask(const PT& task) {
        auto start = steady_clock::now();
        
        // 调用实际的任务处理
        // 这里应该调用Generate函数等
        
        auto end = steady_clock::now();
        double elapsed = duration_cast<microseconds>(end - start).count() / 1e6;
        
        // 报告完成
        int result_data[3] = {1, 1000, static_cast<int>(elapsed * 1000)};
        MPI_Send(result_data, 3, MPI_INT, 0, TAG_RESULT_SUBMIT, MPI_COMM_WORLD);
    }
    
    void heartbeatMonitor() {
        while (running) {
            this_thread::sleep_for(seconds(5));
            
            lock_guard<mutex> lock(status_mutex);
            auto now = steady_clock::now();
            
            for (auto& [rank, status] : worker_status) {
                auto elapsed = duration_cast<seconds>(now - status.last_heartbeat).count();
                
                if (elapsed > 15 && status.is_alive) {
                    cout << "[Master] Worker " << rank << " appears to be dead" << endl;
                    status.is_alive = false;
                    handleWorkerFailure(rank);
                }
            }
        }
    }
    
    void sendHeartbeat() {
        static auto last_heartbeat = steady_clock::now();
        auto now = steady_clock::now();
        
        if (duration_cast<seconds>(now - last_heartbeat).count() >= 3) {
            double metrics[3] = {
                getCPULoad(),
                getMemoryUsage(),
                static_cast<double>(local_buffer.size())
            };
            
            MPI_Send(metrics, 3, MPI_DOUBLE, 0, TAG_HEARTBEAT, MPI_COMM_WORLD);
            last_heartbeat = now;
        }
    }
    
    void handleHeartbeat(int source) {
        double metrics[3];
        MPI_Recv(metrics, 3, MPI_DOUBLE, source, TAG_HEARTBEAT, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        lock_guard<mutex> lock(status_mutex);
        worker_status[source].cpu_load = metrics[0];
        worker_status[source].memory_usage = metrics[1];
        worker_status[source].active_tasks = metrics[2];
        worker_status[source].last_heartbeat = steady_clock::now();
        worker_status[source].is_alive = true;
    }
    
    void checkLoadBalance() {
        lock_guard<mutex> lock(status_mutex);
        
        // 找出负载最高和最低的节点
        int max_load_rank = -1, min_load_rank = -1;
        double max_load = 0, min_load = 1e9;
        
        for (const auto& [rank, status] : worker_status) {
            if (!status.is_alive) continue;
            
            double load = status.cpu_load + status.active_tasks / 100.0;
            if (load > max_load) {
                max_load = load;
                max_load_rank = rank;
            }
            if (load < min_load) {
                min_load = load;
                min_load_rank = rank;
            }
        }
        
        // 如果负载差异过大，触发迁移
        if (max_load - min_load > 0.5 && max_load_rank != -1 && min_load_rank != -1) {
            initiateTaskMigration(max_load_rank, min_load_rank);
        }
    }
    
    void initiateTaskMigration(int from_rank, int to_rank) {
        int migrate_signal[2] = {from_rank, to_rank};
        MPI_Send(migrate_signal, 2, MPI_INT, from_rank, TAG_LOAD_BALANCE, MPI_COMM_WORLD);
    }
    
    void handleWorkerFailure(int failed_rank) {
        // 重新分配该节点的任务
        cout << "[Master] Redistributing tasks from failed worker " << failed_rank << endl;
        
        // 找到所有分配给失败节点的任务
        // 这需要任务追踪机制
        
        // 将任务重新加入队列
        stats.tasks_failed++;
    }
    
    void createCheckpoint() {
        // 创建检查点
        last_checkpoint.timestamp = duration_cast<milliseconds>(
            steady_clock::now().time_since_epoch()
        ).count();
        
        // 收集所有节点的状态
        for (int i = 1; i < world_size; i++) {
            MPI_Send(&i, 1, MPI_INT, i, TAG_CHECKPOINT, MPI_COMM_WORLD);
        }
    }
    
    bool shouldCheckpoint() {
        static auto last_checkpoint_time = steady_clock::now();
        auto now = steady_clock::now();
        
        return duration_cast<minutes>(now - last_checkpoint_time).count() >= 5;
    }
    
    void broadcastShutdown() {
        for (int i = 1; i < world_size; i++) {
            MPI_Send(&i, 1, MPI_INT, i, TAG_SHUTDOWN, MPI_COMM_WORLD);
        }
    }
    
    void shutdown() {
        running = false;
        if (heartbeat_thread.joinable()) {
            heartbeat_thread.join();
        }
    }
    
    bool hasActiveTasks() {
        lock_guard<mutex> lock(status_mutex);
        for (const auto& [rank, status] : worker_status) {
            if (status.active_tasks > 0) return true;
        }
        return false;
    }
    
    void handleWorkerMessages() {
        MPI_Status status;
        int flag;
        
        MPI_Iprobe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        
        if (!flag) return;
        
        switch (status.MPI_TAG) {
            case TAG_SHUTDOWN:
                running = false;
                break;
                
            case TAG_LOAD_BALANCE:
                handleLoadBalanceRequest();
                break;
                
            case TAG_CHECKPOINT:
                handleCheckpointRequest();
                break;
        }
    }
    
    void handleLoadBalanceRequest() {
        int signal[2];
        MPI_Recv(signal, 2, MPI_INT, 0, TAG_LOAD_BALANCE, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // 迁移部分任务
        int migrate_count = local_buffer.size() / 4;
        vector<SerializedPT> tasks_to_migrate;
        
        for (int i = 0; i < migrate_count; i++) {
            PT task;
            if (getLocalTask(task)) {
                tasks_to_migrate.push_back(serializeTask(task));
            }
        }
        
        // 发送给目标节点
        if (!tasks_to_migrate.empty()) {
            MPI_Send(tasks_to_migrate.data(), 
                    tasks_to_migrate.size() * sizeof(SerializedPT),
                    MPI_BYTE, signal[1], TAG_TASK_MIGRATE, MPI_COMM_WORLD);
        }
    }
    
    void handleCheckpointRequest() {
        // 保存当前状态
        // 实际实现需要持久化到文件
    }
    
    // 辅助函数
    SerializedPT serializeTask(const PT& task) {
        SerializedPT serialized;
        serialized.segment_count = task.content.size();
        serialized.priority = task.prob;
        
        for (int i = 0; i < task.content.size(); i++) {
            serialized.segment_types[i] = task.content[i].type;
            serialized.segment_lengths[i] = task.content[i].length;
            serialized.value_counts[i] = task.content[i].ordered_values.size();
        }
        
        // 简化：不序列化实际的value数据
        return serialized;
    }
    
    PT deserializeTask(const SerializedPT& serialized) {
        PT task;
        task.prob = serialized.priority;
        
        for (int i = 0; i < serialized.segment_count; i++) {
            segment seg(serialized.segment_types[i], serialized.segment_lengths[i]);
            // 简化：需要恢复实际的values
            task.content.push_back(seg);
        }
        
        return task;
    }
    
    double getCPULoad() {
        // 获取CPU负载，实际实现需要系统调用
        return 0.5;
    }
    
    double getMemoryUsage() {
        // 获取内存使用率
        return 0.3;
    }
};

// 非阻塞通信优化
class NonBlockingCommunicator {
private:
    struct PendingMessage {
        void* buffer;
        size_t size;
        int dest;
        int tag;
        MPI_Request request;
        bool completed;
    };
    
    vector<PendingMessage> pending_sends;
    vector<PendingMessage> pending_recvs;
    mutex comm_mutex;
    
public:
    void sendAsync(void* data, size_t size, int dest, int tag) {
        PendingMessage msg;
        msg.buffer = malloc(size);
        memcpy(msg.buffer, data, size);
        msg.size = size;
        msg.dest = dest;
        msg.tag = tag;
        msg.completed = false;
        
        MPI_Isend(msg.buffer, size, MPI_BYTE, dest, tag, 
                  MPI_COMM_WORLD, &msg.request);
        
        lock_guard<mutex> lock(comm_mutex);
        pending_sends.push_back(msg);
    }
    
    bool recvAsync(void* buffer, size_t max_size, int source, int tag) {
        MPI_Status status;
        int flag;
        
        MPI_Iprobe(source, tag, MPI_COMM_WORLD, &flag, &status);
        if (!flag) return false;
        
        int count;
        MPI_Get_count(&status, MPI_BYTE, &count);
        
        if (count > max_size) return false;
        
        MPI_Request request;
        MPI_Irecv(buffer, count, MPI_BYTE, source, tag, 
                  MPI_COMM_WORLD, &request);
        
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        return true;
    }
    
    void checkPending() {
        lock_guard<mutex> lock(comm_mutex);
        
        // 检查发送完成
        auto it = pending_sends.begin();
        while (it != pending_sends.end()) {
            int flag;
            MPI_Test(&it->request, &flag, MPI_STATUS_IGNORE);
            
            if (flag) {
                free(it->buffer);
                it = pending_sends.erase(it);
            } else {
                ++it;
            }
        }
    }
};

// 消息聚合器
class MessageAggregator {
private:
    struct AggregateBuffer {
        vector<char> data;
        vector<size_t> offsets;
        size_t total_size = 0;
        steady_clock::time_point last_flush;
    };
    
    unordered_map<int, AggregateBuffer> buffers; // dest -> buffer
    mutex buffer_mutex;
    
    static constexpr size_t MAX_AGGREGATE_SIZE = 1024 * 1024; // 1MB
    static constexpr auto MAX_WAIT_TIME = milliseconds(100);
    
public:
    void addMessage(int dest, const void* data, size_t size) {
        lock_guard<mutex> lock(buffer_mutex);
        auto& buffer = buffers[dest];
        
        // 添加消息
        buffer.offsets.push_back(buffer.total_size);
        buffer.data.insert(buffer.data.end(), 
                          static_cast<const char*>(data),
                          static_cast<const char*>(data) + size);
        buffer.total_size += size;
        
        // 检查是否需要刷新
        auto now = steady_clock::now();
        if (buffer.total_size >= MAX_AGGREGATE_SIZE ||
            duration_cast<milliseconds>(now - buffer.last_flush) >= MAX_WAIT_TIME) {
            flushBuffer(dest);
        }
    }
    
    void flushAll() {
        lock_guard<mutex> lock(buffer_mutex);
        for (auto& [dest, buffer] : buffers) {
            if (buffer.total_size > 0) {
                flushBuffer(dest);
            }
        }
    }
    
private:
    void flushBuffer(int dest) {
        auto& buffer = buffers[dest];
        if (buffer.total_size == 0) return;
        
        // 发送聚合消息
        MPI_Send(buffer.data.data(), buffer.total_size, MPI_BYTE,
                 dest, TAG_TASK_RESPONSE, MPI_COMM_WORLD);
        
        // 清空缓冲区
        buffer.data.clear();
        buffer.offsets.clear();
        buffer.total_size = 0;
        buffer.last_flush = steady_clock::now();
    }
};