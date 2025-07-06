#include "PCFG.h"
#include <omp.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <functional>
using namespace std;

// 全局线程池和任务队列
class HybridThreadPool {
private:
    vector<thread> workers;
    queue<function<void()>> tasks;
    mutex queue_mutex;
    condition_variable condition;
    atomic<bool> stop{false};
    atomic<int> active_tasks{0};
    
    // 工作窃取队列
    struct WorkStealingQueue {
        deque<function<void()>> tasks;
        mutex mtx;
        
        bool steal(function<void()>& task) {
            lock_guard<mutex> lock(mtx);
            if (!tasks.empty()) {
                task = move(tasks.back());
                tasks.pop_back();
                return true;
            }
            return false;
        }
    };
    
    vector<WorkStealingQueue> local_queues;

public:
    HybridThreadPool(size_t num_threads) : local_queues(num_threads) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this, i] {
                workerFunction(i);
            });
        }
    }
    
    ~HybridThreadPool() {
        stop = true;
        condition.notify_all();
        for (thread& worker : workers) {
            worker.join();
        }
    }
    
    template<class F>
    void enqueue(F&& f) {
        {
            unique_lock<mutex> lock(queue_mutex);
            tasks.emplace(forward<F>(f));
        }
        condition.notify_one();
    }
    
    void wait() {
        unique_lock<mutex> lock(queue_mutex);
        condition.wait(lock, [this] { 
            return tasks.empty() && active_tasks == 0; 
        });
    }

private:
    void workerFunction(size_t thread_id) {
        while (!stop) {
            function<void()> task;
            
            // 尝试从全局队列获取任务
            {
                unique_lock<mutex> lock(queue_mutex);
                condition.wait(lock, [this] { 
                    return stop || !tasks.empty(); 
                });
                
                if (stop && tasks.empty()) return;
                
                if (!tasks.empty()) {
                    task = move(tasks.front());
                    tasks.pop();
                    active_tasks++;
                }
            }
            
            // 执行任务
            if (task) {
                task();
                active_tasks--;
                condition.notify_all();
            }
            
            // 工作窃取
            if (!task) {
                for (size_t i = 0; i < local_queues.size(); ++i) {
                    if (i != thread_id && local_queues[i].steal(task)) {
                        active_tasks++;
                        task();
                        active_tasks--;
                        break;
                    }
                }
            }
        }
    }
};

// 全局线程池实例
static HybridThreadPool* g_thread_pool = nullptr;

// CPU-GPU协同决策
struct TaskDecisionMaker {
    static constexpr size_t GPU_THRESHOLD = 10000;
    static constexpr size_t BATCH_SIZE = 1000;
    
    static bool shouldUseGPU(const PT& pt) {
        // 估算任务规模
        size_t task_size = 1;
        for (const auto& seg : pt.content) {
            if (seg.type == 1) {
                // 这里需要访问模型数据，简化处理
                task_size *= 100; // 假设每个segment有100个可能值
            }
        }
        
        // 大于阈值时使用GPU
        return task_size > GPU_THRESHOLD;
    }
};

// 并行化的PriorityQueue实现
void PriorityQueue::CalProb(PT &pt) {
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices) {
        if (pt.content[index].type == 1) {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2) {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3) {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init() {
    // 初始化线程池
    if (!g_thread_pool) {
        int num_threads = omp_get_max_threads();
        g_thread_pool = new HybridThreadPool(num_threads);
    }
    
    for (PT pt : m.ordered_pts) {
        for (segment seg : pt.content) {
            if (seg.type == 1) {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2) {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3) {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext() {
    Generate(priority.front());
    
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts) {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++) {
            if (iter != priority.end() - 1 && iter != priority.begin()) {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob) {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1) {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob) {
                priority.emplace(iter, pt);
                break;
            }
        }
    }
    
    priority.erase(priority.begin());
}

vector<PT> PT::NewPTs() {
    vector<PT> res;
    
    if (content.size() == 1) {
        return res;
    } else {
        int init_pivot = pivot;
        
        for (int i = pivot; i < curr_indices.size() - 1; i += 1) {
            curr_indices[i] += 1;
            
            if (curr_indices[i] < max_indices[i]) {
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

// 并行化的Generate函数
void PriorityQueue::Generate(PT pt) {
    CalProb(pt);
    
    // 检查是否应该使用GPU
    bool use_gpu = TaskDecisionMaker::shouldUseGPU(pt);
    
    if (pt.content.size() == 1) {
        segment *a;
        if (pt.content[0].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        int max_idx = pt.max_indices[0];
        
        // 使用OpenMP并行化
        if (!use_gpu && max_idx > 100) {
            vector<string> local_guesses(max_idx);
            
            #pragma omp parallel for
            for (int i = 0; i < max_idx; i++) {
                local_guesses[i] = a->ordered_values[i];
            }
            
            // 合并结果
            #pragma omp critical
            {
                guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.end());
                total_guesses += max_idx;
            }
        } else {
            // 串行处理小任务
            for (int i = 0; i < max_idx; i++) {
                string guess = a->ordered_values[i];
                guesses.emplace_back(guess);
                total_guesses += 1;
            }
        }
    } else {
        // 多segment的情况
        string guess;
        int seg_idx = 0;
        
        // 构建前缀
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1) {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2) {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3) {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1) {
                break;
            }
        }
        
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        int max_idx = pt.max_indices[pt.content.size() - 1];
        
        // 使用混合并行策略
        if (!use_gpu && max_idx > 1000) {
            // 分块处理
            const int chunk_size = 100;
            const int num_chunks = (max_idx + chunk_size - 1) / chunk_size;
            
            // 使用线程池处理各个块
            mutex result_mutex;
            vector<future<void>> futures;
            
            for (int chunk = 0; chunk < num_chunks; chunk++) {
                int start = chunk * chunk_size;
                int end = min(start + chunk_size, max_idx);
                
                g_thread_pool->enqueue([this, start, end, &guess, a, &result_mutex]() {
                    vector<string> local_batch;
                    local_batch.reserve(end - start);
                    
                    for (int i = start; i < end; i++) {
                        string temp = guess + a->ordered_values[i];
                        local_batch.push_back(move(temp));
                    }
                    
                    lock_guard<mutex> lock(result_mutex);
                    guesses.insert(guesses.end(), 
                                 make_move_iterator(local_batch.begin()),
                                 make_move_iterator(local_batch.end()));
                    total_guesses += (end - start);
                });
            }
            
            // 等待所有任务完成
            g_thread_pool->wait();
            
        } else {
            // 串行处理小任务
            for (int i = 0; i < max_idx; i++) {
                string temp = guess + a->ordered_values[i];
                guesses.emplace_back(temp);
                total_guesses += 1;
            }
        }
    }
}