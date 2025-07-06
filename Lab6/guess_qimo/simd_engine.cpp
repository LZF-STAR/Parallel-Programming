#include <immintrin.h>
#include <queue>
#include <thread>
#include <future>
#include <unordered_map>
#include <list>  // 添加这行
#include "unified_framework.h"
#include "md5.h"  // 现在应该能找到了

using namespace std;

// SIMD MD5引擎
class SIMDHashEngine {
private:
    // 批处理队列
    struct HashBatch {
        vector<string> passwords;
        promise<vector<array<uint32_t, 4>>> result_promise;
        size_t batch_id;
    };
    
    queue<HashBatch> pending_batches;
    mutex queue_mutex;
    condition_variable queue_cv;
    
    // 工作线程
    vector<thread> worker_threads;
    atomic<bool> running{true};
    
    // 性能统计
    struct PerformanceStats {
        atomic<size_t> total_hashes{0};
        atomic<size_t> total_batches{0};
        atomic<double> total_time{0};
        atomic<size_t> cache_hits{0};
        atomic<size_t> cache_misses{0};
    } stats;
    
    // LRU缓存
    class LRUCache {
    private:
        struct CacheNode {
            string key;
            array<uint32_t, 4> value;
            list<string>::iterator lru_iter;
        };
        
        unordered_map<string, CacheNode> cache_map;
        list<string> lru_list;
        size_t capacity;
        mutable mutex cache_mutex;
        
    public:
        LRUCache(size_t cap = 100000) : capacity(cap) {}
        
        bool get(const string& key, array<uint32_t, 4>& value) {
            lock_guard<mutex> lock(cache_mutex);
            
            auto it = cache_map.find(key);
            if (it == cache_map.end()) {
                return false;
            }
            
            // 移动到最前面
            lru_list.erase(it->second.lru_iter);
            lru_list.push_front(key);
            it->second.lru_iter = lru_list.begin();
            
            value = it->second.value;
            return true;
        }
        
        void put(const string& key, const array<uint32_t, 4>& value) {
            lock_guard<mutex> lock(cache_mutex);
            
            // 如果已存在，更新
            auto it = cache_map.find(key);
            if (it != cache_map.end()) {
                lru_list.erase(it->second.lru_iter);
                lru_list.push_front(key);
                it->second.lru_iter = lru_list.begin();
                it->second.value = value;
                return;
            }
            
            // 如果缓存满了，删除最久未使用的
            if (cache_map.size() >= capacity) {
                string old_key = lru_list.back();
                lru_list.pop_back();
                cache_map.erase(old_key);
            }
            
            // 插入新元素
            lru_list.push_front(key);
            cache_map[key] = {key, value, lru_list.begin()};
        }
    } hash_cache;

public:
    SIMDHashEngine(int num_workers = 4) {
        // 启动工作线程
        for (int i = 0; i < num_workers; i++) {
            worker_threads.emplace_back([this] { workerLoop(); });
        }
    }
    
    ~SIMDHashEngine() {
        shutdown();
    }
    
    // 异步批量计算接口
    future<vector<array<uint32_t, 4>>> computeBatchAsync(vector<string>&& passwords) {
        HashBatch batch;
        batch.passwords = move(passwords);
        batch.batch_id = stats.total_batches++;
        
        auto future = batch.result_promise.get_future();
        
        {
            lock_guard<mutex> lock(queue_mutex);
            pending_batches.push(move(batch));
        }
        queue_cv.notify_one();
        
        return future;
    }
    
    // 同步批量计算接口
    void computeBatch(const vector<string>& passwords, vector<array<uint32_t, 4>>& results) {
        results.resize(passwords.size());
        
        // 先检查缓存
        vector<int> uncached_indices;
        for (size_t i = 0; i < passwords.size(); i++) {
            if (hash_cache.get(passwords[i], results[i])) {
                stats.cache_hits++;
            } else {
                uncached_indices.push_back(i);
                stats.cache_misses++;
            }
        }
        
        // 处理未缓存的
        if (!uncached_indices.empty()) {
            processSIMDBatch(passwords, results, uncached_indices);
            
            // 更新缓存
            for (int idx : uncached_indices) {
                hash_cache.put(passwords[idx], results[idx]);
            }
        }
    }

private:
    void workerLoop() {
        while (running) {
            unique_lock<mutex> lock(queue_mutex);
            queue_cv.wait(lock, [this] { 
                return !running || !pending_batches.empty(); 
            });
            
            if (!running) break;
            
            if (!pending_batches.empty()) {
                HashBatch batch = move(pending_batches.front());
                pending_batches.pop();
                lock.unlock();
                
                // 处理批次
                auto start = chrono::steady_clock::now();
                vector<array<uint32_t, 4>> results;
                computeBatch(batch.passwords, results);
                auto end = chrono::steady_clock::now();
                
                double elapsed = chrono::duration<double>(end - start).count();
                stats.total_time += elapsed;
                stats.total_hashes += batch.passwords.size();
                
                // 返回结果
                batch.result_promise.set_value(move(results));
            }
        }
    }
    
    void processSIMDBatch(const vector<string>& passwords, 
                         vector<array<uint32_t, 4>>& results,
                         const vector<int>& indices) {
        // 按4个一组处理
        for (size_t i = 0; i < indices.size(); i += 4) {
            int batch_size = min(4, (int)(indices.size() - i));
            
            if (batch_size == 4) {
                // SIMD处理
                processSIMD4(passwords, results, indices, i);
            } else {
                // 串行处理剩余的
                for (int j = 0; j < batch_size; j++) {
                    int idx = indices[i + j];
                    processSerial(passwords[idx], results[idx]);
                }
            }
        }
    }
    
    void processSIMD4(const vector<string>& passwords,
                     vector<array<uint32_t, 4>>& results,
                     const vector<int>& indices,
                     size_t start_idx) {
        // 准备4个消息
        const char* msgs[4];
        Byte* padded[4];
        int lengths[4];
        
        for (int i = 0; i < 4; i++) {
            int idx = indices[start_idx + i];
            msgs[i] = passwords[idx].c_str();
        }
        
        // 批量预处理
        BatchStringProcess(msgs, 4, padded, lengths);
        
        // SIMD MD5计算
        uint32_t simd_results[4][4];
        SIMD_MD5_Batch4(padded, lengths, simd_results);
        
        // 存储结果
        for (int i = 0; i < 4; i++) {
            int idx = indices[start_idx + i];
            for (int j = 0; j < 4; j++) {
                results[idx][j] = simd_results[i][j];
            }
            delete[] padded[i];
        }
    }
    
    void processSerial(const string& password, array<uint32_t, 4>& result) {
        uint32_t hash[4];
        MD5Hash(password, hash);
        for (int i = 0; i < 4; i++) {
            result[i] = hash[i];
        }
    }
    
    void shutdown() {
        running = false;
        queue_cv.notify_all();
        
        for (auto& thread : worker_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
    
public:
    // 性能报告
    void printStats() {
        cout << "=== SIMD Engine Statistics ===" << endl;
        cout << "Total Hashes: " << stats.total_hashes.load() << endl;
        cout << "Total Batches: " << stats.total_batches.load() << endl;
        cout << "Total Time: " << stats.total_time.load() << " seconds" << endl;
        cout << "Throughput: " << stats.total_hashes.load() / stats.total_time.load() 
             << " hashes/second" << endl;
        cout << "Cache Hit Rate: " 
             << (double)stats.cache_hits / (stats.cache_hits + stats.cache_misses) * 100 
             << "%" << endl;
    }
};

// 优化的SIMD字符串操作
class SIMDStringOps {
public:
    // SIMD字符串比较
    static bool simd_strcmp(const char* s1, const char* s2, size_t len) {
        size_t i = 0;
        
        // SIMD部分（16字节对齐）
        for (; i + 16 <= len; i += 16) {
            __m128i v1 = _mm_loadu_si128((__m128i*)(s1 + i));
            __m128i v2 = _mm_loadu_si128((__m128i*)(s2 + i));
            __m128i cmp = _mm_cmpeq_epi8(v1, v2);
            
            if (_mm_movemask_epi8(cmp) != 0xFFFF) {
                return false;
            }
        }
        
        // 处理剩余部分
        for (; i < len; i++) {
            if (s1[i] != s2[i]) return false;
        }
        
        return true;
    }
    
    // SIMD字符串复制
    static void simd_strcpy(char* dst, const char* src, size_t len) {
        size_t i = 0;
        
        // SIMD部分
        for (; i + 16 <= len; i += 16) {
            __m128i data = _mm_loadu_si128((__m128i*)(src + i));
            _mm_storeu_si128((__m128i*)(dst + i), data);
        }
        
        // 处理剩余部分
        for (; i < len; i++) {
            dst[i] = src[i];
        }
    }
    
    // SIMD内存清零
    static void simd_memset(void* dst, int value, size_t len) {
        __m128i val = _mm_set1_epi8(value);
        size_t i = 0;
        
        // 对齐到16字节边界
        while (i < len && ((uintptr_t)(dst) + i) % 16 != 0) {
            ((char*)dst)[i++] = value;
        }
        
        // SIMD部分
        for (; i + 16 <= len; i += 16) {
            _mm_store_si128((__m128i*)((char*)dst + i), val);
        }
        
        // 处理剩余部分
        for (; i < len; i++) {
            ((char*)dst)[i] = value;
        }
    }
};

// 与上层的集成接口
class SIMDProcessor : public IComputeLayer<StringBatch, StringBatch> {
private:
    SIMDHashEngine hash_engine;
    PerformanceMonitor* monitor;
    
public:
    SIMDProcessor() : hash_engine(thread::hardware_concurrency()) {
        monitor = UnifiedFramework::getInstance()->getMonitor();
    }
    
    bool canProcess(const StringBatch& input) override {
        // SIMD层可以处理所有字符串批次
        return input.size() > 0;
    }
    
    void process(const StringBatch& input, StringBatch& output) override {
        auto start = chrono::steady_clock::now();
        
        // 转换输入格式
        vector<string> passwords;
        passwords.reserve(input.size());
        
        for (size_t i = 0; i < input.size(); i++) {
            passwords.emplace_back(input.data()[i]);
        }
        
        // 计算MD5
        vector<array<uint32_t, 4>> hashes;
        hash_engine.computeBatch(passwords, hashes);
        
        // 转换输出格式（这里简化处理）
        for (const auto& hash : hashes) {
            char hash_str[33];
            sprintf(hash_str, "%08x%08x%08x%08x", 
                   hash[0], hash[1], hash[2], hash[3]);
            output.addString(hash_str, 32);
        }
        
        auto end = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(end - start).count();
        
        // 更新性能指标
        updateMetrics(elapsed, input.size() * 20); // 假设平均密码长度20
    }
    
    double getPerformanceMetrics() override {
        return 1000000.0; // 每秒100万个哈希
    }
    
    void updateMetrics(double time, size_t data_size) override {
        monitor->recordMetric("SIMD", time, data_size);
    }
};

// 数据预取优化
class PrefetchOptimizer {
public:
    template<typename T>
    static void prefetchBatch(const vector<T>& data, size_t start, size_t count) {
        // L1缓存预取
        for (size_t i = start; i < start + count && i < data.size(); i++) {
            _mm_prefetch((const char*)&data[i], _MM_HINT_T0);
        }
        
        // L2缓存预取
        for (size_t i = start + count; i < start + count * 2 && i < data.size(); i++) {
            _mm_prefetch((const char*)&data[i], _MM_HINT_T1);
        }
    }
    
    // 内存预取模式
    enum PrefetchPattern {
        SEQUENTIAL,
        STRIDED,
        RANDOM
    };
    
    static void adaptivePrefetch(void* base, size_t element_size, 
                                 const vector<int>& indices,
                                 PrefetchPattern pattern) {
        switch (pattern) {
            case SEQUENTIAL:
                for (int i = 0; i < min(16, (int)indices.size()); i++) {
                    _mm_prefetch((char*)base + indices[i] * element_size, _MM_HINT_T0);
                }
                break;
                
            case STRIDED:
                // 预取固定间隔的元素
                for (int i = 0; i < indices.size(); i += 4) {
                    _mm_prefetch((char*)base + indices[i] * element_size, _MM_HINT_T0);
                }
                break;
                
            case RANDOM:
                // 随机访问模式，预取可能性较高的元素
                // 这里需要更复杂的预测逻辑
                break;
        }
    }
};

// 注册SIMD层到框架
void registerSIMDLayer() {
    auto framework = UnifiedFramework::getInstance();
    framework->registerLayer("SIMD", make_unique<SIMDProcessor>());
}