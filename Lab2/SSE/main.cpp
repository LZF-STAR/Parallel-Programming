#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <emmintrin.h> 
#include <tmmintrin.h>
#include <vector>
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -O2 -msse2 -mssse3 -o main.exe


int main()
{
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    // std::ofstream a("./files/results.txt");
    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            int generate_n = 10000000;
            if (history + q.total_guesses > 10000000)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                cout << "Hash time:" << time_hash << "seconds" << endl;
                cout << "Train time:" << time_train << "seconds" << endl;
                break;
            }
        }
        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();
            vector<string> &guesses = q.guesses;
            int total = guesses.size();

            for (int i = 0; i < total; i += 4)
            {
                // Prepare up to 4 passwords
                string batch[4];
                for (int j = 0; j < 4; ++j)
                {
                    batch[j] = (i + j < total) ? guesses[i + j] : string();
                }

                // SIMD MD5: compute four hashes at once
                __m128i state[4];
                MD5Hash(batch, state);

                alignas(16) uint32_t outA[4], outB[4], outC[4], outD[4];
                _mm_storeu_si128((__m128i *)outA, state[0]);
                _mm_storeu_si128((__m128i *)outB, state[1]);
                _mm_storeu_si128((__m128i *)outC, state[2]);
                _mm_storeu_si128((__m128i *)outD, state[3]);

                // 用普通数组下标读取
                for (int j = 0; j < 4 && i + j < total; ++j)
                {
                    bit32 a = outA[j];
                    bit32 b = outB[j];
                    bit32 c = outC[j];
                    bit32 d = outD[j];

                    // cout << guesses[i+j] << "\t"
                    //      << setw(8) << setfill('0') << hex << a
                    //      << setw(8) << setfill('0') << hex << b
                    //      << setw(8) << setfill('0') << hex << c
                    //      << setw(8) << setfill('0') << hex << d << endl;
                }
            }
            // 在这里对哈希所需的总时长进行计算
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
}
