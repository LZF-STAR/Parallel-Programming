#include "PCFG.h"
#include <fstream>
#include "md5_neon.h"
#include <iomanip>
#include <arm_neon.h>
#include <vector>
#include <unordered_set>
#include <mpi.h>

using namespace std;

// 编译指令
// mpic++ -O2 main_mpi.cpp train.cpp guessing_mpi.cpp md5_neon.cpp -o main

int main(int argc, char *argv[])
{
    // 初始化MPI环境
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 设置错误处理器
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长

    PriorityQueue q;

    // 所有进程同步，确保一起开始训练
    MPI_Barrier(MPI_COMM_WORLD);
    double start_train = MPI_Wtime();

    // 所有进程都需要训练模型（因为每个进程需要访问ordered_values）
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();

    // 同步所有进程，确保训练完成
    MPI_Barrier(MPI_COMM_WORLD);
    double end_train = MPI_Wtime();
    time_train = end_train - start_train;

    q.init();
    if (rank == 0)
    {
        cout << "here" << endl;
        cout << "MPI initialized with " << size << " processes" << endl;
        cout.flush();
    }

    int curr_num = 0;

    // 同步所有进程，确保一起开始口令生成
    MPI_Barrier(MPI_COMM_WORLD);
    double start_guess = MPI_Wtime();

    // 记录已生成的猜测总数
    long int history = 0;

    // 生成口令上限
    const long int generate_n = 1e7;

    // 主循环 - 采用简化的通信和检查模式
    bool continue_running = true;
    while (continue_running)
    {
        // 检查退出条件 1: 是否已达生成上限
        int limit_reached = 0;
        if (rank == 0)
        {
            limit_reached = (history + q.total_guesses >= generate_n) ? 1 : 0;
            if (limit_reached)
            {
                cout << "达到生成上限，退出循环" << endl;
                cout.flush();
            }
        }

        // 广播退出状态
        MPI_Bcast(&limit_reached, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (limit_reached == 1)
        {
            continue_running = false;
            break;
        }

        // 检查退出条件 2: 队列是否为空
        int queue_empty = 0;
        if (rank == 0)
        {
            queue_empty = q.priority.empty() ? 1 : 0;
            if (queue_empty)
            {
                cout << "队列为空，退出循环" << endl;
                cout.flush();
            }
        }

        // 广播队列状态
        MPI_Bcast(&queue_empty, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (queue_empty == 1)
        {
            continue_running = false;
            break;
        }

        // 调用口令生成函数（所有进程都参与）
        q.PopNext();

        // 只有rank 0更新计数器和处理哈希
        if (rank == 0)
        {
            q.total_guesses = q.guesses.size();

            // 定期报告进度
            if (q.total_guesses - curr_num >= 100000)
            {
                cout << "Guesses generated: " << history + q.total_guesses << endl;
                cout.flush();
                curr_num = q.total_guesses;
            }

            // 为了避免内存超限，定期进行哈希处理
            if (curr_num > 1000000)
            {
                double start_hash = MPI_Wtime();

                vector<string> &guesses = q.guesses;
                int total = guesses.size();

                // 使用NEON优化的MD5哈希
                for (int i = 0; i < total; i += 4)
                {
                    string pw1 = (i + 0 < total) ? guesses[i + 0] : string();
                    string pw2 = (i + 1 < total) ? guesses[i + 1] : string();
                    string pw3 = (i + 2 < total) ? guesses[i + 2] : string();
                    string pw4 = (i + 3 < total) ? guesses[i + 3] : string();

                    uint32x4_t state[4];
                    MD5Hash(pw1, pw2, pw3, pw4, state);
                }

                double end_hash = MPI_Wtime();
                time_hash += end_hash - start_hash;

                // 记录已经生成的口令总数
                history += curr_num;
                curr_num = 0;
                q.guesses.clear();

                cout << history << " passwords has been hashed" << endl;
                cout.flush();
            }
        }

        // 每轮迭代结束，确保所有进程同步
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // 确保所有进程同步退出
    MPI_Barrier(MPI_COMM_WORLD);
    double end_guess = MPI_Wtime();

    // 只有rank 0输出统计信息
    if (rank == 0)
    {
        time_guess = end_guess - start_guess;

        cout << "\n==================== Performance Summary ====================" << endl;
        cout << "Total guesses generated: " << history + q.total_guesses << endl;
        cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
        cout << "Hash time: " << time_hash << " seconds" << endl;
        cout << "Train time: " << time_train << " seconds" << endl;
        cout << "Number of processes: " << size << endl;
        cout.flush();

        // 打印MPI时间统计
        q.printMPITimingStats();
    }

    MPI_Finalize();
    return 0;
}