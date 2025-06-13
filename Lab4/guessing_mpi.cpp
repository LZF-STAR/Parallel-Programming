#include "PCFG.h"
#include <mpi.h>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 使用MPI版本的Generate
    Generate(priority.front());

    // 只有主进程更新优先队列
    if (rank == 0)
    {
        // 生成新的PT
        vector<PT> new_pts = priority.front().NewPTs();
        for (PT pt : new_pts)
        {
            // 计算概率
            CalProb(pt);
            // 将新的PT插入到优先队列中
            for (auto iter = priority.begin(); iter != priority.end(); iter++)
            {
                if (iter != priority.end() - 1 && iter != priority.begin())
                {
                    if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                    {
                        priority.emplace(iter + 1, pt);
                        break;
                    }
                }
                if (iter == priority.end() - 1)
                {
                    priority.emplace_back(pt);
                    break;
                }
                if (iter == priority.begin() && iter->prob < pt.prob)
                {
                    priority.emplace(iter, pt);
                    break;
                }
            }
        }

        // 删除队首PT
        priority.erase(priority.begin());
    }
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

// MPI时间统计变量
static double mpi_total_time = 0.0;
static double mpi_queue_check_time = 0.0;
static double mpi_pt_process_time = 0.0;
static double mpi_segment_setup_time = 0.0;
static double mpi_data_gen_time = 0.0;
static double mpi_comm_time = 0.0;
static int mpi_call_count = 0;


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现*****
// 优化的MPI版本Generate函数
void PriorityQueue::Generate(PT pt)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_total = MPI_Wtime();
    double start_section, end_section;

    // PT处理时间开始
    start_section = MPI_Wtime();

    // 只有rank 0计算概率
    if (rank == 0)
    {
        CalProb(pt);
    }

    if (rank == 0)
    {
        end_section = MPI_Wtime();
        mpi_pt_process_time += (end_section - start_section);
    }

    // Segment设置时间开始
    start_section = MPI_Wtime();

    // 广播PT基本信息
    int content_size = 0;
    if (rank == 0)
    {
        content_size = pt.content.size();
    }
    MPI_Bcast(&content_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 单segment处理
    if (content_size == 1)
    {
        // 需要从rank 0广播的数据
        int segment_type = 0;
        int segment_index = 0;
        int total_values = 0;

        if (rank == 0)
        {
            if (pt.content[0].type == 1)
            {
                segment_type = 1;
                segment_index = m.FindLetter(pt.content[0]);
            }
            else if (pt.content[0].type == 2)
            {
                segment_type = 2;
                segment_index = m.FindDigit(pt.content[0]);
            }
            else
            {
                segment_type = 3;
                segment_index = m.FindSymbol(pt.content[0]);
            }
            total_values = pt.max_indices[0];
        }

        // 广播必要信息
        int info[3] = {segment_type, segment_index, total_values};
        MPI_Bcast(info, 3, MPI_INT, 0, MPI_COMM_WORLD);
        segment_type = info[0];
        segment_index = info[1];
        total_values = info[2];

        // 计算每个进程处理的数据范围
        int items_per_proc = total_values / size;
        int remainder = total_values % size;
        int start_idx = rank * items_per_proc + min(rank, remainder);
        int end_idx = start_idx + items_per_proc + (rank < remainder ? 1 : 0);
        int local_count = end_idx - start_idx;

        if (rank == 0)
        {
            end_section = MPI_Wtime();
            mpi_segment_setup_time += (end_section - start_section);
        }

        // 数据生成时间开始
        start_section = MPI_Wtime();

        // 创建本地结果数组
        vector<string> local_results(local_count);

        // 每个进程使用自己的ordered_values数据
        segment *a;
        if (segment_type == 1)
        {
            a = &m.letters[segment_index];
        }
        else if (segment_type == 2)
        {
            a = &m.digits[segment_index];
        }
        else
        {
            a = &m.symbols[segment_index];
        }

        // 每个进程生成自己负责的部分
        for (int i = 0; i < local_count; i++)
        {
            local_results[i] = a->ordered_values[start_idx + i];
        }

        if (rank == 0)
        {
            end_section = MPI_Wtime();
            mpi_data_gen_time += (end_section - start_section);
        }

        // 通信时间开始
        start_section = MPI_Wtime();

        // ===== 优化的通信部分 =====

        // 1. 准备接收计数和偏移数组
        int *recvcounts = new int[size];
        int *displs = new int[size];

        // 每个进程计算自己的字符串长度总和
        vector<int> str_lengths(local_count);
        int local_total_size = 0;

        for (int i = 0; i < local_count; i++)
        {
            str_lengths[i] = local_results[i].length();
            local_total_size += str_lengths[i];
        }

        // 2. 收集所有进程的字符串计数
        int *all_counts = nullptr;
        if (rank == 0)
        {
            all_counts = new int[size];
        }

        MPI_Gather(&local_count, 1, MPI_INT, all_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 3. 收集所有进程的字符串长度总和
        int *all_sizes = nullptr;
        if (rank == 0)
        {
            all_sizes = new int[size];
        }

        MPI_Gather(&local_total_size, 1, MPI_INT, all_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 4. 准备接收所有字符串长度
        vector<int> all_str_lengths;
        if (rank == 0)
        {
            // 设置接收计数和偏移
            int total_count = 0;
            for (int i = 0; i < size; i++)
            {
                recvcounts[i] = all_counts[i];
                displs[i] = total_count;
                total_count += all_counts[i];
            }

            all_str_lengths.resize(total_count);
        }

        // 5. 收集所有字符串长度
        MPI_Gatherv(str_lengths.data(), local_count, MPI_INT,
                    all_str_lengths.data(), recvcounts, displs, MPI_INT,
                    0, MPI_COMM_WORLD);

        // 6. 处理字符串内容
        if (rank == 0)
        {
            // 复制本地结果
            for (int i = 0; i < local_count; i++)
            {
                guesses.push_back(move(local_results[i]));
            }

            // 从所有进程收集字符串内容
            for (int src_rank = 1; src_rank < size; src_rank++)
            {
                int remote_count = all_counts[src_rank];

                if (remote_count <= 0)
                    continue;

                // 计算总数据大小
                int total_size = all_sizes[src_rank];

                // 接收字符串内容
                char *buffer = new char[total_size];
                MPI_Status status;
                MPI_Recv(buffer, total_size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &status);

                // 解包字符串
                int offset = 0;
                for (int i = 0; i < remote_count; i++)
                {
                    int str_len = all_str_lengths[displs[src_rank] + i];
                    guesses.push_back(string(buffer + offset, str_len));
                    offset += str_len;
                }

                delete[] buffer;
            }

            total_guesses += total_values;

            delete[] all_counts;
            delete[] all_sizes;
        }
        else if (local_count > 0)
        {
            // 非rank 0进程发送字符串内容
            char *buffer = new char[local_total_size];
            int offset = 0;

            for (int i = 0; i < local_count; i++)
            {
                memcpy(buffer + offset, local_results[i].c_str(), str_lengths[i]);
                offset += str_lengths[i];
            }

            // 发送字符串内容
            MPI_Send(buffer, local_total_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);

            delete[] buffer;
        }

        delete[] recvcounts;
        delete[] displs;

        if (rank == 0)
        {
            end_section = MPI_Wtime();
            mpi_comm_time += (end_section - start_section);
        }
    }
    else
    {
        // 多segment情况处理
        string prefix;
        int last_segment_type = 0;
        int last_segment_index = 0;
        int total_values = 0;
        vector<int> curr_indices;

        if (rank == 0)
        {
            // 构建前缀信息
            last_segment_type = pt.content[pt.content.size() - 1].type;
            if (last_segment_type == 1)
            {
                last_segment_index = m.FindLetter(pt.content[pt.content.size() - 1]);
            }
            else if (last_segment_type == 2)
            {
                last_segment_index = m.FindDigit(pt.content[pt.content.size() - 1]);
            }
            else
            {
                last_segment_index = m.FindSymbol(pt.content[pt.content.size() - 1]);
            }

            total_values = pt.max_indices[pt.content.size() - 1];
            curr_indices = pt.curr_indices;
        }

        // 广播当前索引数组大小
        int indices_size = 0;
        if (rank == 0)
        {
            indices_size = curr_indices.size();
        }
        MPI_Bcast(&indices_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 广播当前索引数组
        if (rank != 0)
        {
            curr_indices.resize(indices_size);
        }
        MPI_Bcast(curr_indices.data(), indices_size, MPI_INT, 0, MPI_COMM_WORLD);

        // 广播内容类型
        vector<int> content_types;
        if (rank == 0)
        {
            content_types.resize(pt.content.size());
            for (size_t i = 0; i < pt.content.size(); i++)
            {
                content_types[i] = pt.content[i].type;
            }
        }
        else
        {
            content_types.resize(content_size);
        }
        MPI_Bcast(content_types.data(), content_size, MPI_INT, 0, MPI_COMM_WORLD);

        // 广播其他必要信息
        int info[3] = {last_segment_type, last_segment_index, total_values};
        if (rank == 0)
        {
            info[0] = last_segment_type;
            info[1] = last_segment_index;
            info[2] = total_values;
        }
        MPI_Bcast(info, 3, MPI_INT, 0, MPI_COMM_WORLD);
        last_segment_type = info[0];
        last_segment_index = info[1];
        total_values = info[2];

        // 广播segment索引
        vector<int> segment_indices;
        if (rank == 0)
        {
            segment_indices.resize(pt.content.size());
            for (size_t i = 0; i < pt.content.size(); i++)
            {
                if (pt.content[i].type == 1)
                {
                    segment_indices[i] = m.FindLetter(pt.content[i]);
                }
                else if (pt.content[i].type == 2)
                {
                    segment_indices[i] = m.FindDigit(pt.content[i]);
                }
                else
                {
                    segment_indices[i] = m.FindSymbol(pt.content[i]);
                }
            }
        }
        else
        {
            segment_indices.resize(content_size);
        }
        MPI_Bcast(segment_indices.data(), content_size, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            end_section = MPI_Wtime();
            mpi_segment_setup_time += (end_section - start_section);
        }

        // 数据生成时间开始
        start_section = MPI_Wtime();

        // 每个进程都构建相同的前缀
        prefix.reserve(100); // 预估的前缀长度
        int seg_idx = 0;
        for (int idx : curr_indices)
        {
            if (content_types[seg_idx] == 1)
            {
                prefix += m.letters[segment_indices[seg_idx]].ordered_values[idx];
            }
            else if (content_types[seg_idx] == 2)
            {
                prefix += m.digits[segment_indices[seg_idx]].ordered_values[idx];
            }
            else if (content_types[seg_idx] == 3)
            {
                prefix += m.symbols[segment_indices[seg_idx]].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == content_size - 1)
            {
                break;
            }
        }

        // 计算每个进程处理的数据范围
        int items_per_proc = total_values / size;
        int remainder = total_values % size;
        int start_idx = rank * items_per_proc + min(rank, remainder);
        int end_idx = start_idx + items_per_proc + (rank < remainder ? 1 : 0);
        int local_count = end_idx - start_idx;

        // 创建本地结果数组
        vector<string> local_results(local_count);

        // 获取最后一个segment
        segment *a;
        if (last_segment_type == 1)
        {
            a = &m.letters[last_segment_index];
        }
        else if (last_segment_type == 2)
        {
            a = &m.digits[last_segment_index];
        }
        else
        {
            a = &m.symbols[last_segment_index];
        }

        // 每个进程生成自己负责的部分
        for (int i = 0; i < local_count; i++)
        {
            local_results[i] = prefix + a->ordered_values[start_idx + i];
        }

        if (rank == 0)
        {
            end_section = MPI_Wtime();
            mpi_data_gen_time += (end_section - start_section);
        }

        // 通信时间开始
        start_section = MPI_Wtime();

        // ===== 优化的通信部分（与单segment相同）=====

        // 1. 准备接收计数和偏移数组
        int *recvcounts = new int[size];
        int *displs = new int[size];

        // 每个进程计算自己的字符串长度总和
        vector<int> str_lengths(local_count);
        int local_total_size = 0;

        for (int i = 0; i < local_count; i++)
        {
            str_lengths[i] = local_results[i].length();
            local_total_size += str_lengths[i];
        }

        // 2. 收集所有进程的字符串计数
        int *all_counts = nullptr;
        if (rank == 0)
        {
            all_counts = new int[size];
        }

        MPI_Gather(&local_count, 1, MPI_INT, all_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 3. 收集所有进程的字符串长度总和
        int *all_sizes = nullptr;
        if (rank == 0)
        {
            all_sizes = new int[size];
        }

        MPI_Gather(&local_total_size, 1, MPI_INT, all_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 4. 准备接收所有字符串长度
        vector<int> all_str_lengths;
        if (rank == 0)
        {
            // 设置接收计数和偏移
            int total_count = 0;
            for (int i = 0; i < size; i++)
            {
                recvcounts[i] = all_counts[i];
                displs[i] = total_count;
                total_count += all_counts[i];
            }

            all_str_lengths.resize(total_count);
        }

        // 5. 收集所有字符串长度
        MPI_Gatherv(str_lengths.data(), local_count, MPI_INT,
                    all_str_lengths.data(), recvcounts, displs, MPI_INT,
                    0, MPI_COMM_WORLD);

        // 6. 处理字符串内容
        if (rank == 0)
        {
            // 复制本地结果
            for (int i = 0; i < local_count; i++)
            {
                guesses.push_back(move(local_results[i]));
            }

            // 从所有进程收集字符串内容
            for (int src_rank = 1; src_rank < size; src_rank++)
            {
                int remote_count = all_counts[src_rank];

                if (remote_count <= 0)
                    continue;

                // 计算总数据大小
                int total_size = all_sizes[src_rank];

                // 接收字符串内容
                char *buffer = new char[total_size];
                MPI_Status status;
                MPI_Recv(buffer, total_size, MPI_CHAR, src_rank, 1, MPI_COMM_WORLD, &status);

                // 解包字符串
                int offset = 0;
                for (int i = 0; i < remote_count; i++)
                {
                    int str_len = all_str_lengths[displs[src_rank] + i];
                    guesses.push_back(string(buffer + offset, str_len));
                    offset += str_len;
                }

                delete[] buffer;
            }

            total_guesses += total_values;

            delete[] all_counts;
            delete[] all_sizes;
        }
        else if (local_count > 0)
        {
            // 非rank 0进程发送字符串内容
            char *buffer = new char[local_total_size];
            int offset = 0;

            for (int i = 0; i < local_count; i++)
            {
                memcpy(buffer + offset, local_results[i].c_str(), str_lengths[i]);
                offset += str_lengths[i];
            }

            // 发送字符串内容
            MPI_Send(buffer, local_total_size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);

            delete[] buffer;
        }

        delete[] recvcounts;
        delete[] displs;

        if (rank == 0)
        {
            end_section = MPI_Wtime();
            mpi_comm_time += (end_section - start_section);
        }
    }

    // 记录总时间
    if (rank == 0)
    {
        double end_total = MPI_Wtime();
        mpi_total_time += (end_total - start_total);
        mpi_call_count++;
    }
}

// 打印MPI时间统计
void PriorityQueue::printMPITimingStats()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0 && mpi_call_count > 0)
    {
        cout << "\n========== MPI Timing Statistics ==========" << endl;
        cout << "Total MPI calls: " << mpi_call_count << endl;
        cout << "Total MPI time: " << mpi_total_time << " seconds" << endl;
        cout << "Average time per call: " << mpi_total_time / mpi_call_count << " seconds" << endl;
        cout << "\nTime breakdown:" << endl;
        cout << "  Queue check time: " << mpi_queue_check_time << " seconds ("
             << (mpi_queue_check_time / mpi_total_time * 100) << "%)" << endl;
        cout << "  PT process time: " << mpi_pt_process_time << " seconds ("
             << (mpi_pt_process_time / mpi_total_time * 100) << "%)" << endl;
        cout << "  Segment setup time: " << mpi_segment_setup_time << " seconds ("
             << (mpi_segment_setup_time / mpi_total_time * 100) << "%)" << endl;
        cout << "  Data generation time: " << mpi_data_gen_time << " seconds ("
             << (mpi_data_gen_time / mpi_total_time * 100) << "%)" << endl;
        cout << "  Communication time: " << mpi_comm_time << " seconds ("
             << (mpi_comm_time / mpi_total_time * 100) << "%)" << endl;
        cout << "==========================================\n"
             << endl;
    }
}
