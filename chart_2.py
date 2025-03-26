import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False

sizes = np.arange(10000, 110000, 10000)
x_labels = [f'{i//10000}w' for i in sizes]  # x轴标签


naive_times = [1623.38, 3246.51, 4890.69, 6455.39, 8099.28,
               9671.98, 11277.9, 12870.4, 14462.9, 16058.4]

four_way_times = [651.02, 1315.02, 1957.16, 2578.77, 3260.06,
                  3881.28, 4538.52, 5190.16, 5842.36, 6494.43]

plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(sizes, naive_times, marker='o', label='oridinary algorithms')
plt.plot(sizes, four_way_times, marker='s', label='Four-way chain accumulation')

# 坐标轴设置
plt.xticks(sizes, x_labels)  # x轴刻度标签
plt.yticks(np.arange(0, 17001, 2000))  # y轴均匀刻度
plt.ylim(0, 17000)  # y轴范围

# 添加标签和标题
plt.xlabel('Data scale')
plt.ylabel('Running time (ms)')
plt.title('Algorithm performance comparison')

plt.grid(alpha=0.3)
plt.legend()
plt.show()