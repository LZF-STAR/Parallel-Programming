import numpy as np
import matplotlib.pyplot as plt

matrix_sizes = np.arange(1000, 16000, 1000)  # [1000, 2000, ..., 15000]

naive_times = [4.37392, 21.7361, 50.3097, 120.588, 169.37, 230.176, 316.722, 832.031,
               854.385, 1197.79, 1403.02, 2003.15, 2425.67, 3290.16, 2876.98]

cache_times = [2.02866, 9.15982, 20.7011, 33.1801, 46.1555, 69.3572, 92.0338, 147.739,
               195.885, 257.895, 367.117, 367.796, 450.681, 564.13, 449.519]

plt.figure(figsize=(12, 6))

# 绘制两条曲线
plt.plot(matrix_sizes, naive_times, marker='o', label='Naive Algorithm', linewidth=2)
plt.plot(matrix_sizes, cache_times, marker='s', label='Cache Optimized', linewidth=2)

# 设置坐标轴
plt.ylim(0, 3500)
plt.yticks(np.arange(0, 3500, 500))  # 均匀分布的纵轴刻度
plt.xticks(matrix_sizes, rotation=45)  # 旋转横轴标签

# 添加标注
plt.title('Matrix Multiplication Algorithm Performance Comparison', fontsize=14)
plt.xlabel('Matrix Size', fontsize=12)
plt.ylabel('Execution Time (ms)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left')

# 显示图表
plt.tight_layout()
plt.show()