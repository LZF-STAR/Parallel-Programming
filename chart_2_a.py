import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False

# 数据定义
sizes = np.arange(10000, 110000, 10000)
x_labels = [f'{i//10000}w' for i in sizes]
naive_times = [1623.38, 3246.51, 4890.69, 6455.39, 8099.28, 
               9671.98, 11277.9, 12870.4, 14462.9, 16058.4]
four_way_times = [651.02, 1315.02, 1957.16, 2578.77, 3260.06,
                  3881.28, 4538.52, 5190.16, 5842.36, 6494.43]

# 计算加速比
speedup_ratio = np.array(naive_times) / np.array(four_way_times)

# 图表绘制
plt.figure(figsize=(10, 6))
plt.plot(sizes, speedup_ratio, marker='D', linestyle='--', 
         color='#BF95C1', linewidth=2, markersize=8, 
         markerfacecolor='white', markeredgewidth=2, 
         label='四路链式 vs 朴素算法加速比')

# 坐标轴与样式优化
plt.xticks(sizes, x_labels, fontsize=10)  # 网页1中的坐标轴标签设计
plt.yticks(np.arange(0, 3.5, 0.5), fontsize=10)
plt.ylim(0, 3)
plt.xlabel('数据规模', fontsize=12, labelpad=10)
plt.ylabel('加速比', fontsize=12, labelpad=10)
plt.title('算法加速比趋势分析', fontsize=14, pad=20)

# 添加辅助元素
plt.grid(alpha=0.3, linestyle=':')  # 网页2中推荐的柔和网格线
plt.legend(frameon=False, fontsize=10, loc='upper left')

# 添加数据标注
for x, y in zip(sizes, speedup_ratio):
    plt.text(x, y+0.1, f'{y:.1f}x', ha='center', 
             fontsize=9, color='#752995')  # 网页3中的颜色方案

# 保存高清图
plt.savefig('speedup_ratio.png', dpi=300, bbox_inches='tight')
plt.show()