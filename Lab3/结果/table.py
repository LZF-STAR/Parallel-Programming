import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, ScalarFormatter

# ============ 中文显示配置 ============ #  [1,2,6](@ref)
plt.rcParams['font.family'] = 'SimHei'  # 设置全局中文字体（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常问题

# 数据转换
task_values = [1e4, 1e5, 1e6, 1e7, 1e8]
serial_times = [0.010325, 0.010453, 0.067951, 0.560718, 5.96798]
openmp_times = [0.00331, 0.003489, 0.036022, 0.373772, 3.21209]
pthread_times = [0.002733, 0.002583, 0.034395, 0.343279, 2.72124]

plt.figure(figsize=(14, 7), dpi=100)
ax = plt.gca()

# 绘制折线（添加中文标签）
(ser_line,) = plt.plot(task_values, serial_times, 'o--', color='#1f77b4',
                      linewidth=2.5, markersize=12, label='串行时间')
(omp_line,) = plt.plot(task_values, openmp_times, 's-', color='#2ca02c',
                      linewidth=2.5, markersize=10, alpha=0.9, label='OpenMP')
(pth_line,) = plt.plot(task_values, pthread_times, 'D-.', color='#d62728',
                      linewidth=2.5, markersize=9, alpha=0.9, label='Pthread')

# ============ X轴定制 ============ #
ax.xaxis.set_major_locator(MultipleLocator(1e7))
ax.xaxis.set_minor_locator(MultipleLocator(0.5e7))
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
ax.set_xlim(0, 1.1e8)

# ============ Y轴定制 ============ #
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.2))
ax.yaxis.set_major_formatter('{x:.1f} s')
ax.set_ylim(0, 6.5)

# ============ 中文元素配置 ============ #  [3,5](@ref)
plt.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)
plt.title('多线程方案执行时间对比（8线程）', fontsize=16, pad=20)  # 中文标题
plt.xlabel('任务规模', fontsize=12)  # 中文X轴标签
plt.ylabel('执行时间', fontsize=12)  # 中文Y轴标签

# 图例中文显示
plt.legend(frameon=True, fontsize=10, prop={'family':'SimHei'})  # 强制中文字体

# 标签防重叠处理
plt.xticks(rotation=45, ha='right')  # 45度旋转标签防重叠 [3](@ref)
plt.tight_layout()
plt.show()