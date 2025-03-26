import matplotlib.pyplot as plt
import numpy as np

# 从用户提供的文本数据中提取结构化数据
raw_data = [
    475.299, 1.7863, 266.08,
    414.799, 7.84904, 52.8471,
    410.892, 17.2342, 23.8417,
    413.63, 31.7332, 13.0346,
    403.784, 45.5024, 8.8739,
    413.655, 69.2827, 5.97054,
    405.219, 88.6292, 4.57207,
    409.04, 115.072, 3.55464,
    404.921, 153.466, 2.6385,
    406.85, 180.012, 2.26013,
    415.326, 224.086, 1.85342,
    409.05, 262.379, 1.559,
    413.21, 308.434, 1.3397,
    409.859, 354.586, 1.15588,
    419.35, 418.582, 1.00184
]

# 重组为3列数据
optimized_time = raw_data[::3]  # 每行第一列为优化算法时间
bm_time = raw_data[1::3]        # 每行第二列为大模型提示优化时间
speedup = raw_data[2::3]         # 每行第三列为加速比

sizes = np.arange(1000, 16000, 1000)  # 1000到15000规模
x_labels = [f'{s//1000}K' for s in sizes]  # 简化标签
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax1 = plt.subplots(figsize=(12, 7))

# 绘制优化算法时间（左轴）
ax1.plot(sizes, optimized_time, 'o-', color='#4B72B0', linewidth=2, 
         markersize=8, label='优化算法')
ax1.set_xlabel('数据规模', fontsize=12)
ax1.set_ylabel('优化算法时间 (ms)', color='#4B72B0', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#4B72B0')
ax1.set_yscale('log')  # 对数坐标
ax1.set_ylim(1e2, 1e4)

# 绘制大模型优化时间（右轴）
ax2 = ax1.twinx()
ax2.plot(sizes, bm_time, 's--', color='#C44E52', linewidth=2, 
         markersize=8, label='大模型提示优化')
ax2.set_ylabel('大模型优化时间 (ms)', color='#C44E52', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#C44E52')
ax2.set_yscale('log')
ax2.set_ylim(1e0, 1e3)

# 全局设置
plt.xticks(sizes, x_labels)
plt.title('算法时间对比（对数坐标系）', fontsize=14, pad=20)
ax1.grid(alpha=0.3, linestyle='--')

# 合并图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

# 时间对比子图
plt.subplot(1, 2, 1)
plt.semilogy(sizes, optimized_time, 'o-', color='#4B72B0', label='优化算法')
plt.semilogy(sizes, bm_time, 's--', color='#C44E52', label='大模型优化')
plt.xticks(sizes, x_labels)
plt.grid(alpha=0.3)
plt.legend()
plt.title('运行时间对比')

# 加速比子图
plt.subplot(1, 2, 2)
plt.plot(sizes, speedup, 'D-', color='#55A868')
plt.xticks(sizes, x_labels)
plt.ylabel('加速比 (倍)')
plt.grid(alpha=0.3)
plt.title('加速比趋势')

plt.tight_layout()
plt.show()