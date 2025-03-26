import matplotlib.pyplot as plt

naive_times = [4.37392, 21.7361, 50.3097, 120.588, 169.37, 230.176, 316.722, 832.031,
               854.385, 1197.79, 1403.02, 2003.15, 2425.67, 3290.16, 2876.98]

cache_times = [2.02866, 9.15982, 20.7011, 33.1801, 46.1555, 69.3572, 92.0338, 147.739,
               195.885, 257.895, 367.117, 367.796, 450.681, 564.13, 449.519]

# 计算加速比（naive_time / cache_time）
speedup_ratio = [n/c for n, c in zip(naive_times, cache_times)]

# 创建画布和坐标系
plt.figure(figsize=(12, 6))
plt.grid(True, linestyle='--', alpha=0.6)  # 添加虚线网格

# 绘制折线图
line, = plt.plot(speedup_ratio, 
                marker='o',  # 圆形数据点标记
                markersize=8, 
                color='#2c7fb8', 
                linewidth=2.5,
                linestyle='--',
                label='Acceleration ratio curve')

# 添加数据标签
for i, ratio in enumerate(speedup_ratio):
    plt.text(i, ratio+0.1, f"{ratio:.1f}x", 
            ha='center', 
            fontsize=9,
            color='#2c7fb8')

# 图表装饰
plt.title("Comparison of cache optimization algorithm speedup ratios (ordinary vs cache)", fontsize=14, pad=20)
plt.xlabel("Data point index", fontsize=12)
plt.ylabel("Acceleration multiple", fontsize=12)

plt.xticks(
    ticks=range(len(speedup_ratio)),  
    labels=[f"{i+1}k" for i in range(len(speedup_ratio))],  
    fontsize=10 
)
plt.ylim(0, max(speedup_ratio)+1)  # y轴留白

# 添加横向参考线
plt.axhline(y=1, color='gray', linestyle=':', linewidth=1.5)

# 显示图例
plt.legend(loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.show()