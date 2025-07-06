import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 真实的层执行数据
layers = ['SIMD Layer\n(MD5)', 'Threading\n(OpenMP)', 'MPI\n(Distribution)', 'Overhead']
times = [0.257162, 0.899656, 0.768531, 0.044]  # 最后一个是估算的开销
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

# 计算百分比
total_time = sum(times)
percentages = [t/total_time * 100 for t in times]

# 创建水平条形图
y_pos = np.arange(len(layers))
bars = ax.barh(y_pos, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 添加数值标签
for i, (bar, time, pct) in enumerate(zip(bars, times, percentages)):
    # 时间标签
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{time:.3f}s', ha='left', va='center', fontsize=11, fontweight='bold')
    # 百分比标签
    ax.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
            f'{pct:.1f}%', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# 设置标签和标题
ax.set_yticks(y_pos)
ax.set_yticklabels(layers)
ax.set_xlabel('Execution Time (seconds)', fontsize=12)
ax.set_title('Layer Contribution Analysis - Execution Time Breakdown', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# 添加总时间标注
ax.text(0.02, 0.95, f'Total Execution Time: {total_time:.3f}s', 
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# 添加吞吐量信息
throughput_text = 'Layer Performance:\n• SIMD: 3.85×10⁷ pw/s\n• Threading: 8 threads, 77.5% efficiency\n• MPI: 4 processes, 387 operations'
ax.text(0.98, 0.05, throughput_text, transform=ax.transAxes, fontsize=10,
        ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 设置x轴范围
ax.set_xlim(0, max(times) * 1.15)

plt.tight_layout()
plt.savefig('layer_contribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出详细分析
print("\n=== Layer Performance Analysis ===")
for layer, time, pct in zip(layers, times, percentages):
    print(f"{layer.replace(chr(10), ' ')}: {time:.3f}s ({pct:.1f}%)")
print(f"\nTotal Time: {total_time:.3f}s")
print(f"\nKey Insights:")
print(f"- Threading layer dominates execution time ({percentages[1]:.1f}%)")
print(f"- SIMD provides 2.72x speedup despite small time share ({percentages[0]:.1f}%)")
print(f"- MPI overhead is reasonable for single-node execution ({percentages[2]:.1f}%)")