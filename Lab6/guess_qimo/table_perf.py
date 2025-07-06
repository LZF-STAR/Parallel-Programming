import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 只使用真实测量的数据
techniques = ['Fusion\nFramework', 'Single\nSIMD', 'Multi-\nthread', 'Multi-\nprocess', 'GPU\nOnly']
technique_labels = ['Fusion Framework', 'Single SIMD', 'Multi-thread', 'Multi-process', 'GPU Only']

# 真实数据矩阵
real_data = {
    'Fusion Framework': {'guess': 1.713, 'hash': 0.257},
    'Single SIMD': {'guess': None, 'hash': 0.7},
    'Multi-thread': {'guess': 0.34, 'hash': None},
    'Multi-process': {'guess': 0.46, 'hash': None},
    'GPU Only': {'guess': 0.6, 'hash': None}
}

# 创建图形
fig, ax = plt.subplots(figsize=(12, 7))

# 准备数据
x = np.arange(len(techniques))
width = 0.35

# 提取数据
guess_times = [real_data[label]['guess'] for label in technique_labels]
hash_times = [real_data[label]['hash'] for label in technique_labels]

# 绘制柱状图，只显示有数据的部分
for i, (g, h) in enumerate(zip(guess_times, hash_times)):
    if g is not None:
        bar1 = ax.bar(x[i] - width/2, g, width, label='Guess Time' if i == 0 else "", 
                      color='#2E86AB', alpha=0.8)
        ax.text(x[i] - width/2, g + 0.02, f'{g:.3f}s', ha='center', va='bottom', fontsize=10)
    
    if h is not None:
        bar2 = ax.bar(x[i] + width/2, h, width, label='Hash Time' if i == 0 else "", 
                      color='#A23B72', alpha=0.8)
        ax.text(x[i] + width/2, h + 0.02, f'{h:.3f}s', ha='center', va='bottom', fontsize=10)

# 添加"N/A"标记
for i, (g, h) in enumerate(zip(guess_times, hash_times)):
    if g is None:
        ax.text(x[i] - width/2, 0.05, 'N/A', ha='center', va='bottom', 
                fontsize=10, style='italic', color='gray')
    if h is None:
        ax.text(x[i] + width/2, 0.05, 'N/A', ha='center', va='bottom', 
                fontsize=10, style='italic', color='gray')

ax.set_xlabel('Parallel Technique', fontsize=12)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title('Performance Comparison: Real Measurement Data Only', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(techniques)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# 设置y轴范围
ax.set_ylim(0, 2.0)

# 添加关键发现文本框
textstr = 'Key Findings:\n• Fusion Framework: Complete pipeline measurement\n• Hash Time: 2.72x speedup (Fusion vs SIMD-only)\n• Generation: Higher coordination overhead in fusion'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出真实数据分析
print("\n=== Real Data Analysis ===")
print(f"Fusion Framework - Guess: {real_data['Fusion Framework']['guess']:.3f}s, Hash: {real_data['Fusion Framework']['hash']:.3f}s")
print(f"SIMD-only Hash Time: {real_data['Single SIMD']['hash']:.3f}s")
print(f"Hash Speedup: {real_data['Single SIMD']['hash'] / real_data['Fusion Framework']['hash']:.2f}x")
print("\nNote: Only measured data is displayed. N/A indicates no measurement for that component.")