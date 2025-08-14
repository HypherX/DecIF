import matplotlib.pyplot as plt
import numpy as np

# 数据设置
categories = ['IFEval', 'Multi-IF', 'FollowBench', 'LiveBench']
bar_width = 0.35
index = np.arange(len(categories))

# 第一组柱状图数据
data1_8B = [74.03, 60.46, 63.43, 66.80]  # 8B模型在第一组数据中的表现
data1_32B = [75.67, 61.15, 64.87, 67.20]  # 32B模型在第一组数据中的表现

# 第二组柱状图数据
data2_8B = [84.22, 67.67, 67.02, 52.20]  # 8B模型在第二组数据中的表现
data2_32B = [89.34, 71.62, 69.70, 59.50]  # 32B模型在第二组数据中的表现

color_8B = '#4E79A7'
color_32B = '#F28E2B'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw={'wspace': 0.10})

# --- 第一个柱状图 ---
bars1_8B = ax1.bar(index - bar_width/2, data1_8B, bar_width, color=color_8B, label='hard', edgecolor='white', linewidth=1.1)
bars1_32B = ax1.bar(index + bar_width/2, data1_32B, bar_width, color=color_32B, label='soft', edgecolor='white', linewidth=1.1)

ax1.set_ylabel('Performance (%)', fontsize=10,labelpad=-5)
ax1.set_xticks(index)
ax1.set_xticklabels(categories, fontsize=9, fontweight='bold', rotation=28, ha='right')
ax1.set_ylim(0, 100)
ax1.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.65)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(0.9)
ax1.spines['bottom'].set_linewidth(0.9)
ax1.legend(fontsize=9, loc='upper right', frameon=False, borderpad=0.2, handlelength=1.3, handletextpad=0.5, labelspacing=0.2)
ax1.set_title('(a)', fontsize=10, pad=3)
ax1.tick_params(axis='y', labelsize=9)

# --- 第二个柱状图 ---
bars2_8B = ax2.bar(index - bar_width/2, data2_8B, bar_width, color=color_8B, label='hard', edgecolor='white', linewidth=1.1)
bars2_32B = ax2.bar(index + bar_width/2, data2_32B, bar_width, color=color_32B, label='soft', edgecolor='white', linewidth=1.1)

ax2.set_ylabel('Performance (%)', fontsize=10,labelpad=-5)
ax2.set_xticks(index)
ax2.set_xticklabels(categories, fontsize=9, fontweight='bold', rotation=28, ha='right')
ax2.set_ylim(0, 100)
ax2.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.65)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(0.9)
ax2.spines['bottom'].set_linewidth(0.9)
ax2.legend(fontsize=9, loc='upper right', frameon=False, borderpad=0.2, handlelength=1.3, handletextpad=0.5, labelspacing=0.2)
ax2.set_title('(b)', fontsize=10, pad=3)
ax2.tick_params(axis='y', labelsize=9)
# --- 让y轴标签不重叠 ---
ax2.yaxis.label.set_visible(False)  # 只留左图有ylabel

plt.savefig('pic/bar_reward.pdf', dpi=300, bbox_inches='tight')
plt.show()