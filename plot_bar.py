import matplotlib.pyplot as plt
import numpy as np

# 设置图形大小和布局
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

# 数据设置
categories = ['IFEval', 'Multi-IF', 'FollowBench', 'LiveBench']
bar_width = 0.35
index = np.arange(len(categories))

# 第一组柱状图数据
data1_8B = [68.93, 49.55, 53.84, 50.80]  # 8B模型在第一组数据中的表现
data1_32B = [70.97, 49.94, 55.29, 50.50]  # 32B模型在第一组数据中的表现

# 第二组柱状图数据
data2_8B = [69.88, 51.46, 58.30, 53.10]  # 8B模型在第二组数据中的表现
data2_32B = [71.56, 52.47, 59.17, 53.40]  # 32B模型在第二组数据中的表现

# 颜色定义
color_8B = '#4E79A7'  # 8B模型的颜色
color_32B = '#F28E2B'  # 32B模型的颜色

# ====================== 第一个柱状图 ======================
# 绘制8B柱状图
bars1_8B = ax1.bar(index - bar_width/2, data1_8B, bar_width, 
                  color=color_8B, label='8B', edgecolor='white')

# 绘制32B柱状图
bars1_32B = ax1.bar(index + bar_width/2, data1_32B, bar_width, 
                  color=color_32B, label='70B', edgecolor='white')

# 设置图表属性
ax1.set_ylabel('Performance (%)', fontsize=14)
ax1.set_xticks(index)
ax1.set_xticklabels(categories, fontsize=12)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.legend(fontsize=10)

# ====================== 第二个柱状图 ======================
# 绘制8B柱状图
bars2_8B = ax2.bar(index - bar_width/2, data2_8B, bar_width, 
                  color=color_8B, label='8B', edgecolor='white')

# 绘制32B柱状图
bars2_32B = ax2.bar(index + bar_width/2, data2_32B, bar_width, 
                  color=color_32B, label='32B', edgecolor='white')

# 设置图表属性
ax2.set_ylabel('Performance (%)', fontsize=14)
ax2.set_xticks(index)
ax2.set_xticklabels(categories, fontsize=12)
ax2.set_ylim(0, 100)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.legend(fontsize=10)

# 调整布局
plt.tight_layout()

# 保存和显示
plt.savefig('pic/bar.pdf', bbox_inches='tight')
plt.show()