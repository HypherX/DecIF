import matplotlib.pyplot as plt
import numpy as np

# ====================== 折线图数据 ======================
# 横轴数据（保持等间距）
x = np.arange(5)
x_labels = ['10k', '20k', '30k', '50k', '90k']

# 每个系列的 mock 百分比数据（按10k, 20k, 30k, 50k, 90k顺序）
y1 = np.array([73.85, 76.46, 79.10, 79.51, 78.82])  # Prompt
y2 = np.array([49.40, 53.11, 53.02, 52.95, 51.68])  # Rate
y3 = np.array([55.29, 55.30, 56.83, 57.60, 59.73])  # New Method
y4 = np.array([50.50, 50.00, 54.30, 53.10, 51.60])  # Baseline

# 第二组折线图数据（示例数据，你可以替换为实际需要的数据）
y1_2 = np.array([75.0, 77.5, 80.0, 81.0, 80.5])  # Prompt 第二组
y2_2 = np.array([50.0, 54.0, 54.5, 54.0, 53.0])  # Rate 第二组
y3_2 = np.array([56.0, 56.5, 58.0, 59.0, 60.5])  # New Method 第二组
y4_2 = np.array([51.0, 51.5, 55.0, 54.0, 52.5])  # Baseline 第二组

# ====================== 雷达图数据 ======================
# 标签统一
labels = ['Human-\nEval', 'BBH', 'HellaSwag', 'GSM8K', 'MATH', 'MMLU', 'GPQA', 'LiveBench [All]', 'IFEval']
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

# 数据设置（左图）
left_data = {
    'Tulu-3-Mixture': [82.32, 72.25, 82.32, 82.79, 48.78, 78.51, 28.28, 31.30, 78.06],
    'Tulu-3-DecIF':   [82.32, 73.66, 78.58, 84.84, 49.44, 78.45, 33.33, 32.40, 79.85]
}

# 数据设置（右图）
right_data = {
    'UltraChat':       [0.00, 76.89, 17.96, 85.97, 45.16, 77.71, 22.73, 7.40, 61.81],
    'Magpie-Pro':      [79.88, 76.42, 73.49, 91.05, 47.62, 78.20, 2.53, 34.50, 67.22],
    'Tulu-3-Mixture': [82.32, 72.25, 82.32, 82.79, 48.78, 78.51, 28.28, 31.30, 78.06],
    'FuseChat-3.0':   [84.15, 77.25, 54.74, 90.60, 59.18, 78.32, 32.83, 35.80, 68.08],
    'DecIF':           [85.98, 79.50, 76.56, 87.49, 58.08, 78.66, 45.45, 35.40, 71.42]
}

# 扩展数据用于闭合图形
def extend_data(data_dict):
    for key in data_dict:
        data_dict[key] += data_dict[key][:1]
    return data_dict

left_data = extend_data(left_data)
right_data = extend_data(right_data)
angles += angles[:1]

# ====================== 创建图形 ======================
# 设置图形大小和布局 (1行4列)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
ax1 = axes[0]  # 第一个折线图
ax2 = axes[1]  # 第二个折线图
ax3 = axes[2]  # 第一个雷达图
ax4 = axes[3]  # 第二个雷达图

# 颜色定义
color1 = '#1868B2'
color2 = '#DE582B'
color3 = '#018A67'
color4 = '#F3A332'

# ====================== 第一个折线图 ======================
# 上轴（70–80）
ax1.plot(x, y1, color=color1, linestyle='-', linewidth=2.5, label='Prompt (As Labeled)')
ax1.plot(x, y2, color=color2, linestyle='-', linewidth=2.5, marker='o', markerfacecolor=color2, label='Rate (As Labeled)')
ax1.plot(x, y3, color=color3, linestyle='-', linewidth=2.5, marker='^', label='New Method')
ax1.plot(x, y4, color=color4, linestyle='-', linewidth=2.5, marker='s', label='Baseline')
ax1.scatter(x, y1, facecolors='white', edgecolors=color1, marker='o', zorder=3)  # 空心圆点
ax1.set_ylim(70, 80)
ax1.set_yticks(np.arange(70, 85, 5))
ax1.set_yticklabels([f'{v:.0f}%' for v in np.arange(70, 85, 5)], fontsize=12)
ax1.grid(True)
ax1.spines['bottom'].set_visible(False)

# 下轴（45–55）
ax1_plot1 = ax1.twinx()
ax1_plot1.plot(x, y1, color=color1, linestyle='-', linewidth=2.5)
ax1_plot1.plot(x, y2, color=color2, linestyle='-', linewidth=2.5, marker='o', markerfacecolor=color2)
ax1_plot1.plot(x, y3, color=color3, linestyle='-', linewidth=2.5, marker='^')
ax1_plot1.plot(x, y4, color=color4, linestyle='-', linewidth=2.5, marker='s')
ax1_plot1.scatter(x, y1, facecolors='white', edgecolors=color1, marker='o', zorder=3)
ax1_plot1.set_ylim(45, 55)
ax1_plot1.set_yticks(np.arange(45, 60, 5))
ax1_plot1.set_yticklabels([f'{v:.0f}%' for v in np.arange(45, 60, 5)], fontsize=12)
ax1_plot1.grid(True)
ax1_plot1.spines['top'].set_visible(False)

# 添加断轴标记
kwargs = dict(marker=[(-1, -0.3), (1, 0.3)], markersize=10, linestyle='none', color='k', clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax1_plot1.plot([0, 1], [1, 1], transform=ax1_plot1.transAxes, **kwargs)

# x轴设置
ax1_plot1.set_xticks(x)
ax1_plot1.set_xticklabels(x_labels, fontsize=12)
ax1_plot1.set_xlabel('Training Data Size', fontsize=14)
ax1_plot1.set_ylabel('Performance', fontsize=14)

# 图例放在上轴
ax1.legend(loc='upper left', fontsize=10)

# ====================== 第二个折线图 ======================
# 上轴（70–80）
ax2.plot(x, y1_2, color=color1, linestyle='-', linewidth=2.5, label='Prompt (As Labeled)')
ax2.plot(x, y2_2, color=color2, linestyle='-', linewidth=2.5, marker='o', markerfacecolor=color2, label='Rate (As Labeled)')
ax2.plot(x, y3_2, color=color3, linestyle='-', linewidth=2.5, marker='^', label='New Method')
ax2.plot(x, y4_2, color=color4, linestyle='-', linewidth=2.5, marker='s', label='Baseline')
ax2.scatter(x, y1_2, facecolors='white', edgecolors=color1, marker='o', zorder=3)  # 空心圆点
ax2.set_ylim(70, 80)
ax2.set_yticks(np.arange(70, 85, 5))
ax2.set_yticklabels([f'{v:.0f}%' for v in np.arange(70, 85, 5)], fontsize=12)
ax2.grid(True)
ax2.spines['bottom'].set_visible(False)

# 下轴（45–55）
ax2_plot1 = ax2.twinx()
ax2_plot1.plot(x, y1_2, color=color1, linestyle='-', linewidth=2.5)
ax2_plot1.plot(x, y2_2, color=color2, linestyle='-', linewidth=2.5, marker='o', markerfacecolor=color2)
ax2_plot1.plot(x, y3_2, color=color3, linestyle='-', linewidth=2.5, marker='^')
ax2_plot1.plot(x, y4_2, color=color4, linestyle='-', linewidth=2.5, marker='s')
ax2_plot1.scatter(x, y1_2, facecolors='white', edgecolors=color1, marker='o', zorder=3)
ax2_plot1.set_ylim(45, 55)
ax2_plot1.set_yticks(np.arange(45, 60, 5))
ax2_plot1.set_yticklabels([f'{v:.0f}%' for v in np.arange(45, 60, 5)], fontsize=12)
ax2_plot1.grid(True)
ax2_plot1.spines['top'].set_visible(False)

# 添加断轴标记
ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
ax2_plot1.plot([0, 1], [1, 1], transform=ax2_plot1.transAxes, **kwargs)

# x轴设置
ax2_plot1.set_xticks(x)
ax2_plot1.set_xticklabels(x_labels, fontsize=12)
ax2_plot1.set_xlabel('Training Data Size', fontsize=14)
ax2_plot1.set_ylabel('Performance', fontsize=14)

# 图例放在上轴
ax2.legend(loc='upper left', fontsize=10)

# ====================== 雷达图部分 ======================
# 使用一组更清晰专业的颜色
colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
          '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']

def plot_radar(ax, data, title):
    handles = []
    for idx, (label, values) in enumerate(data.items()):
        color = colors[idx % len(colors)]
        line, = ax.plot(angles, values, color=color, linewidth=2, linestyle='solid', label=label)
        ax.fill(angles, values, color=color, alpha=0.2)
        handles.append(line)

    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], labels, fontsize=10)
    
    # 统一 y 轴范围和刻度
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(['20', '40', '60', '80'], fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10)
    
    # 添加图例在每个子图正下方，自动换行
    legend = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=2 if len(handles) <= 4 else 3,
        fontsize=8,
        frameon=False
    )
    return legend

# 绘制雷达图
ax3 = plt.subplot(1, 4, 3, polar=True)
plot_radar(ax3, left_data, "Tulu Series Comparison")

ax4 = plt.subplot(1, 4, 4, polar=True)
plot_radar(ax4, right_data, "DecIF vs Others")

# 调整整体布局
plt.tight_layout()
plt.savefig("pic/combined_plots.pdf", bbox_inches='tight')