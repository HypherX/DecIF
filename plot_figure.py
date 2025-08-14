import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ====================== 折线图数据 ======================
# 横轴数据（保持等间距）
x = np.arange(5)
x_labels = ['10k', '20k', '30k', '50k', '90k']
x_labels_2 = ['5k', '10k', '15k', '20k', '25k']

# 每个系列的 mock 百分比数据（按10k, 20k, 30k, 50k, 90k顺序）
y1 = np.array([73.85, 76.46, 79.10, 79.51, 78.82])  # Prompt
y2 = np.array([49.40, 53.11, 53.02, 52.95, 51.68])  # Rate
y3 = np.array([55.29, 55.30, 56.83, 57.60, 59.73])  # New Method
y4 = np.array([50.50, 50.00, 54.30, 53.10, 51.60])  # Baseline

# 第二组折线图数据（示例数据，你可以替换为实际需要的数据）
y1_2 = np.array([84.89, 86.29, 88.61, 89.83, 88.74])  # Prompt 第二组
y2_2 = np.array([66.14, 67.73, 71.35, 70.87, 68.64])  # Rate 第二组
y3_2 = np.array([64.32, 65.46, 68.66, 70.25, 60.5])  # New Method 第二组
y4_2 = np.array([56.90, 57.30, 58.00, 60.00, 57.80])  # Baseline 第二组

# ====================== 雷达图数据 ======================
labels = ['Human-\nEval', 'BBH', 'HellaSwag', 'GSM8K', 'MATH', 'MMLU', 'GPQA', 'LiveBench [All]', 'IFEval']
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # 闭合

left_data = {
    'Tulu-3-Mixture': [82.32, 72.25, 82.32, 82.79, 48.78, 78.51, 28.28, 31.30, 78.06],
    'Tulu-3-DecIF':   [82.32, 73.66, 78.58, 84.84, 49.44, 78.45, 33.33, 32.40, 79.85]
}
right_data = {
    'UltraChat':       [0.00, 76.89, 17.96, 85.97, 45.16, 77.71, 22.73, 7.40, 61.81],
    'Magpie-Pro':      [79.88, 76.42, 73.49, 91.05, 47.62, 78.20, 2.53, 34.50, 67.22],
    'Tulu-3-Mixture': [82.32, 72.25, 82.32, 82.79, 48.78, 78.51, 28.28, 31.30, 78.06],
    'FuseChat-3.0':   [84.15, 77.25, 54.74, 90.60, 59.18, 78.32, 32.83, 35.80, 68.08],
    'DecIF':           [85.98, 79.50, 76.56, 87.49, 58.08, 78.66, 45.45, 35.40, 71.42]
}
for data in [left_data, right_data]:
    for k in data:
        data[k].append(data[k][0])  # 闭合

# ====================== 专业配色 ======================
colors = ['#1868B2', '#DE582B', '#018A67', '#F3A332']

# ====================== 画布和子图（紧凑） ======================
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), gridspec_kw={'wspace':0.30})

ax1, ax2, ax3, ax4 = axes
fig.delaxes(ax3)
fig.delaxes(ax4)
ax3 = fig.add_subplot(1, 4, 3, polar=True)
ax4 = fig.add_subplot(1, 4, 4, polar=True)

# ====================== 折线图1 ======================
ax1.plot(x, y1, color=colors[0], linewidth=1.8, marker='o', markersize=5, label='IFEval')
ax1.plot(x, y2, color=colors[1], linewidth=1.8, marker='^', markersize=5, label='Multi-IF')
ax1.plot(x, y3, color=colors[2], linewidth=1.8, marker='s', markersize=5, label='FollowBench')
ax1.plot(x, y4, color=colors[3], linewidth=1.8, marker='d', markersize=5, label='LiveBench')
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, fontsize=9)
ax1.set_ylim(48, 83)
ax1.set_yticks(np.arange(48, 84, 6))
ax1.set_ylabel('Performance (%)', fontsize=10)
ax1.set_xlabel('SFT Data Size', fontsize=10)
ax1.legend(
    fontsize=9, frameon=True,
    loc='center left', bbox_to_anchor=(0.05, 0.5), borderaxespad=0
)
ax1.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
ax1.set_title('(a)', fontsize=10, pad=5)

# ====================== 折线图2 ======================
ax2.plot(x, y1_2, color=colors[0], linewidth=1.8, marker='o', markersize=5, label='IFEval')
ax2.plot(x, y2_2, color=colors[1], linewidth=1.8, marker='^', markersize=5, label='Multi-IF')
ax2.plot(x, y3_2, color=colors[2], linewidth=1.8, marker='s', markersize=5, label='FollowBench')
ax2.plot(x, y4_2, color=colors[3], linewidth=1.8, marker='d', markersize=5, label='Baseline')
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels_2, fontsize=9)
ax2.set_ylim(53, 92)
ax2.set_yticks(np.arange(53, 93, 7))
ax2.set_ylabel('Performance (%)', fontsize=10)
ax2.set_xlabel('RL Data Size', fontsize=10)
ax2.legend(
    fontsize=9, frameon=True,
    loc='center left', bbox_to_anchor=(0.05, 0.5), borderaxespad=0
)
ax2.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
ax2.set_title('(b)', fontsize=10, pad=5)

def plot_radar(ax, data, title):
    radar_colors = ['#1868B2', '#DE582B', '#018A67', '#F3A332', '#7A52A6', '#4E79A7', '#E15759']
    legend_lines = []
    labels_list = []
    for i, (name, scores) in enumerate(data.items()):
        color = radar_colors[i % len(radar_colors)]
        ax.plot(angles, scores, color=color, linewidth=1.5)
        ax.fill(angles, scores, color=color, alpha=0.13)
        legend_lines.append(Line2D([0], [0], color=color, lw=1.5))
        labels_list.append(name)
    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], labels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(['20', '40', '60', '80'], fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_title(title, fontsize=10, pad=8)
    ax.legend(legend_lines, labels_list,
            loc='lower center', bbox_to_anchor=(0.6, -0.40), ncol=2, fontsize=8, frameon=True)
plot_radar(ax3, left_data, '(c)')
plot_radar(ax4, right_data, '(d)')

# plt.tight_layout()
plt.savefig("pic/combined_picture.pdf", dpi=300, bbox_inches='tight')