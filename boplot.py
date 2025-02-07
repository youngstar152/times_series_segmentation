import numpy as np
import matplotlib.pyplot as plt

# 📌 データの設定
datasets = ["Synthetic", "MoCap", "ActRecTut", "PAMAP2", "USC-HAD", "UCR-SEG"]
methods = ["LSE", "TS2Vec", "Triplet", "TNC", "CPC"]
scores = np.array([
    [0.80, 0.70, 0.60, 0.50, 0.40],
    [0.75, 0.65, 0.55, 0.45, 0.35],
    [0.70, 0.60, 0.50, 0.50, 0.45],
    [0.45, 0.40, 0.42, 0.38, 0.35],
    [0.65, 0.55, 0.50, 0.48, 0.45],
    [0.50, 0.45, 0.40, 0.38, 0.36]
])

# 📌 グラフのスタイル設定
bar_width = 0.15
x = np.arange(len(datasets))  # X軸の位置
hatches = ['//', '\\\\', '||', '--', 'xx']  # 斜線パターン
colors = ['#d73027', '#4575b4', '#8073ac', '#636363', '#fdae61']  # 色

# 📌 プロットの準備
fig, ax = plt.subplots(figsize=(10, 4))

# 📌 各手法のバーをプロット
for i, (method, hatch, color) in enumerate(zip(methods, hatches, colors)):
    bars = ax.bar(x + i * bar_width, scores[:, i], width=bar_width, label=method,
                  hatch=hatch, edgecolor='black', color=color, alpha=0.8)

    # データラベル（"a", "b", "c" ...）を追加
    for bar, label in zip(bars, ["a", "b", "c", "d", "e", "ab"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# 📌 軸ラベルとタイトル
ax.set_ylabel("ARI", fontsize=12)
ax.set_xticks(x + bar_width * 2)  # X軸の位置調整
ax.set_xticklabels(datasets, fontsize=12)
ax.legend(loc='upper center', ncol=6, fontsize=10, frameon=True)

# 📌 グリッド追加
ax.grid(axis='y', linestyle='dashed', alpha=0.5)

# 📌 仕上げ
plt.tight_layout()
plt.savefig("bar_chart.png", dpi=300)  # 高解像度で保存
plt.show()
