import numpy as np
import matplotlib.pyplot as plt

# データセットのリスト
datasets = ["Synthetic", "MoCap", "ActRecTut", "PAMAP2", "USC-HAD"]
methods = [
    "HVGH", "HDP_HSMM", "TICC", "AutoPlait", "ClasPTS_KMeans",
    "TS2Vec", "Time2State", "Proposed(Ours)"
]

# LaTeX表から取得した ARI 値
ari_values = np.array([
    [0.0809, 0.0900, 0.0881, 0.0032, 0.0788],  # HVGH
    [0.6619, 0.5509, 0.6644, 0.2882, 0.4678],  # HDP_HSMM
    [0.6242, 0.7218, 0.7839, 0.3008, 0.3947],  # TICC
    [0.0713, 0.8057, 0.0586, None, 0.2948],    # AutoPlait (N/A は None として扱う)
    [0.2950, 0.5450, 0.2825, 0.1700, 0.5075],  # ClasPTS_KMeans
    [0.8176, 0.7529, 0.7670, 0.3135, 0.6522],  # TS2Vec
    [0.8843, 0.7896, 0.7909, 0.3345, 0.6833],  # Time2State
    [0.8802, 0.7603, 0.7644, 0.3490, 0.7804],  # Proposed(Ours)
])

# バーの設定
bar_width = 0.1
x = np.arange(len(datasets))

# カラーとパターンの設定
colors = [
    'black', 'blue', 'purple', 'gray', 'orange', 
    'green', 'cyan', 'red'
]
hatch_patterns = ['//', '\\\\', 'xx', '--', '..', 'oo', '**', '']

# プロット
fig, ax = plt.subplots(figsize=(12, 6))

for i, (method, color, hatch) in enumerate(zip(methods, colors, hatch_patterns)):
    ari_data = [v if v is not None else 0 for v in ari_values[i]]  # None (N/A) を 0 に置換
    ax.bar(x + i * bar_width, ari_data, bar_width, label=method, color=color, hatch=hatch, edgecolor='black')

# 軸ラベルと凡例
ax.set_xlabel("Datasets")
ax.set_ylabel("ARI")
ax.set_title("Comparison of ARI across Methods")
ax.set_xticks(x + bar_width * (len(methods) / 2))
ax.set_xticklabels(datasets)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# グラフを表示
plt.savefig("bar_ari.png", dpi=300)  # 高解像度で保存
plt.show()
