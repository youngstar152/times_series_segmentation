import numpy as np
import matplotlib.pyplot as plt

# データセットのリスト
datasets = ["Synthetic", "MoCap", "ActRecTut", "PAMAP2", "USC-HAD"]
methods = [
    "HVGH", "HDP_HSMM", "TICC", "AutoPlait", "ClasPTS_KMeans",
    "TS2Vec", "Time2State", "Proposed(Ours)"
]

# LaTeX表から取得した NMI 値
nmi_values = np.array([
    [0.1606, 0.1523, 0.2088, 0.0374, 0.1883],  # HVGH
    [0.7798, 0.7230, 0.6473, 0.5338, 0.6839],  # HDP_HSMM
    [0.7481, 0.7524, 0.7466, 0.5955, 0.7028],  # TICC
    [0.1307, 0.8289, 0.1148, None, 0.5413],    # AutoPlait (N/A は None として扱う)
    [0.4480, 0.6763, 0.2309, 0.5830, 0.6940],  # ClasPTS_KMeans
    [0.8407, 0.7584, 0.7407, 0.5905, 0.8126],  # TS2Vec
    [0.8025, 0.7812, 0.7473, 0.6143, 0.8164],  # Time2State
    [0.8731, 0.7753, 0.7476, 0.6020, 0.8671],  # Proposed(Ours)
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
    nmi_data = [v if v is not None else 0 for v in nmi_values[i]]  # None (N/A) を 0 に置換
    ax.bar(x + i * bar_width, nmi_data, bar_width, label=method, color=color, hatch=hatch, edgecolor='black')

# 軸ラベルと凡例
ax.set_xlabel("Datasets")
ax.set_ylabel("NMI")
ax.set_title("Comparison of NMI across Methods")
ax.set_xticks(x + bar_width * (len(methods) / 2))
ax.set_xticklabels(datasets)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# グラフを表示
plt.savefig("bar_nmi.png", dpi=300)  # 高解像度で保存
plt.show()
