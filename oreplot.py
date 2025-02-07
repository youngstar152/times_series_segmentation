import numpy as np
import matplotlib.pyplot as plt

# 📌 データの設定
dimensions = np.array([-1, 0, 1, 2, 4])
time_cpu = np.array([0.63,0.67,0.68,0.69,np.nan])  # Time2State-CPU
time_gpu = np.array([0.72,0.74,0.75,0.76,np.nan])   # Time2State-GPU
time_ticc = np.array([0.63,0.68,0.72,0.72,0.71])   # TICC
time_autoplait = np.array([0.257,0.315,0.346,0.367,np.nan])  # AutoPlait
time_hvgh = np.array([0.10,0.11,0.111,0.159,np.nan])  # HVGH
time_hdp_hsmm = np.array([0.291,0.337,0.404,0.439,0.478])   # HDP-HSMM
#time_clasp =np.array([0.63,0.67,0.68,0.69,np.nan])  # ClaSP

# 📌 色とマーカーの設定
methods = [
    ("K-Means(Sliding Window)", time_cpu, "o", "tab:red"),
    ("GMM(Sliding Window)", time_gpu, "^", "tab:blue"),
    ("DPGMM(Sliding Window)", time_ticc, "s", "tab:purple"),
    ("K-Means(All)", time_autoplait, "D", "tab:gray"),
    ("GMM(All)", time_hvgh, "v", "tab:orange"),
    ("DPGMM(All)", time_hdp_hsmm, "<", "tab:green"),
    # ("ClaSP", time_cla#p, "x", "pink")
]

# 📌 図の準備
fig, ax = plt.subplots(figsize=(8, 5))

# 📌 各手法のプロット
for name, time, marker, color in methods:
    ax.plot(dimensions, time, marker=marker, linestyle="-", label=name, color=color, alpha=0.8)
    
# 📌 "Refused to work" の注釈
refused_x = 4
refused_y = 0.8
ax.text(refused_x, refused_y, "", fontsize=12, fontweight="bold", ha="center")
#ax.scatter(refused_x, refused_y, marker="x", color="black", s=100)

# 📌 軸ラベル・タイトル
ax.set_xlabel("Number of detected clustering states", fontsize=12)
ax.set_ylabel("NMI", fontsize=12)

# 📌 凡例の設定
ax.legend(loc="lower right", fontsize=10, frameon=True)

# 📌 グリッド追加
ax.grid(True, linestyle="dashed", alpha=0.5)

# 📌 LaTeX 風のタイトル
plt.title("NMI(Sliding Window VS All) based on the number of detected states", fontsize=14)

# 📌 仕上げ
plt.tight_layout()
plt.savefig("NMI(SlidingWindowVSAll).png", dpi=300)  # 高解像度で保存
plt.show()
