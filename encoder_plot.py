import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_box(ax, x, y, width, height, text, color="lightblue", fontsize=12):
    """矩形ボックスを描画"""
    rect = mpatches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.2", 
                                   edgecolor="black", facecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, text, fontsize=fontsize, ha="center", va="center", fontweight="bold")

def draw_arrow(ax, x1, y1, x2, y2):
    """矢印を描画"""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), 
                arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

fig, ax = plt.subplots(figsize=(10, 16))
ax.set_xlim(-2, 7)
ax.set_ylim(-20, 1)
ax.axis("off")

# --- 入力層 ---
draw_box(ax, 2, -1, 3, 0.8, "Inputs (B, T, C)", "lightpink", fontsize=14)
draw_arrow(ax, 3.5, -1.7, 3.5, -2)

draw_box(ax, 2, -2, 3, 0.6, "Positional Encoding", "lightgray")
draw_arrow(ax, 3.5, -2.6, 3.5, -3)

draw_box(ax, 1.5, -3, 4, 1.5, "Feature Extractor\n(DilatedConvEncoder)", "lightblue", fontsize=14)
draw_arrow(ax, 3.5, -4.5, 3.5, -5)

# --- DilatedConvEncoder の詳細部分 ---
draw_box(ax, 1.5, -5, 4, 1, "ConvBlock 1: \nDilated Conv (Kernel=3, Dilation=1)", "lightblue")
draw_arrow(ax, 3.5, -6, 3.5, -7)

draw_box(ax, 1.5, -7, 4, 1, "ConvBlock 2: \nDilated Conv (Kernel=3, Dilation=2)", "lightblue")
draw_arrow(ax, 3.5, -8, 3.5, -9)

draw_box(ax, 1.5, -9, 4, 1, "ConvBlock 3: \nDilated Conv (Kernel=3, Dilation=4)", "lightblue")
draw_arrow(ax, 3.5, -10, 3.5, -11)

draw_box(ax, 1.5, -11, 4, 1, "ConvBlock 4: \nDilated Conv (Kernel=3, Dilation=8)", "lightblue")
draw_arrow(ax, 3.5, -12, 3.5, -13)

# --- 残差接続 ---
draw_box(ax, 2, -13, 3, 0.6, "Residual Connection", "lightyellow")
draw_arrow(ax, 3.5, -13.6, 3.5, -14)

# --- 周期性補正 Conv1D ---
draw_box(ax, 1.5, -14, 4, 0.8, "Conv1D (Periodicity)", "lightblue")
draw_arrow(ax, 3.5, -14.8, 3.5, -15)

# --- 出力処理 ---
draw_box(ax, 2, -15, 3, 0.6, "Dropout (0.1)", "lightgray")
draw_arrow(ax, 3.5, -15.6, 3.5, -16)

draw_box(ax, 2, -16, 3, 0.8, "Encoded Output (B, T, Co)", "lightpink", fontsize=14)
plt.savefig("diagram.png", dpi=300, bbox_inches="tight")
plt.show()
