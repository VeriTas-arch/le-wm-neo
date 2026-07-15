import h5py
import numpy as np
import matplotlib.pyplot as plt

# 配置映射字典，方便人类阅读
COLOR_MAP = {0: "Red", 1: "Blue", 2: "Green"}
ACT_MAP = {0: "Forward", 1: "Left", 2: "Right"}

data_path = "/root/autodl-tmp/.stable_worldmodel/datasets/wm_maze_optimized.h5"

print(f"正在读取数据集: {data_path}")
with h5py.File(data_path, 'r') as f:
    # 随机挑一个 Episode (比如第 42 个)
    ep_idx = np.random.randint(0, len(f['ep_len']))
    off = f['ep_offset'][ep_idx]
    length = f['ep_len'][ep_idx]  # 应该是 100

    pixels = f['pixels'][off : off+length]
    actions = f['action'][off : off+length]
    cues = f['cue_color'][off : off+length]

print(f"\n========== Episode {ep_idx} 时间线 ==========")
# 找到所有发生转弯的帧
turn_indices = np.where(actions != 0)[0]

print(f"开局提示颜色: {COLOR_MAP[cues[0]]}")
for t in turn_indices:
    print(f"第 {t:03d} 帧 | 画面到了路口 | 此时要求记忆: {COLOR_MAP[cues[t]]} | 专家动作: {ACT_MAP[actions[t]]}")
print("===========================================\n")

# ---- 开始画图 ----
# 我们挑 8 张最具代表性的图：前2张是开局和发呆，后面是各个路口瞬间
sample_indices = [0, 15] + list(turn_indices)
# 如果路口太多，最多只画 8 张
sample_indices = sample_indices[:8]

fig, axes = plt.subplots(1, len(sample_indices), figsize=(4 * len(sample_indices), 4))
if len(sample_indices) == 1: axes = [axes] # 兼容单图情况

for ax, idx in zip(axes, sample_indices):
    ax.imshow(pixels[idx])
    title_color = "red" if actions[idx] != 0 else "black"
    ax.set_title(f"Step {idx}\nCue: {COLOR_MAP[cues[idx]]}\nAct: {ACT_MAP[actions[idx]]}", color=title_color, fontsize=14)
    ax.axis('off')

plt.tight_layout()
save_path = "/root/le-wm-main/dataset_preview.png"
plt.savefig(save_path, bbox_inches='tight')
print(f"✅ 可视化图片已保存至: {save_path}")
print("请在 AutoDL 网页端的文件目录里，双击打开 dataset_preview.png 查看画面！")
