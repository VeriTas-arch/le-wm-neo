import h5py
import numpy as np
import matplotlib.pyplot as plt

data_path = "/root/autodl-tmp/.stable_worldmodel/datasets/wm_maze_optimized.h5"
COLOR_MAP = {0: "Red", 1: "Blue", 2: "Green"}
ACT_MAP = {0: "FWD", 1: "LEFT", 2: "RIGHT"}

print(f"正在打开文件: {data_path}")
with h5py.File(data_path, 'r') as f:
    # 1. 宏观统计
    actions_all = f['action'][:]
    total_steps = len(actions_all)
    turns = np.sum(actions_all != 0)
    print("\n========== 宏观数据统计 ==========")
    print(f"总帧数: {total_steps}")
    print(f"直行 (Action 0): {np.sum(actions_all == 0)} 帧")
    print(f"左转 (Action 1): {np.sum(actions_all == 1)} 帧")
    print(f"右转 (Action 2): {np.sum(actions_all == 2)} 帧")
    print(f"转弯占比: {turns / total_steps:.2%}")
    print("==================================\n")

    # 2. 微观随机抽查 1 个 Episode
    ep_idx = np.random.randint(0, len(f['ep_len']))
    off = f['ep_offset'][ep_idx]
    length = f['ep_len'][ep_idx] # 这里应该是100

    pixels = f['pixels'][off : off+length]
    actions = f['action'][off : off+length]
    cues = f['cue_color'][off : off+length]

    turn_indices = np.where(actions != 0)[0]

    print(f"========== Episode {ep_idx} 抽查 ==========")
    print(f"迷宫总长度: {length} 帧")
    print(f"开局提示颜色: {COLOR_MAP[cues[0]]}")

    if len(turn_indices) == 0:
        print("警告：这局迷宫没有转弯！请重新运行脚本抽查另一局。")
    else:
        for t in turn_indices:
            print(f"第 {t:03d} 帧 | 画面到达路口 | 要求记忆颜色: {COLOR_MAP[cues[t]]} | 专家执行动作: {ACT_MAP[actions[t]]}")

    # 3. 提取关键帧画图 (开局第0帧 + 所有转弯帧)
    plot_indices = [0] + list(turn_indices)
    # 最多画 6 张图，防止挤不下
    plot_indices = plot_indices[:6]

    fig, axes = plt.subplots(1, len(plot_indices), figsize=(4 * len(plot_indices), 4))
    if len(plot_indices) == 1: axes = [axes]

    for ax, idx in zip(axes, plot_indices):
        ax.imshow(pixels[idx])
        title_color = "red" if actions[idx] != 0 else "black"
        ax.set_title(f"Step {idx}\nCue: {COLOR_MAP[cues[idx]]}\nAct: {ACT_MAP[actions[idx]]}",
                     color=title_color, fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    save_path = "/root/le-wm-main/dataset_check.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n✅ 可视化图片已保存至: {save_path}")
    print("请在 AutoDL 网页端双击打开 dataset_check.png 查看画面！")
