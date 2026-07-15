import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from wm_maze_env import WMMazeEnv, TEMPLATES, COLORS_RGB

# 1. 参数优化：更短的 Delay 和更密集的记忆要求
NUM_EPISODES = 5000
MAX_STEPS = 200
CUE_DURATION = 10
DELAY_MIN, DELAY_MAX = 2, 10

# 2. 路径修复：存入大容量数据盘
OUTPUT_DIR = Path("/root/autodl-tmp/.stable_worldmodel/datasets")
OUTPUT_NAME = "wm_maze_15x15_optimized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUTPUT_DIR / (OUTPUT_NAME + ".h5")

rng = np.random.default_rng(42)
colors = list(COLORS_RGB.keys())[:3]
COLOR_TO_IDX = {"red": 0, "blue": 1, "green": 2}

print("Pass 1: generating episodes...")
episode_data = []

for ep in tqdm(range(NUM_EPISODES)):
    cue = list(rng.choice(colors, size=3, replace=False))
    delay = int(rng.integers(DELAY_MIN, DELAY_MAX + 1))
    tmpl = int(rng.integers(len(TEMPLATES)))

    env = WMMazeEnv(cue_sequence=cue, cue_duration=CUE_DURATION,
                    delay_duration=delay, template_idx=tmpl,
                    seed=int(rng.integers(2**31)))
    obs, _ = env.reset()
    frames, acts, cue_labels = [], [], []
    done = False

    # 动态记忆映射逻辑
    for step in range(MAX_STEPS):
        frames.append(obs)
        act = 0
        for ii, (pi, correct) in enumerate(env.intersections):
            if env.path_idx == pi and env.inter_done == ii:
                act = correct
        acts.append(act)

        # ✨ 关键：根据路口进度切换 Cue 标签
        if env.phase == "cue":
            active_cue = COLOR_TO_IDX[env.cue_sequence[env.cue_i]]
        else:
            # 根据已过路口数切换，逼迫模型学习“颜色切换”序列
            idx = min(env.inter_done, len(cue)-1)
            active_cue = COLOR_TO_IDX[cue[idx]]
        cue_labels.append(active_cue)

        obs, r, done, _, info = env.step(act)
        if done:
            # 补齐终止帧
            frames.append(obs)
            acts.append(0)
            cue_labels.append(active_cue)
            break

    episode_data.append((frames, acts, cue_labels))

# Pass 2: 写入 HDF5 (使用 lzf 压缩，速度飞快)
print("Pass 2: writing HDF5...")
ep_len_arr = np.array([len(f) for f, _, _ in episode_data], dtype=np.int64)
ep_off_arr = np.zeros_like(ep_len_arr)
ep_off_arr[1:] = np.cumsum(ep_len_arr[:-1])
total_frames = int(ep_len_arr.sum())

with h5py.File(out_path, 'w') as f:
    pix_ds = f.create_dataset('pixels', shape=(total_frames, 224, 224, 3), dtype=np.uint8,
                               compression='lzf', chunks=(1, 224, 224, 3))
    act_ds = f.create_dataset('action', shape=(total_frames,), dtype=np.int32)
    cue_ds = f.create_dataset('cue_color', shape=(total_frames,), dtype=np.int32)
    f.create_dataset('ep_len', data=ep_len_arr)
    f.create_dataset('ep_offset', data=ep_off_arr)

    offset = 0
    for frames, acts, cues in tqdm(episode_data):
        n = len(frames)
        pix_ds[offset:offset+n] = np.stack(frames)
        act_ds[offset:offset+n] = np.array(acts, dtype=np.int32)
        cue_ds[offset:offset+n] = np.array(cues, dtype=np.int32)
        offset += n

print(f"Done! 存放在: {out_path}")