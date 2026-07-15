import numpy as np, h5py
from pathlib import Path
from tqdm import tqdm
from wm_maze_env import WMMazeEnv, H_DELTA, TURN_TO_COLOR

# ── 昨天你贴的那个完美的迷宫生成器 ──
def turn_heading(h, d): 
    return h if d == 0 else ((h - 1) % 4 if d == 1 else (h + 1) % 4)

def gen_maze(rng=None):
    if rng is None: rng = np.random.default_rng()
    for attempt in range(3000):
        g = np.ones((15, 15), dtype=np.int32)
        path = []; sr = rng.integers(2, 13); path.append((sr, 0)); g[sr, 0] = 0; r, c = sr, 0
        direction = (0, 1)
        while c < 14:
            if direction == (0, 1):
                dist = rng.integers(2, 6)
                for _ in range(dist):
                    if c < 14: c += 1; path.append((r, c)); g[r, c] = 0
            else:
                dist = rng.integers(2, 5)
                for _ in range(dist):
                    if 1 <= r + direction[0] <= 13: r += direction[0]; path.append((r, c)); g[r, c] = 0
                    else: break
            if c >= 14: break
            if direction == (0, 1):
                dirs = []
                if r > 3: dirs.append((-1, 0))
                if r < 11: dirs.append((1, 0))
                direction = dirs[rng.choice(len(dirs))] if dirs else (0, 1)
            else: direction = (0, 1)
                
        if c < 14: continue 
        g[r, c] = 2 
        
        d0 = (path[1][0] - path[0][0], path[1][1] - path[0][1]); start_heading = 1
        for h, (hr, hc) in H_DELTA.items():
            if (hr, hc) == d0: start_heading = h

        inters, last_inter_idx = [], -5 
        for i in range(1, len(path)-1):
            if i - last_inter_idx < 3: continue  
            di = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            do = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            if di == do and rng.random() > 0.15: continue
                
            hi = ho = None
            for h, (hr, hc) in H_DELTA.items():
                if (hr, hc) == di: hi = h
                if (hr, hc) == do: ho = h
            if hi is None or ho is None: continue
            
            if ho == hi: ct = 0
            elif ho == (hi - 1) % 4: ct = 1
            elif ho == (hi + 1) % 4: ct = 2
            else: continue
                
            r2, c2 = path[i]; fake_branches_dug = 0 
            for wt in [0, 1, 2]:
                if wt == ct: continue 
                wh = turn_heading(hi, wt); dr, dc = H_DELTA[wh]
                branch_dug = False
                for step in range(1, rng.integers(3, 6)):
                    nr, nc = r2 + dr*step, c2 + dc*step
                    if 1 <= nr < 14 and 1 <= nc < 14:
                        if g[nr, nc] == 1:
                            zeros = sum(1 for ar, ac in [(-1,0),(1,0),(0,-1),(0,1)] if g[nr+ar, nc+ac] == 0)
                            if zeros > 1: break
                            g[nr, nc] = 0; branch_dug = True
                        else: break
                    else: break
                if branch_dug: fake_branches_dug += 1
            if fake_branches_dug > 0: inters.append((i, ct)); last_inter_idx = i
            
        if len(inters) < 2: continue
        ok = True
        for rr in range(14):
            for cc in range(14):
                if g[rr][cc]==0 and g[rr+1][cc]==0 and g[rr][cc+1]==0 and g[rr+1][cc+1]==0: ok=False
        if not ok: continue
        return {"grid": g.tolist(), "path": path, "intersections": inters, "start_heading": start_heading, "cue_sequence": [TURN_TO_COLOR[t] for _, t in inters]}
    raise RuntimeError("Failed to generate a valid maze.")


# ── 生成 HDF5 数据集 (完美锁定 100 帧) ──
TOTAL = 5000
TARGET_LEN = 100
CUE_DUR = 10
DELAY_MIN, DELAY_MAX = 2, 10
OUT_DIR = Path("/root/autodl-tmp/.stable_worldmodel/datasets")
out_path = OUT_DIR / "wm_maze_optimized.h5"

rng = np.random.default_rng(42)
COLOR_TO_IDX = {"red": 0, "blue": 1, "green": 2}

episode_data = []
print("Pass 1: 生成基于完美算法的 100 帧等长迷宫...")
for ep in tqdm(range(TOTAL)):
    m = gen_maze(rng=rng)
    cue = m['cue_sequence']
    delay = int(rng.integers(DELAY_MIN, DELAY_MAX+1))
    
    env = WMMazeEnv(cue_sequence=cue, cue_duration=CUE_DUR, delay_duration=delay, seed=ep)
    obs, _ = env.reset()
    env.load_maze(m)  # <--- 昨天最聪明的这一步，终于回来了！
    env.phase = "action"; env.border = None
    
    frames, acts, cues = [], [], []
    done = False
    
    for step in range(TARGET_LEN):
        if not done:
            frames.append(obs.copy())
            act = 0
            for ii, (pi, _) in enumerate(env.intersections):
                if env.path_idx == pi and env.inter_done == ii: act = m['intersections'][ii][1]
            acts.append(act)
            
            # 动态 Cue 记录
            if env.phase == "cue":
                active_cue = COLOR_TO_IDX[env.cue_sequence[env.cue_i]]
            else:
                target_idx = min(env.inter_done, len(m['cue_sequence']) - 1)
                active_cue = COLOR_TO_IDX[m['cue_sequence'][target_idx]]
            cues.append(active_cue)
            
            try:
                obs, r, done, _, info = env.step(act)
            except IndexError:
                done = True
        else:
            # 走完了，原地复制发呆，撑满 100 帧
            frames.append(frames[-1])
            acts.append(0)
            cues.append(cues[-1])
            
    episode_data.append((frames, acts, cues))

print("Pass 2: 写入 HDF5...")
ep_len_arr = np.array([len(f) for f, _, _ in episode_data], dtype=np.int64)
ep_off_arr = np.zeros_like(ep_len_arr)
ep_off_arr[1:] = np.cumsum(ep_len_arr[:-1])
total_frames = int(ep_len_arr.sum())

with h5py.File(out_path, 'w') as f:
    pix_ds = f.create_dataset('pixels', shape=(total_frames, 224, 224, 3), dtype=np.uint8, compression='lzf', chunks=(1, 224, 224, 3))
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
print("完美生成！")
