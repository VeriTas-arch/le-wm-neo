import h5py
import numpy as np

data_path = "/root/autodl-tmp/.stable_worldmodel/datasets/wm_maze_15x15_v2.h5"

print(f"正在打开文件: {data_path}")
with h5py.File(data_path, 'r') as f:
    actions = f['action'][:] # 读取全部动作
    total_steps = len(actions)
    
    # 统计转弯动作
    turns = np.sum(actions != 0)
    print(f"总帧数: {total_steps}")
    print(f"转弯总数 (Action != 0): {turns}")
    print(f"直行总数 (Action == 0): {np.sum(actions == 0)}")
    
    if turns == 0:
        print("致命错误：数据集中没有任何转弯动作！")
    else:
        print(f"转弯占比: {turns / total_steps:.4%}")
        
    # 查看一下前 100 个动作的分布
    print("前100个动作分布:", np.bincount(actions[:100].astype(int)))
