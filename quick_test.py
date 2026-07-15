import torch
import h5py
from jepa import JEPA # 确保在当前目录下能找到 jepa.py

# 1. 设置路径
ckpt_path = "/root/.cache/stable-pretraining/runs/20260713/232409/5ab829169e7f/checkpoints/epoch=2-step=3804.ckpt"
data_path = "/root/autodl-tmp/.stable_worldmodel/datasets/wm_maze_15x15_v2.h5"

# 2. 手动加载权重
print("Loading checkpoint...")
checkpoint = torch.load(ckpt_path, map_location="cuda")

# 3. 这里的关键：如果你的训练代码是 PyTorch Lightning
# 我们可以尝试直接提取 state_dict
state_dict = checkpoint['state_dict']

# 4. 打印 keys 看看模型参数名，确认如何加载
print("First 5 keys in state_dict:")
print(list(state_dict.keys())[:5])

# 5. 读取数据验证一下维度
with h5py.File(data_path, 'r') as f:
    pixels = f['pixels'][0:5]
    print(f"Data shape from HDF5: {pixels.shape}")
