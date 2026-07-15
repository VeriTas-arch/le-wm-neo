import torch
import h5py
import numpy as np
import hydra
from omegaconf import open_dict
import os
from pathlib import Path

# 1. 自动寻找最新生成的权重
ckpt_dir = Path("/root/.cache/stable-pretraining/runs/")
all_runs = sorted(ckpt_dir.glob("**/checkpoints/*.ckpt"), key=os.path.getmtime)
if not all_runs:
    print("未找到任何权重文件。")
    exit()

latest_ckpt = str(all_runs[-1])
print(f"正在加载最新权重: {latest_ckpt}")

# 2. 还原模型结构与参数
with hydra.initialize(version_base=None, config_path="config/train"):
    cfg = hydra.compose(config_name="lewm", overrides=["data=wm_maze"])
    with open_dict(cfg):
        cfg.model.predictor.num_frames = 100  # 完美匹配 100 帧
        cfg.model.action_encoder.input_dim = 1
    
    model = hydra.utils.instantiate(cfg.model).cuda().eval()
    checkpoint = torch.load(latest_ckpt, map_location="cuda")
    state_dict = {k.replace("model.", "").replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict, strict=False)

# 映射字典用于打印
COLOR_MAP = {0: "Red  ", 1: "Blue ", 2: "Green"}
ACT_MAP = {0: "FWD  ", 1: "LEFT ", 2: "RIGHT"}

# 3. 开始深度诊断测试
data_path = "/root/autodl-tmp/.stable_worldmodel/datasets/wm_maze_optimized.h5"
with h5py.File(data_path, 'r') as f:
    # 随机抽取 3 个迷宫进行极限诊断
    indices = np.random.choice(len(f['ep_len']), 3, replace=False)
    
    for idx in indices:
        ep_off = f['ep_offset'][idx]
        ep_len = 100  # 100 帧
        
        pixels = torch.tensor(f['pixels'][ep_off : ep_off+ep_len]).float().cuda() / 255.0
        actions = torch.tensor(f['action'][ep_off : ep_off+ep_len]).cuda().squeeze()
        cues = torch.tensor(f['cue_color'][ep_off : ep_off+ep_len]).cuda()
        
        pixels = pixels.permute(0, 3, 1, 2).unsqueeze(0) 

        with torch.no_grad():
            info = model.encode({"pixels": pixels})
            emb = info["emb"]
            
            # 特征对齐推理
            T = emb.size(1)
            x = emb + model.predictor.pos_embedding[:, :T]
            x = model.predictor.dropout(x)
            gru_out, _ = model.predictor.gru(x)
            gru_out_full = model.predictor.gru_norm(x + gru_out)
            
            # 同时通过两个探针把脑子里的“想法”和手里的“动作”抠出来
            act_logits = model.action_probe(gru_out_full)
            cue_logits = model.cue_probe(gru_out_full)
            
            pred_actions = torch.argmax(act_logits, dim=-1).cpu().numpy().squeeze()
            pred_cues = torch.argmax(cue_logits, dim=-1).cpu().numpy().squeeze()

        # 对齐真实标签
        true_actions = actions[:T].cpu().numpy()
        true_cues = cues[:T].cpu().numpy()

        # 仅关注路口决策
        mask = true_actions != 0
        if mask.sum() > 0:
            act_acc = (pred_actions[mask] == true_actions[mask]).mean()
            cue_acc = (pred_cues[mask] == true_cues[mask]).mean()
            
            print(f"\n==================== 迷宫 Episode {idx} 深度诊断报告 ====================")
            print(f"🌟 记忆准确率 (Cue Acc): {cue_acc:.2%} | 决策准确率 (Act Acc): {act_acc:.2%}")
            print("-" * 75)
            print(f"{'步数 (Step)':<10} | {'真实颜色 (True Cue)':<16} | {'模型记忆 (Pred Cue)':<16} | {'真实动作 (True Act)':<16} | {'模型动作 (Pred Act)':<16}")
            print("-" * 75)
            
            for t in np.where(mask)[0]:
                c_true_str = COLOR_MAP[true_cues[t]]
                c_pred_str = COLOR_MAP[pred_cues[t]]
                a_true_str = ACT_MAP[true_actions[t]]
                a_pred_str = ACT_MAP[pred_actions[t]]
                
                c_ok = "✅" if true_cues[t] == pred_cues[t] else "❌"
                a_ok = "✅" if true_actions[t] == pred_actions[t] else "❌"
                
                print(f"Step {t:03d}    | {c_true_str}            | {c_pred_str} {c_ok}         | {a_true_str}            | {a_pred_str} {a_ok}")
            print("========================================================================\n")
        else:
            print(f"--- 迷宫 Episode {idx} 没有检测到路口 ---")
