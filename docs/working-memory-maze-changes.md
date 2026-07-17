# 工作记忆迷宫版本修改说明

本文记录当前工作区相对基线仓库 `origin/main` 的修改。这里的“原版本”特指该 Git 基线，而不是论文中抽象描述的 LeWorldModel。

基线仓库已经包含一些尚未收敛的迷宫实验代码，例如 ViT 编码器、离散动作 embedding、GRU、cue/action probe，以及只处理最后 5 帧的运行时覆盖。本次修改的重点是修正这些实验代码的数据和时间语义，使其形成一条可迁移、可验证、可训练的完整链路。

## 1. 修改目标

当前任务要求模型：

1. 依次观察三个颜色 cue；
2. 经历一段不可见的 delay；
3. 在三个局部视觉相似的路口，根据记住的 cue 选择前进、左转或右转；
4. 同时继续学习下一帧 latent prediction，而不是退化为纯分类器。

颜色与动作的固定映射为：

| Cue | 类别编号 | 正确动作 | 动作编号 |
|---|---:|---|---:|
| red | 0 | left | 1 |
| blue | 1 | right | 2 |
| green | 2 | forward | 0 |

## 2. 当前模型结构

当前模型的数据流为：

```text
64 帧 RGB episode
    │
    ├─ 前 63 帧作为 context
    └─ 后移 1 帧作为 prediction target

context pixels
    ↓
ViT-Tiny frame encoder
    ↓
每帧 192 维 CLS latent
    ↓
位置编码 + 单层 GRU + 残差 LayerNorm
    ↓
完整 63 帧 memory states
    ├─ cue_probe:    Linear(192, 3)
    ├─ action_probe: Linear(192, 3)
    └─ 6 层动作条件因果 Transformer
            ↓
        下一帧 latent prediction
```

ViT、离散动作 embedding、双 probe 和 GRU 在基线中已经存在。本次对模型路径的核心修正位于 `module.py`：

- 删除了文件末尾对 `ARPredictor.forward` 的 monkey-patch；
- 删除硬编码的 `W = 5` Transformer 窗口；
- 新增 `memory_states()`，集中实现完整上下文的 GRU 记忆编码；
- Transformer 现在处理全部 63 个 context token；
- context 与 action condition 长度不一致时立即报错；
- context 超出位置编码容量时立即报错，不再静默产生错误切片。

因此，当前实现不再是“GRU 读完整历史、Transformer 和世界模型损失只看最后 5 帧”，而是完整序列逐位置训练。

## 3. 时间对齐修正

基线训练代码采用了以下逻辑：

```python
ctx_emb = emb[:, :history_size]
tgt_emb = emb[:, num_preds:]
pred_emb = model.predict(ctx_emb, ctx_act)
min_len = min(pred_len, tgt_len)
loss = mse(pred_emb[:, -min_len:], tgt_emb[:, -min_len:])
```

当数据有 100 帧、history 为 60 且 predictor 只返回最后 5 帧时，这会把 context 的第 55～59 帧预测与 target 的第 95～99 帧错误配对。

现在改为严格的同源切片：

```python
ctx_emb = emb[:, :ctx_len]
tgt_emb = emb[:, num_preds:ctx_len + num_preds]
pred_emb = model.predict(ctx_emb, ctx_act)
```

在当前 `num_preds = 1` 下，每个位置优化的是：

```text
(z[t], action[t]) → z[t+1]
```

不再使用 `min_len` 或双方独立取尾部的方式掩盖时间长度错误。

## 4. 数据生成器重写

唯一正式生成入口现为：

```text
generate_wm_maze.py
```

旧版多个生成器存在不同问题，包括：

- 直接把环境切换到 `action`，跳过 cue 和 delay；
- 在加载 maze 前取得第一张 observation，导致首帧为空；
- 未调用 `load_maze()`；
- 完成后重复静止帧，但没有提供 padding mask；
- 不同脚本生成相同文件名，难以判断真实数据来源；
- 路口与动作类别分布不受约束。

新生成器执行以下数据契约：

- 每个 episode 固定 64 帧；
- 每个 episode 只对应一个 dataset clip，避免滑动窗口从 cue 之后开始采样；
- 三个 cue 构成 `[red, blue, green]` 的随机排列；
- 每个 episode 恰好包含 forward、left、right 各一次；
- 默认 cue 每种显示 4 帧；
- delay 从 6～10 帧随机采样；
- episode 未在 64 帧内完成时丢弃并重新生成；
- 保留真实 terminal observation，后续位置才作为 padding；
- HDF5 使用内置 LZF 压缩；
- 随机种子和时序参数写入 HDF5 attributes。

### 4.1 新增数据字段

| 字段 | 形状 | 用途 |
|---|---|---|
| `pixels` | `(N, 224, 224, 3)` | RGB observation |
| `action` | `(N,)` | 专家动作 |
| `cue_color` | `(N,)` | 当前应记忆的 cue |
| `valid_mask` | `(N,)` | 真实帧，排除 padding |
| `transition_mask` | `(N,)` | 存在真实 `t → t+1` 转移的位置 |
| `decision_mask` | `(N,)` | 三个路口决策位置 |
| `memory_mask` | `(N,)` | delay 和 action 阶段的记忆监督位置 |
| `ep_len` | `(episodes,)` | 固定为 64 |
| `ep_offset` | `(episodes,)` | episode 在扁平数组中的偏移 |

旧数据集如果缺少这四个 mask，不能直接用于当前训练入口，需要重新生成。

## 5. 环境与视觉捷径修正

The maze environment is now kept next to its only caller in
`generate_wm_maze.py`. It includes the following changes:

- `reset()` 可以直接接收 maze，确保第一帧同时包含 maze 和首个 cue；
- observation 改为以 agent 为中心、半径为 2 的局部视野；
- 终点渲染为普通地面，不再从全局地图泄漏正确路线；
- 每个任务路口要求两条错误分支都真实存在；
- agent heading 始终根据实际路径移动更新，避免跳过非任务转弯后朝向漂移；
- delay 仍渲染为全黑帧；
- 错误动作立即终止，正确动作推进相应 cue。

这些修改的目标是降低“直接看完整地图或终点猜动作”的捷径，使路口动作更依赖早期 cue。

## 6. Masked 训练目标

The training forward pass was moved out of the temporary `train_fix.py` and is
now defined directly in its only caller:

```text
train.py
```

当前总损失为：

```text
loss = pred_loss
     + 0.09 × sigreg_loss
     + 2.0  × cue_loss
     + 3.0  × act_loss
```

各损失的监督范围为：

| 损失/指标 | 使用的 mask | 说明 |
|---|---|---|
| `pred_loss` | `transition_mask` | 只训练真实状态转移 |
| `sigreg_loss` | `valid_mask` | 排除 padding latent |
| `cue_loss`, `cue_acc` | `memory_mask` | 在 delay/action 阶段检查记忆 |
| `act_loss`, `act_acc` | `decision_mask` | 只在三个路口训练和评估动作 |

这替代了基线中“所有帧都计算 cue/action CE，并用 `[0.1, 10, 10]` 类别权重补偿”的做法。由于新数据每个 episode 的三个动作严格均衡，动作头不再需要极端类别权重。

空 mask 会返回与计算图连接的零损失，避免特殊 batch 导致 `cross_entropy` 报错或产生 NaN。

## 7. 配置变化

`config/train/lewm.yaml` 的主要变化：

| 配置 | 基线 | 当前 | 原因 |
|---|---:|---:|---|
| `history_size` | 60 | 63 | 64 帧 episode 留 1 帧作为 target |
| `loader.batch_size` | 64 | 8 | 完整 63-token Transformer 与逐帧 ViT 更耗显存 |
| `optimizer.lr` | `1e-5` | `1e-4` | 从零训练 ViT 时提供更有效的更新幅度 |
| `trainer.precision` | `bf16` | `bf16-mixed` | 使用 Lightning 推荐的 bf16 AMP 写法 |
| `max_epochs` | 100 | 100 | 未变 |

另外设置 `stop_after_epoch: 10`。这不是基于验证指标 patience 的 early
stopping，而是论文式的固定 checkpoint 选择：Trainer 和学习率调度器仍以
100 epoch 为上限构造，但第 10 个 epoch 完成并保存
`weights_epoch_10.pt` 后，回调设置 `trainer.should_stop = True`。

不能简单把 `trainer.max_epochs` 改成 10，因为这会同时把 cosine learning-rate
schedule 从 100 epoch 压缩到 10 epoch，与“按 100 epoch 调度、使用 epoch 10”
不是同一训练过程。若要关闭固定停止并完整训练 100 epoch，可运行：

```bash
python train.py data=wm_maze stop_after_epoch=null
```

`config/train/data/wm_maze.yaml` 现在：

- 使用 `num_steps = history_size + num_preds = 64`；
- 数据名改为 `wm_maze.h5`；
- 显式指定 `format: hdf5`；
- 加载并缓存四类 mask。

当前根配置仍保留基线默认值 `data: pusht`，因此迷宫训练必须显式运行：

```bash
python train.py data=wm_maze
```

## 8. 可迁移路径与运行环境

删除了 `/root/autodl-tmp/...`、`/root/.cache/...` 等机器绑定路径。

新增 `.envrc.example`：

```bash
use conda wm
export STABLEWM_HOME="$PWD/data"
export SPT_CACHE_DIR="$STABLEWM_HOME/cache/stable-pretraining"
export MPLCONFIGDIR="$STABLEWM_HOME/cache/matplotlib"
mkdir -p "$SPT_CACHE_DIR" "$MPLCONFIGDIR"
```

初始化方式：

```bash
cp .envrc.example .envrc
direnv allow
```

默认目录布局：

```text
$STABLEWM_HOME/
├── datasets/wm_maze.h5
├── checkpoints/wm_maze/weights_epoch_*.pt
└── cache/stable-pretraining/
```

如果未设置环境变量，代码回退到仓库内的 `data/`。

## 9. 本地 HDF5 reader

新增 `hdf5_dataset.py`。

原因是部分 `stable_worldmodel` 版本只有在安装 `hdf5plugin` 后才能注册 HDF5 backend，而当前数据只使用 h5py 自带的 LZF，并不需要该额外插件。

本地 reader：

- 继承 `stable_worldmodel.data.dataset.Dataset`；
- 保留 episode/clip 索引语义；
- 支持 lazy worker-local HDF5 handle；
- 支持 `keys_to_load` 和 `keys_to_cache`；
- 自动把 NHWC pixels 转成 NCHW；
- 与现有 DataLoader 和 transform 链路兼容。

## 10. 训练入口变化

`train.py` 现在：

- 从 `STABLEWM_HOME` 解析 dataset 和 checkpoint 根目录；
- 同步设置 stable-worldmodel 与 stable-pretraining 的输出位置；
- HDF5 maze 数据使用本地 reader；
- 不再把 `*_mask` 当作连续状态做 z-score；
- uses the local `wm_maze_forward` implementation in `train.py`;
- 设置 `torch.set_float32_matmul_precision("high")`，利用 GPU Tensor Core；
- 保留每 epoch 权重导出和 Lightning checkpoint。

验证默认每个 epoch 执行一次，因此完整 Metric 表每个 epoch 输出一次；训练开始前还会由 `num_sanity_val_steps=1` 输出一次 sanity-validation 结果。

每个正式 validation epoch 还会通过 `WMMazeValidationVideo` 导出首个验证
episode。视频显示真实/预测 cue、预测置信度、真实/预测动作以及 decision 正误；
同名 CSV 保存逐帧预测。默认路径为：

```text
$STABLEWM_HOME/validation/epoch_NNN_episode_00.mp4
$STABLEWM_HOME/validation/epoch_NNN_episode_00.csv
```

sanity validation 不导出视频。频率由以下配置控制：

```yaml
validation_video:
  enabled: true
  every_n_epochs: 1
  fps: 4.0
  sample_index: 0
  padding: 128
  video_preset: standard
```

`standard` uses 1x output with H.264 CRF 18. `report` uses 2x output,
CRF 12, and the slow encoder preset; select it with
`--video-preset report` for presentation material.

## 11. 新增正式工具

### 11.1 数据验证

```bash
python validate_wm_maze.py
```

检查内容包括：

- 必需 HDF5 字段；
- 固定 episode 长度；
- 每 episode 恰好三个 decision；
- 三类动作各一次；
- cue/action 映射；
- mask 的包含关系；
- terminal transition；
- cue 帧和全黑 delay 帧是否存在。

默认还会把 episode 0 导出为带标注 MP4，并写出同名 CSV。视频逐帧显示
phase、真实 cue、真实动作和 decision 标记；CSV 保存对应的 cue 时间线。可使用：

```bash
python validate_wm_maze.py --episode 12 --fps 4
python validate_wm_maze.py --no-video
```

### 11.2 Probe 评估

```bash
python eval_wm_maze.py \
  "$STABLEWM_HOME/checkpoints/wm_maze/weights_epoch_10.pt"
```

输出：

- decision action accuracy；
- decision cue accuracy；
- 全部 memory frame 的 cue accuracy。

训练日志中的 `cue_acc` 覆盖所有 `memory_mask` 帧，而不是只看路口，因此新增的 decision cue accuracy 更适合判断模型在真正决策时是否记得 cue。

### 11.3 从固定 checkpoint 导出正式验证视频

```bash
python eval_wm_maze.py \
  "$STABLEWM_HOME/checkpoints/wm_maze/weights_epoch_10.pt" \
  --video-only \
  --padding 24
```

该命令使用与训练相同的 `seed` 和 train/validation 比例重建 validation split，
默认选择第一个验证样本。输出到：

```text
$STABLEWM_HOME/validation/formal_epoch_010_validation_0000.mp4
$STABLEWM_HOME/validation/formal_epoch_010_validation_0000.csv
```

可用 `--video-index` 选择其他验证样本、`--fps` 调整帧率、
`--padding` 调整画布外边距，或用 `--video-output` 指定 MP4 路径。
CSV 路径始终与 MP4 同名。

训练回调和数据验证器共用 `H264VideoWriter`：OpenCV 只负责写临时帧流，
随后由 FFmpeg 转码为 H.264、`yuv420p` 并设置 `faststart`，最后原子替换目标
MP4。这样生成的视频可以直接在 VS Code/Chromium 中预览。若 FFmpeg 缺失或
编码失败，导出会明确失败，不会留下伪装成可用结果的 MPEG-4 Part 2 文件。
默认还会在内容四周增加 24 px 白色 padding，从而扩大画布而不拉伸迷宫或
文字。训练视频通过 `validation_video.padding` 配置，数据验证视频通过
`--padding` 调整。

## 12. 仓库清理

删除了重复、失效或明显属于临时调试的内容：

- `gen_data_equal100.py`
- `gen_data_optimized.py`
- `generate_data.py`
- `maze_generator.py`
- `mem_maze_wrapper.py`
- `module_patch.py`
- `quick_check.py`
- `quick_test.py`
- `run_inference_test.py`
- `verify_new_dataset.py`
- `visualize_dataset.py`
- `wm_maze_env.py` (merged into `generate_wm_maze.py`)
- `wm_maze_training.py` (merged into `train.py`)
- `video_export.py` (merged into `wm_maze_video.py`)
- notebook checkpoint 配置副本
- 临时数据预览图片

必要功能已分别合并到 `generate_wm_maze.py`、`validate_wm_maze.py` 和 `eval_wm_maze.py`。

## 13. 兼容性说明

### 13.1 旧数据集

旧 `wm_maze_optimized.h5` 通常缺少 mask，且可能跳过 cue/delay，不能用于当前训练。应重新生成：

```bash
python generate_wm_maze.py --episodes 5000
python validate_wm_maze.py
```

### 13.2 旧 checkpoint

旧 checkpoint 可能具有长度为 60 的位置编码，而当前 predictor 使用 63。因此不能保证严格加载兼容。建议用当前数据和配置重新训练；如果必须迁移，需要单独处理 `predictor.pos_embedding`。

### 13.3 原有 PushT/TwoRoom 训练

当前 `JEPA` 使用三类离散动作 embedding，训练 forward 又要求 maze mask，因此这条修改后的 `train.py` 已专用于工作记忆迷宫。README 中的其他任务内容主要保留为上游项目背景，不应假设它们仍能直接使用当前 maze forward。

## 14. 已完成验证

本次修改已执行：

- 全部 Python 文件语法检查；
- Hydra maze 配置解析；
- 小型 HDF5 数据生成与本地 reader 加载；
- 每 episode 三类动作平衡检查；
- masked loss 合成前向/反向；
- RTX 5090、CUDA 13.0、bf16 可用性检查；
- CPU 两个 optimizer step；
- RTX 5090 上两个 bf16 optimizer step；
- 全部约 1650 万参数首个 backward 均收到梯度；
- 权重导出和 Lightning checkpoint 写入验证；
- `eval_wm_maze.py` checkpoint 加载与 probe 评估。

这些 smoke test 证明训练链路可以运行，但不代表模型已经收敛。最终工作记忆能力仍应以验证集 action accuracy、decision cue accuracy 和固定迷宫更换 cue 的反事实测试为准。

## 15. 当前标准工作流

```bash
# 1. 进入环境
direnv allow

# 2. 生成并验证数据
python generate_wm_maze.py --episodes 5000
python validate_wm_maze.py

# 3. GPU 训练
python train.py data=wm_maze

# 4. 评估某个导出权重
python eval_wm_maze.py \
  "$STABLEWM_HOME/checkpoints/wm_maze/weights_epoch_10.pt"
```
