import torch
import torch.nn.functional as F

def lejepa_forward_fixed(self, batch, stage, cfg):
    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]

    # ✨ 1. 恢复：LeWM 世界模型的核心，预测未来的 latent
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]

    # 正常调用模型预测
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    # 使用自适应最小长度对齐，防止边缘切片报错，彻底保证 100% 稳定
    pred_len = pred_emb.size(1)
    tgt_len = tgt_emb.size(1)
    min_len = min(pred_len, tgt_len)

    output["pred_loss"] = (pred_emb[:, -min_len:] - tgt_emb[:, -min_len:]).pow(2).mean() if min_len > 0 else torch.tensor(0.0, device=emb.device)
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    B, T, D = emb.shape

    # 2. 特征对齐：手动过一遍 GRU 拿到最干净的记忆特征给探针
    x = emb + self.model.predictor.pos_embedding[:, :T]
    x = self.model.predictor.dropout(x)
    gru_out, _ = self.model.predictor.gru(x)
    gru_out_full = self.model.predictor.gru_norm(x + gru_out)

    # 3. 记忆探针
    cue_logits = self.model.cue_probe(gru_out_full.reshape(B * T, D))
    cue_label = batch["cue_color"].long().reshape(B * T)
    output["cue_loss"] = F.cross_entropy(cue_logits, cue_label)

    # 4. 动作探针
    act_logits = self.model.action_probe(gru_out_full.reshape(B * T, D))
    act_label = batch["action"].squeeze(-1).long().reshape(B * T)

    # 转弯罚 10 倍
    weights = torch.tensor([0.1, 10.0, 10.0], device=act_logits.device)
    output["act_loss"] = F.cross_entropy(act_logits, act_label, weight=weights)

    # ✨ 5. 总 Loss：完美平衡了物理世界的学习和动作的学习
    output["loss"] = (output["pred_loss"]
                      + lambd * output["sigreg_loss"]
                      + 2.0 * output["cue_loss"]
                      + 3.0 * output["act_loss"])

    output["act_acc"] = (act_logits.argmax(-1) == act_label).float().mean()
    output["cue_acc"] = (cue_logits.argmax(-1) == cue_label).float().mean()

    self.log_dict({
        f"{stage}/loss": output["loss"],
        f"{stage}/pred_loss": output["pred_loss"],
        f"{stage}/act_loss": output["act_loss"],
        f"{stage}/act_acc": output["act_acc"],
        f"{stage}/cue_acc": output["cue_acc"]
    }, on_step=True)
    return output
