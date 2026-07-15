import torch
import torch.nn.functional as F


def _masked_mean(values, mask):
    mask = mask.to(device=values.device, dtype=torch.bool)
    if not mask.any():
        return values.sum() * 0.0
    return values[mask].mean()


def _masked_cross_entropy(logits, labels, mask):
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if not mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], labels[mask])


def lejepa_forward_fixed(self, batch, stage, cfg):
    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]

    if emb.size(1) < ctx_len + n_preds:
        raise ValueError(
            f"batch has {emb.size(1)} frames, but training requires "
            f"history_size + num_preds = {ctx_len + n_preds}"
        )

    # Predict z[t + n_preds] from the state/action at t.  Both slices use
    # the same time origin; never align unrelated tails by minimum length.
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds : ctx_len + n_preds]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    transition_mask = batch["transition_mask"][:, :ctx_len].bool()
    pred_error = (pred_emb - tgt_emb).pow(2).mean(dim=-1)
    output["pred_loss"] = _masked_mean(pred_error, transition_mask)

    valid_mask = batch["valid_mask"][:, :ctx_len].bool()
    valid_emb = ctx_emb[valid_mask]
    output["sigreg_loss"] = (
        self.sigreg(valid_emb.unsqueeze(1))
        if valid_emb.numel()
        else ctx_emb.sum() * 0.0
    )

    memory = self.model.predictor.memory_states(ctx_emb)
    cue_logits = self.model.cue_probe(memory)
    cue_label = batch["cue_color"][:, :ctx_len].long()
    memory_mask = batch["memory_mask"][:, :ctx_len].bool()
    output["cue_loss"] = _masked_cross_entropy(cue_logits, cue_label, memory_mask)

    act_logits = self.model.action_probe(memory)
    act_label = batch["action"][:, :ctx_len].squeeze(-1).long()
    decision_mask = batch["decision_mask"][:, :ctx_len].bool()
    output["act_loss"] = _masked_cross_entropy(act_logits, act_label, decision_mask)

    output["loss"] = (
        output["pred_loss"]
        + lambd * output["sigreg_loss"]
        + 2.0 * output["cue_loss"]
        + 3.0 * output["act_loss"]
    )

    output["act_acc"] = _masked_mean(
        (act_logits.argmax(-1) == act_label).float(), decision_mask
    )
    output["cue_acc"] = _masked_mean(
        (cue_logits.argmax(-1) == cue_label).float(), memory_mask
    )

    self.log_dict(
        {
            f"{stage}/loss": output["loss"],
            f"{stage}/pred_loss": output["pred_loss"],
            f"{stage}/sigreg_loss": output["sigreg_loss"],
            f"{stage}/cue_loss": output["cue_loss"],
            f"{stage}/act_loss": output["act_loss"],
            f"{stage}/act_acc": output["act_acc"],
            f"{stage}/cue_acc": output["cue_acc"],
        },
        on_step=True,
    )
    return output
