import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from torch.nn.utils.rnn import pad_sequence

from module import SIGReg
from utils import (
    SaveCkptCallback,
    StopAfterEpoch,
    get_column_normalizer,
    get_img_preprocessor,
)

from wm_maze_video import WMMazeValidationVideo


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


def wm_maze_forward(self, batch, stage, cfg):
    """Train aligned latent prediction and masked maze probes."""
    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb, act_emb = output["emb"], output["act_emb"]
    if emb.size(1) < ctx_len + n_preds:
        raise ValueError(
            f"batch has {emb.size(1)} frames, but training requires "
            f"history_size + num_preds = {ctx_len + n_preds}"
        )

    ctx_emb = emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds : ctx_len + n_preds]
    pred_emb = self.model.predict(ctx_emb, act_emb[:, :ctx_len])
    transition_mask = batch["transition_mask"][:, :ctx_len].bool()
    output["pred_loss"] = _masked_mean(
        (pred_emb - tgt_emb).pow(2).mean(dim=-1), transition_mask
    )

    valid_mask = batch["valid_mask"][:, :ctx_len].bool()
    valid_emb = ctx_emb[valid_mask]
    output["sigreg_loss"] = (
        self.sigreg(valid_emb.unsqueeze(1))
        if valid_emb.numel()
        else ctx_emb.sum() * 0.0
    )

    memory = self.model.predictor.memory_states(ctx_emb)
    cue_logits = self.model.cue_probe(memory)
    act_logits = self.model.action_probe(memory)
    cue_label = batch["cue_color"][:, :ctx_len].long()
    act_label = batch["action"][:, :ctx_len].squeeze(-1).long()
    memory_mask = batch["memory_mask"][:, :ctx_len].bool()
    decision_mask = batch["decision_mask"][:, :ctx_len].bool()
    output["cue_loss"] = _masked_cross_entropy(cue_logits, cue_label, memory_mask)
    output["act_loss"] = _masked_cross_entropy(act_logits, act_label, decision_mask)
    output["cue_logits"] = cue_logits
    output["act_logits"] = act_logits
    output["loss"] = (
        output["pred_loss"]
        + cfg.loss.sigreg.weight * output["sigreg_loss"]
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
            f"{stage}/{key}": output[key]
            for key in (
                "loss",
                "pred_loss",
                "sigreg_loss",
                "cue_loss",
                "act_loss",
                "act_acc",
                "cue_acc",
            )
        },
        on_step=True,
    )
    return output


def custom_collate(batch):
    """Pad variable-length episode dictionaries into a batch."""
    out = {}
    for key in batch[0]:
        if key == "pixels":
            max_len = max(item[key].shape[0] for item in batch)
            batch_size = len(batch)
            channels, height, width = batch[0][key].shape[1:]
            padded = torch.zeros(
                (batch_size, max_len, channels, height, width),
                dtype=batch[0][key].dtype,
            )
            for index, item in enumerate(batch):
                length = item[key].shape[0]
                padded[index, :length] = item[key]
            out[key] = padded
        elif torch.is_tensor(batch[0][key]):
            out[key] = pad_sequence(
                [item[key] for item in batch], batch_first=True, padding_value=0
            )
        else:
            out[key] = [item[key] for item in batch]
    return out


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    torch.set_float32_matmul_precision("high")
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset_format = dataset_cfg.pop("format", None)
    cache_dir = (
        os.environ.get("LOCAL_DATASET_DIR")
        or os.environ.get("STABLEWM_HOME")
        or str(Path(__file__).resolve().parent / "data")
    )
    # Keep dataset and checkpoint resolution on the same portable root.  An
    # explicit STABLEWM_HOME still takes precedence over the repository default.
    os.environ.setdefault("STABLEWM_HOME", cache_dir)
    if dataset_format == "hdf5":
        from hdf5_dataset import HDF5Dataset

        dataset_path = Path(dataset_name)
        if not dataset_path.is_absolute():
            dataset_path = Path(cache_dir) / "datasets" / dataset_path
        dataset = HDF5Dataset(dataset_path, transform=None, **dataset_cfg)
    else:
        dataset = swm.data.load_dataset(
            dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
        )
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if (
                col.startswith("pixels")
                or col.startswith("cue")
                or col.startswith("action")
                or col.endswith("_mask")
            ):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False, collate_fn=custom_collate
    )

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        }
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(wm_maze_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    checkpoint_run_name = cfg.checkpoint_run_name
    run_dir = Path(
        swm.data.utils.get_cache_dir(Path(cache_dir), sub_folder="checkpoints"),
        checkpoint_run_name,
        run_id,
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=checkpoint_run_name, cfg=cfg.model, epoch_interval=1
    )
    callbacks = [object_dump_callback]
    if cfg.stop_after_epoch is not None:
        callbacks.append(StopAfterEpoch(cfg.stop_after_epoch))
    if cfg.validation_video.enabled:
        callbacks.append(
            WMMazeValidationVideo(
                every_n_epochs=cfg.validation_video.every_n_epochs,
                fps=cfg.validation_video.fps,
                sample_index=cfg.validation_video.sample_index,
                padding=cfg.validation_video.padding,
            )
        )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
