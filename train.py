import torch
from torch.nn.utils.rnn import pad_sequence

import os
from functools import partial
from pathlib import Path

import hydra

import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


from train_fix import lejepa_forward_fixed as lejepa_forward


def custom_collate(batch):
    # 假设 batch 是一个列表，里面每个元素是一条完整的数据字典
    out = {}
    for k in batch[0].keys():
        if k == "pixels":
            # 图片由于是 4D 的，单独 pad 可能会慢，所以我们以最大长度为基准创建一个空 Tensor
            max_len = max([b[k].shape[0] for b in batch])
            B = len(batch)
            C, H, W = batch[0][k].shape[1:]
            padded = torch.zeros((B, max_len, C, H, W), dtype=batch[0][k].dtype)
            for i, b in enumerate(batch):
                l = b[k].shape[0]  # noqa: E741
                padded[i, :l] = b[k]
            out[k] = padded
        elif torch.is_tensor(batch[0][k]):
            # 其他一维标签 (action, cue_color) 直接用 pad_sequence
            out[k] = pad_sequence(
                [b[k] for b in batch], batch_first=True, padding_value=0
            )
        else:
            out[k] = [b[k] for b in batch]
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

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim("action")
        )

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
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(
        swm.data.utils.get_cache_dir(Path(cache_dir), sub_folder="checkpoints"), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
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
