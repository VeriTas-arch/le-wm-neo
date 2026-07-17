"""Evaluate memory and action probes on maze decision frames."""

import argparse
import os
import re
from pathlib import Path

import hydra
import stable_pretraining as spt
import torch

from generate_wm_maze import default_output_path
from hdf5_dataset import HDF5Dataset
from utils import get_img_preprocessor
from wm_maze_video import VIDEO_PRESETS, WMMazeValidationVideo


def load_model(checkpoint, device):
    with hydra.initialize(version_base=None, config_path="config/train"):
        cfg = hydra.compose(config_name="lewm", overrides=["data=wm_maze"])
    model = hydra.utils.instantiate(cfg.model).to(device).eval()
    saved = torch.load(checkpoint, map_location=device)
    raw_state = saved.get("state_dict", saved)
    state = {}
    for key, value in raw_state.items():
        if key.startswith("module.model."):
            state[key.removeprefix("module.model.")] = value
        elif key.startswith("model."):
            state[key.removeprefix("model.")] = value
        elif not key.startswith(("module.", "sigreg.", "callbacks_")):
            state[key] = value
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model, cfg


@torch.inference_mode()
def evaluate(model, cfg, dataset, device, max_episodes):
    action_correct = cue_correct = decisions = 0
    memory_correct = memory_total = 0
    preprocess = get_img_preprocessor("pixels", "pixels", cfg.img_size)
    count = min(len(dataset), max_episodes or len(dataset))
    for index in range(count):
        batch = preprocess(dataset[index])
        batch = {key: value.unsqueeze(0).to(device) for key, value in batch.items()}
        encoded = model.encode(batch)
        context = encoded["emb"][:, : cfg.history_size]
        memory = model.predictor.memory_states(context)
        action_logits = model.action_probe(memory)
        cue_logits = model.cue_probe(memory)

        action_labels = batch["action"][:, : cfg.history_size].squeeze(-1)
        cue_labels = batch["cue_color"][:, : cfg.history_size]
        decision_mask = batch["decision_mask"][:, : cfg.history_size].bool()
        memory_mask = batch["memory_mask"][:, : cfg.history_size].bool()

        action_correct += int(
            (
                action_logits.argmax(-1)[decision_mask] == action_labels[decision_mask]
            ).sum()
        )
        cue_correct += int(
            (cue_logits.argmax(-1)[decision_mask] == cue_labels[decision_mask]).sum()
        )
        decisions += int(decision_mask.sum())
        memory_correct += int(
            (cue_logits.argmax(-1)[memory_mask] == cue_labels[memory_mask]).sum()
        )
        memory_total += int(memory_mask.sum())

    print(f"Episodes: {count}")
    print(f"Decision action accuracy: {action_correct / decisions:.2%}")
    print(f"Decision cue accuracy: {cue_correct / decisions:.2%}")
    print(f"All-memory cue accuracy: {memory_correct / memory_total:.2%}")


@torch.inference_mode()
def export_validation_video(
    model,
    cfg,
    dataset,
    checkpoint,
    device,
    sample_index,
    fps,
    padding,
    video_preset,
    output,
):
    """Export one deterministic sample from the same validation split as train.py."""
    generator = torch.Generator().manual_seed(cfg.seed)
    _, validation = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=generator
    )
    if not 0 <= sample_index < len(validation):
        raise IndexError(
            f"validation sample {sample_index} is outside [0, {len(validation)})"
        )

    preprocess = get_img_preprocessor("pixels", "pixels", cfg.img_size)
    sample = preprocess(validation[sample_index])
    batch = {key: value.unsqueeze(0).to(device) for key, value in sample.items()}
    encoded = model.encode(batch)
    context = encoded["emb"][:, : cfg.history_size]
    memory = model.predictor.memory_states(context)
    outputs = {
        "cue_logits": model.cue_probe(memory),
        "act_logits": model.action_probe(memory),
    }

    match = re.search(r"epoch_(\d+)", checkpoint.stem)
    epoch = int(match.group(1)) if match else 0
    if output is None:
        root = Path(os.environ.get("STABLEWM_HOME", "data")) / "validation"
        suffix = "_report" if video_preset == "report" else ""
        output = root / (
            f"formal_epoch_{epoch:03d}_validation_{sample_index:04d}{suffix}.mp4"
        )
    exporter = WMMazeValidationVideo(
        fps=fps,
        sample_index=0,
        padding=padding,
        video_preset=video_preset,
    )
    exporter._write(epoch, batch, outputs, video_path=output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=default_output_path())
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--video", action="store_true", help="export one annotated validation video"
    )
    parser.add_argument(
        "--video-only", action="store_true", help="skip aggregate metrics"
    )
    parser.add_argument("--video-index", type=int, default=0)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument(
        "--video-preset",
        choices=VIDEO_PRESETS,
        default="standard",
        help="encoding preset (default: standard)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=128,
        help="outer canvas padding in pixels (default: 128)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint, args.device)
    dataset = HDF5Dataset(
        args.data,
        num_steps=cfg.history_size + cfg.num_preds,
        frameskip=1,
        keys_to_load=list(cfg.data.dataset.keys_to_load),
    )
    if not args.video_only:
        evaluate(model, cfg, dataset, args.device, args.episodes)
    if args.video or args.video_only:
        export_validation_video(
            model,
            cfg,
            dataset,
            args.checkpoint,
            args.device,
            args.video_index,
            args.fps,
            args.padding,
            args.video_preset,
            args.video_output,
        )


if __name__ == "__main__":
    main()
