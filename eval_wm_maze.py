"""Evaluate memory and action probes on maze decision frames."""

import argparse
from pathlib import Path

import hydra
import torch
from omegaconf import open_dict

from gen_data_perfect import default_output_path
from hdf5_dataset import HDF5Dataset
from utils import get_img_preprocessor


def load_model(checkpoint, device):
    with hydra.initialize(version_base=None, config_path="config/train"):
        cfg = hydra.compose(config_name="lewm", overrides=["data=wm_maze"])
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = 1
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
        batch = {
            key: value.unsqueeze(0).to(device)
            for key, value in batch.items()
        }
        encoded = model.encode(batch)
        context = encoded["emb"][:, :cfg.history_size]
        memory = model.predictor.memory_states(context)
        action_logits = model.action_probe(memory)
        cue_logits = model.cue_probe(memory)

        action_labels = batch["action"][:, :cfg.history_size].squeeze(-1)
        cue_labels = batch["cue_color"][:, :cfg.history_size]
        decision_mask = batch["decision_mask"][:, :cfg.history_size].bool()
        memory_mask = batch["memory_mask"][:, :cfg.history_size].bool()

        action_correct += int(
            (action_logits.argmax(-1)[decision_mask] == action_labels[decision_mask]).sum()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=default_output_path())
    parser.add_argument("--episodes", type=int, default=100)
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
    evaluate(model, cfg, dataset, args.device, args.episodes)


if __name__ == "__main__":
    main()
