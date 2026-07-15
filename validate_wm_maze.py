"""Validate the working-memory maze HDF5 data contract."""

import argparse

import h5py
import numpy as np

from gen_data_perfect import default_output_path
from wm_maze_env import COLOR_TO_TURN


REQUIRED_KEYS = {
    "pixels",
    "action",
    "cue_color",
    "valid_mask",
    "transition_mask",
    "decision_mask",
    "memory_mask",
    "ep_len",
    "ep_offset",
}


def validate(path):
    with h5py.File(path, "r") as handle:
        missing = REQUIRED_KEYS.difference(handle.keys())
        if missing:
            raise ValueError(f"missing datasets: {sorted(missing)}")

        lengths = handle["ep_len"][:]
        offsets = handle["ep_offset"][:]
        if len(lengths) == 0 or not np.all(lengths == lengths[0]):
            raise ValueError("episodes must have one fixed length")
        action_counts = np.zeros(3, dtype=np.int64)
        valid_lengths = []

        for episode, (offset, length) in enumerate(zip(offsets, lengths)):
            section = slice(int(offset), int(offset + length))
            action = handle["action"][section]
            cue = handle["cue_color"][section]
            valid = handle["valid_mask"][section].astype(bool)
            transition = handle["transition_mask"][section].astype(bool)
            decision = handle["decision_mask"][section].astype(bool)
            memory = handle["memory_mask"][section].astype(bool)

            if decision.sum() != 3:
                raise ValueError(f"episode {episode} has {decision.sum()} decisions")
            if not np.all(transition <= valid) or not np.all(memory <= valid):
                raise ValueError(f"episode {episode} has a mask outside valid frames")
            if transition.sum() + 1 != valid.sum():
                raise ValueError(f"episode {episode} has an invalid terminal transition")

            decision_actions = action[decision]
            if sorted(decision_actions.tolist()) != [0, 1, 2]:
                raise ValueError(f"episode {episode} decisions are not balanced")
            expected_actions = np.array(
                [
                    COLOR_TO_TURN[color]
                    for color in ("red", "blue", "green")
                ]
            )
            cue_to_action = expected_actions[cue[decision]]
            if not np.array_equal(cue_to_action, decision_actions):
                raise ValueError(f"episode {episode} cue/action mapping is wrong")

            pixels = handle["pixels"][section]
            if pixels[0].sum() == 0 or not np.any(pixels.sum(axis=(1, 2, 3)) == 0):
                raise ValueError(f"episode {episode} is missing cue or delay frames")
            action_counts += np.bincount(decision_actions, minlength=3)
            valid_lengths.append(int(valid.sum()))

    print(f"Dataset: {path}")
    print(f"Episodes: {len(lengths)}, frames per episode: {int(lengths[0])}")
    print(f"Valid frames: {min(valid_lengths)}–{max(valid_lengths)}")
    print(f"Decision actions [forward, left, right]: {action_counts.tolist()}")
    print("Validation passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=default_output_path())
    args = parser.parse_args()
    validate(args.path)


if __name__ == "__main__":
    main()
