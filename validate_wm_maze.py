"""Validate the working-memory maze HDF5 data contract."""

import argparse
import csv
import os
from pathlib import Path

import cv2
import h5py
import numpy as np

from generate_wm_maze import default_output_path
from video_export import H264VideoWriter, add_canvas_padding
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

CUE_NAMES = ("red", "blue", "green")
ACTION_NAMES = ("forward", "left", "right")
CUE_BGR = {
    "red": (50, 50, 220),
    "blue": (240, 100, 50),
    "green": (50, 200, 50),
}


def _phase_name(frame, valid, transition, decision, memory):
    if not valid:
        return "padding"
    if not transition:
        return "terminal"
    if not memory:
        return "cue"
    if frame.sum() == 0:
        return "delay"
    if decision:
        return "decision"
    return "action"


def _episode_data(handle, episode):
    if not 0 <= episode < len(handle["ep_len"]):
        raise IndexError(
            f"episode {episode} is outside [0, {len(handle['ep_len']) - 1}]"
        )
    offset = int(handle["ep_offset"][episode])
    length = int(handle["ep_len"][episode])
    section = slice(offset, offset + length)
    return {
        key: handle[key][section]
        for key in REQUIRED_KEYS
        if key not in ("ep_len", "ep_offset")
    }


def export_episode(path, episode, video_path, fps, padding=24):
    with h5py.File(path, "r") as handle:
        data = _episode_data(handle, episode)

    valid = data["valid_mask"].astype(bool)
    transition = data["transition_mask"].astype(bool)
    decision = data["decision_mask"].astype(bool)
    memory = data["memory_mask"].astype(bool)
    cue = data["cue_color"].astype(np.int64)
    action = data["action"].astype(np.int64)
    phases = [
        _phase_name(frame, v, tr, dec, mem)
        for frame, v, tr, dec, mem in zip(
            data["pixels"], valid, transition, decision, memory
        )
    ]

    video_path = Path(video_path).expanduser().resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = video_path.with_suffix(".csv")
    frame_height, frame_width = data["pixels"].shape[1:3]
    panel_width = 360
    canvas_width = frame_width + panel_width + 2 * padding
    canvas_height = frame_height + 2 * padding
    rows = []
    with H264VideoWriter(
        video_path, fps, (canvas_width, canvas_height)
    ) as writer:
        for step in np.flatnonzero(valid):
            cue_name = CUE_NAMES[cue[step]]
            action_name = ACTION_NAMES[action[step]]
            phase = phases[step]
            canvas = np.full(
                (frame_height, frame_width + panel_width, 3), 28, dtype=np.uint8
            )
            canvas[:, :frame_width] = cv2.cvtColor(
                data["pixels"][step], cv2.COLOR_RGB2BGR
            )
            x = frame_width + 20
            lines = (
                f"Episode: {episode}",
                f"Step: {step:02d}",
                f"Phase: {phase}",
                f"Cue GT: {cue_name}",
                f"Action GT: {action_name}",
                f"Decision: {'yes' if decision[step] else 'no'}",
            )
            for line_index, line in enumerate(lines):
                cv2.putText(
                    canvas,
                    line,
                    (x, 32 + line_index * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
            cv2.rectangle(
                canvas,
                (frame_width + panel_width - 65, 98),
                (frame_width + panel_width - 25, 138),
                CUE_BGR[cue_name],
                thickness=-1,
            )
            if decision[step]:
                cv2.rectangle(
                    canvas,
                    (2, 2),
                    (frame_width - 3, frame_height - 3),
                    (0, 215, 255),
                    thickness=4,
                )
            writer.write(add_canvas_padding(canvas, padding))
            rows.append(
                {
                    "episode": episode,
                    "step": int(step),
                    "phase": phase,
                    "cue_id": int(cue[step]),
                    "cue": cue_name,
                    "action_id": int(action[step]),
                    "action": action_name,
                    "decision": int(decision[step]),
                    "memory": int(memory[step]),
                    "transition": int(transition[step]),
                }
            )
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    cue_sequence = []
    for step, phase in enumerate(phases):
        if phase != "cue":
            continue
        cue_name = CUE_NAMES[cue[step]]
        if not cue_sequence or cue_sequence[-1] != cue_name:
            cue_sequence.append(cue_name)
    decision_cues = [CUE_NAMES[value] for value in cue[decision]]
    print(f"Episode {episode} displayed cue sequence: {cue_sequence}")
    print(f"Episode {episode} decision cues: {decision_cues}")
    print(f"Video: {video_path}")
    print(f"Cue timeline: {csv_path}")
    return video_path, csv_path


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
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument(
        "--padding",
        type=int,
        default=24,
        help="outer canvas padding in pixels (default: 24)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="MP4 output path (default: $STABLEWM_HOME/validation/episode_NNNN.mp4)",
    )
    parser.add_argument(
        "--no-video", action="store_true", help="only validate the HDF5 contract"
    )
    args = parser.parse_args()
    validate(args.path)
    if not args.no_video:
        root = Path(
            os.environ.get("STABLEWM_HOME", Path(args.path).resolve().parent.parent)
        )
        video = args.video or root / "validation" / f"episode_{args.episode:04d}.mp4"
        export_episode(args.path, args.episode, video, args.fps, args.padding)


if __name__ == "__main__":
    main()
