"""Generate the color-cued working-memory maze dataset.

The default output is portable: ``data/datasets/wm_maze.h5`` inside this
repository.  Set LOCAL_DATASET_DIR or STABLEWM_HOME to place it elsewhere.
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

COLOR_TO_IDX = {"red": 0, "blue": 1, "green": 2}
COLORS_RGB = {"red": (220, 50, 50), "blue": (50, 100, 240), "green": (50, 200, 50)}
COLOR_TO_TURN = {"red": 1, "blue": 2, "green": 0}
TURN_TO_COLOR = {1: "red", 2: "blue", 0: "green"}
H_DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
CELL = 14
BORDER = 7


def turn_heading(heading, action):
    if action == 0:
        return heading
    return (heading - 1) % 4 if action == 1 else (heading + 1) % 4


class WMMazeEnv:
    """Partially observable maze used by the dataset generator."""

    def __init__(
        self, cue_sequence=None, cue_duration=30, delay_duration=60, view_radius=2
    ):
        self.cue_sequence = cue_sequence
        self.cue_duration = cue_duration
        self.delay_duration = delay_duration
        self.view_radius = view_radius
        self._reset_state()

    def _reset_state(self):
        self.path_idx = 0
        self.heading = 0
        self.inter_done = 0
        self.phase = "cue"
        self.phase_frame = 0
        self.cue_i = 0
        self.border = None
        self.done = False
        self.success = False
        self.grid = None
        self.path = []
        self.intersections = []

    def reset(self, maze=None):
        self._reset_state()
        if self.cue_sequence:
            self.border = COLORS_RGB[self.cue_sequence[0]]
        if maze is not None:
            self.grid = np.asarray(maze["grid"], dtype=np.int32)
            self.path = list(maze["path"])
            self.intersections = list(maze["intersections"])
            self.heading = maze["start_heading"]
        return self._render(), {}

    @property
    def pos(self):
        return self.path[self.path_idx]

    def step(self, action):
        reward = 0.0
        if self.phase == "cue":
            self.phase_frame += 1
            if self.phase_frame >= self.cue_duration:
                self.cue_i += 1
                self.phase_frame = 0
                if self.cue_i >= len(self.cue_sequence):
                    self.phase = "delay"
                    self.border = None
                else:
                    self.border = COLORS_RGB[self.cue_sequence[self.cue_i]]
        elif self.phase == "delay":
            self.phase_frame += 1
            if self.phase_frame >= self.delay_duration:
                self.phase = "action"
        elif self.phase == "action":
            for index, (path_index, _) in enumerate(self.intersections):
                if self.path_idx == path_index and self.inter_done == index:
                    expected = COLOR_TO_TURN[self.cue_sequence[index]]
                    if action == expected:
                        reward = 0.2
                        self.inter_done += 1
                        self.heading = turn_heading(self.heading, action)
                    else:
                        reward = -0.5
                        self.done = True
                    break
            if not self.done and self.path_idx < len(self.path) - 1:
                previous = self.pos
                self.path_idx += 1
                current = self.pos
                movement = (current[0] - previous[0], current[1] - previous[1])
                self.heading = next(
                    heading for heading, delta in H_DELTA.items() if delta == movement
                )
            row, col = self.pos
            if self.grid[row, col] == 2:
                reward = 1.0
                self.success = True
                self.done = True
        return (
            self._render(),
            reward,
            self.done,
            False,
            {"success": self.success, "inter_done": self.inter_done},
        )

    def _render(self):
        if self.phase == "delay":
            return np.zeros((224, 224, 3), dtype=np.uint8)
        size = CELL * 15 + 2 * BORDER
        image = Image.new("RGB", (size, size), (240, 240, 240))
        draw = ImageDraw.Draw(image)
        for row in range(15):
            for col in range(15):
                x0, y0 = BORDER + col * CELL, BORDER + row * CELL
                visible = (
                    abs(row - self.pos[0]) <= self.view_radius
                    and abs(col - self.pos[1]) <= self.view_radius
                )
                value = self.grid[row, col] if visible else 1
                fill = {0: (250, 250, 250), 1: (50, 50, 60), 2: (250, 250, 250)}[value]
                draw.rectangle(
                    [x0, y0, x0 + CELL, y0 + CELL], fill=fill, outline=(180, 180, 185)
                )
        row, col = self.pos
        cx = BORDER + col * CELL + CELL // 2
        cy = BORDER + row * CELL + CELL // 2
        radius = max(CELL // 3, 2)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(220, 50, 50),
            outline=(160, 30, 30),
        )
        dr, dc = H_DELTA[self.heading]
        draw.line(
            [cx, cy, cx + dc * radius * 3, cy + dr * radius * 3],
            fill=(255, 255, 255),
            width=2,
        )
        draw.rectangle(
            [0, 0, size - 1, size - 1], outline=self.border or (0, 0, 0), width=BORDER
        )
        return np.asarray(image.resize((224, 224), Image.NEAREST), dtype=np.uint8)


def generate_maze(rng, required_actions=None):
    """Generate a route with three locally ambiguous decision points."""
    if required_actions is None:
        required_actions = rng.permutation(3)
    required_actions = list(required_actions)
    for _ in range(3000):
        grid = np.ones((15, 15), dtype=np.int32)
        row = int(rng.integers(2, 13))
        col = 0
        path = [(row, col)]
        grid[row, col] = 0
        direction = (0, 1)

        while col < 14:
            if direction == (0, 1):
                for _ in range(int(rng.integers(2, 6))):
                    if col < 14:
                        col += 1
                        path.append((row, col))
                        grid[row, col] = 0
            else:
                for _ in range(int(rng.integers(2, 5))):
                    if 1 <= row + direction[0] <= 13:
                        row += direction[0]
                        path.append((row, col))
                        grid[row, col] = 0
                    else:
                        break
            if col >= 14:
                break
            if direction == (0, 1):
                candidates = []
                if row > 3:
                    candidates.append((-1, 0))
                if row < 11:
                    candidates.append((1, 0))
                direction = candidates[int(rng.integers(len(candidates)))]
            else:
                direction = (0, 1)

        grid[path[-1]] = 2
        intersections = []
        last_intersection = -5
        for i in range(1, len(path) - 1):
            if i - last_intersection < 3:
                continue
            incoming = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            outgoing = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            if incoming == outgoing and rng.random() > 0.15:
                continue
            heading_in = next(
                (h for h, delta in H_DELTA.items() if delta == incoming), None
            )
            heading_out = next(
                (h for h, delta in H_DELTA.items() if delta == outgoing), None
            )
            if heading_in is None or heading_out is None:
                continue
            if heading_out == heading_in:
                correct_action = 0
            elif heading_out == (heading_in - 1) % 4:
                correct_action = 1
            elif heading_out == (heading_in + 1) % 4:
                correct_action = 2
            else:
                continue

            branch_count = 0
            decision_row, decision_col = path[i]
            for wrong_action in (0, 1, 2):
                if wrong_action == correct_action:
                    continue
                branch_heading = turn_heading(heading_in, wrong_action)
                dr, dc = H_DELTA[branch_heading]
                branch_dug = False
                for step in range(1, int(rng.integers(3, 6))):
                    rr = decision_row + dr * step
                    cc = decision_col + dc * step
                    if not (1 <= rr < 14 and 1 <= cc < 14):
                        break
                    if grid[rr, cc] != 1:
                        break
                    neighbours = sum(
                        grid[rr + ar, cc + ac] == 0
                        for ar, ac in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    )
                    if neighbours > 1:
                        break
                    grid[rr, cc] = 0
                    branch_dug = True
                branch_count += int(branch_dug)

            # Both wrong branches must be visible in the local observation.
            if branch_count == 2:
                intersections.append((i, correct_action))
                last_intersection = i

        if len(intersections) < 3:
            continue
        selected = next(
            (
                [intersections[index] for index in indices]
                for indices in itertools.combinations(range(len(intersections)), 3)
                if [intersections[index][1] for index in indices] == required_actions
            ),
            None,
        )
        if selected is None:
            continue
        intersections = selected
        first_delta = (path[1][0] - path[0][0], path[1][1] - path[0][1])
        start_heading = next(h for h, delta in H_DELTA.items() if delta == first_delta)
        cues = [TURN_TO_COLOR[action] for _, action in intersections]
        return {
            "grid": grid.tolist(),
            "path": path,
            "intersections": intersections,
            "start_heading": start_heading,
            "cue_sequence": cues,
        }
    raise RuntimeError("failed to generate a valid maze")


def rollout_episode(rng, target_len, cue_duration, delay_min, delay_max):
    maze = generate_maze(rng, rng.permutation(3))
    delay = int(rng.integers(delay_min, delay_max + 1))
    env = WMMazeEnv(
        cue_sequence=maze["cue_sequence"],
        cue_duration=cue_duration,
        delay_duration=delay,
        view_radius=2,
    )
    obs, _ = env.reset(maze=maze)

    columns = {
        "pixels": [],
        "action": [],
        "cue_color": [],
        "valid_mask": [],
        "transition_mask": [],
        "decision_mask": [],
        "memory_mask": [],
    }
    done = False
    while len(columns["pixels"]) < target_len - 1 and not done:
        decision = (
            any(
                env.path_idx == path_idx and env.inter_done == index
                for index, (path_idx, _) in enumerate(env.intersections)
            )
            if env.phase == "action"
            else False
        )
        action = 0
        if decision:
            action = env.intersections[env.inter_done][1]
        cue_index = env.cue_i if env.phase == "cue" else env.inter_done
        cue_index = min(cue_index, len(env.cue_sequence) - 1)

        columns["pixels"].append(obs.copy())
        columns["action"].append(action)
        columns["cue_color"].append(COLOR_TO_IDX[env.cue_sequence[cue_index]])
        columns["valid_mask"].append(1)
        columns["transition_mask"].append(1)
        columns["decision_mask"].append(int(decision))
        columns["memory_mask"].append(int(env.phase in ("delay", "action")))
        obs, _, done, _, _ = env.step(action)

    if not done:
        return None

    # Keep the genuine terminal observation, but do not train a transition
    # out of it.  Everything after it is explicit padding.
    columns["pixels"].append(obs.copy())
    columns["action"].append(0)
    columns["cue_color"].append(columns["cue_color"][-1])
    columns["valid_mask"].append(1)
    columns["transition_mask"].append(0)
    columns["decision_mask"].append(0)
    columns["memory_mask"].append(1)

    while len(columns["pixels"]) < target_len:
        columns["pixels"].append(obs.copy())
        columns["action"].append(0)
        columns["cue_color"].append(columns["cue_color"][-1])
        columns["valid_mask"].append(0)
        columns["transition_mask"].append(0)
        columns["decision_mask"].append(0)
        columns["memory_mask"].append(0)
    return {key: np.asarray(value) for key, value in columns.items()}


def default_output_path():
    root = os.environ.get("LOCAL_DATASET_DIR") or os.environ.get("STABLEWM_HOME")
    root = Path(root) if root else Path(__file__).resolve().parent / "data"
    return root / "datasets" / "wm_maze.h5"


def generate_dataset(
    output, episodes, target_len, cue_duration, delay_min, delay_max, seed
):
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    total_steps = episodes * target_len
    with h5py.File(output, "w") as handle:
        datasets = {
            "pixels": handle.create_dataset(
                "pixels",
                (total_steps, 224, 224, 3),
                dtype=np.uint8,
                compression="lzf",
                chunks=(1, 224, 224, 3),
            ),
            "action": handle.create_dataset("action", (total_steps,), dtype=np.int64),
            "cue_color": handle.create_dataset(
                "cue_color", (total_steps,), dtype=np.int64
            ),
            "valid_mask": handle.create_dataset(
                "valid_mask", (total_steps,), dtype=np.uint8
            ),
            "transition_mask": handle.create_dataset(
                "transition_mask", (total_steps,), dtype=np.uint8
            ),
            "decision_mask": handle.create_dataset(
                "decision_mask", (total_steps,), dtype=np.uint8
            ),
            "memory_mask": handle.create_dataset(
                "memory_mask", (total_steps,), dtype=np.uint8
            ),
        }
        handle.create_dataset("ep_len", data=np.full(episodes, target_len, np.int32))
        handle.create_dataset(
            "ep_offset", data=np.arange(episodes, dtype=np.int64) * target_len
        )
        handle.attrs["cue_duration"] = cue_duration
        handle.attrs["delay_min"] = delay_min
        handle.attrs["delay_max"] = delay_max
        handle.attrs["seed"] = seed

        for episode in tqdm(range(episodes), desc="Generating mazes"):
            data = None
            while data is None:
                data = rollout_episode(
                    rng, target_len, cue_duration, delay_min, delay_max
                )
            start = episode * target_len
            end = start + target_len
            for key, values in data.items():
                datasets[key][start:end] = values
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--target-len", type=int, default=64)
    parser.add_argument("--cue-duration", type=int, default=4)
    parser.add_argument("--delay-min", type=int, default=6)
    parser.add_argument("--delay-max", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    output = generate_dataset(
        args.output,
        args.episodes,
        args.target_len,
        args.cue_duration,
        args.delay_min,
        args.delay_max,
        args.seed,
    )
    print(f"Dataset written to {output}")


if __name__ == "__main__":
    main()
