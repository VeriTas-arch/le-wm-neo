import numpy as np
from PIL import Image, ImageDraw

COLORS_RGB = {"red": (220, 50, 50), "blue": (50, 100, 240), "green": (50, 200, 50)}
COLOR_TO_TURN = {"red": 1, "blue": 2, "green": 0}
TURN_TO_COLOR = {1: "red", 2: "blue", 0: "green"}
H_DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


def turn_heading(h, d):
    return h if d == 0 else ((h - 1) % 4 if d == 1 else (h + 1) % 4)


SIZE = 15
CELL = 14
BORDER = 7


class WMMazeEnv:
    IMG = 224
    CELL = 14
    BORDER = 7

    def __init__(
        self,
        cue_sequence=None,
        cue_duration=30,
        delay_duration=60,
        seed=None,
        view_radius=2,
    ):
        self.cue_sequence = cue_sequence
        self.cue_duration = cue_duration
        self.delay_duration = delay_duration
        self.rng = np.random.default_rng(seed)
        self.view_radius = view_radius
        self._rst()

    def _rst(self):
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

    def reset(self, seed=None, maze=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._rst()
        self.phase = "cue"
        if self.cue_sequence:
            self.border = COLORS_RGB[self.cue_sequence[0]]
        if maze is not None:
            self.load_maze(maze)
        if self.grid is not None:
            return self._render(), {}
        return (
            __import__("numpy").zeros((224, 224, 3), dtype=__import__("numpy").uint8),
            {},
        )

    def load_maze(self, maze_dict):
        self.grid = np.array(maze_dict["grid"], dtype=np.int32)
        self.path = list(maze_dict["path"])
        self.intersections = list(maze_dict["intersections"])
        self.heading = maze_dict["start_heading"]

    @property
    def pos(self):
        return self.path[self.path_idx]

    def step(self, action):
        reward = 0.0
        info = {}
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
            for ii, (pi, _) in enumerate(self.intersections):
                if self.path_idx == pi and self.inter_done == ii:
                    expected = COLOR_TO_TURN[self.cue_sequence[ii]]
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
                    h for h, delta in H_DELTA.items() if delta == movement
                )
            r, c = self.pos
            if self.grid[r, c] == 2:
                reward = 1.0
                self.success = True
                self.done = True
        info["success"] = self.success
        info["inter_done"] = self.inter_done
        return self._render(), reward, self.done, False, info

    def _render(self):
        if self.phase == "delay":
            return np.zeros((224, 224, 3), dtype=np.uint8)
        total = CELL * 15 + 2 * BORDER
        img = Image.new("RGB", (total, total), (240, 240, 240))
        d = ImageDraw.Draw(img)
        for r in range(15):
            for cc in range(15):
                x0, y0 = BORDER + cc * CELL, BORDER + r * CELL
                visible = (
                    self.grid is not None
                    and abs(r - self.pos[0]) <= self.view_radius
                    and abs(cc - self.pos[1]) <= self.view_radius
                )
                v = self.grid[r, cc] if visible else 1
                # The goal is deliberately rendered as ordinary floor.  At a
                # decision point all three local branches look alike, so the
                # cue rather than the global route determines the action.
                fill = {0: (250, 250, 250), 1: (50, 50, 60), 2: (250, 250, 250)}.get(v)
                d.rectangle(
                    [x0, y0, x0 + CELL, y0 + CELL], fill=fill, outline=(180, 180, 185)
                )
        r, cc = self.pos
        cx, cy = BORDER + cc * CELL + CELL // 2, BORDER + r * CELL + CELL // 2
        rpx = max(CELL // 3, 2)
        d.ellipse(
            [cx - rpx, cy - rpx, cx + rpx, cy + rpx],
            fill=(220, 50, 50),
            outline=(160, 30, 30),
        )
        dr, dc = H_DELTA[self.heading]
        d.line(
            [cx, cy, cx + dc * rpx * 3, cy + dr * rpx * 3],
            fill=(255, 255, 255),
            width=2,
        )
        bc = self.border or (0, 0, 0)
        d.rectangle([0, 0, total - 1, total - 1], outline=bc, width=BORDER)
        img = img.resize((224, 224), Image.NEAREST)
        return np.array(img, dtype=np.uint8)
