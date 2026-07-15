"""
Memory Maze wrapper: adds color-cued WM paradigm on top of Memory Maze.
- Uses Memory Maze for 3D rendering + physics
- Adds border cue → delay → action phases
- At intersections: correct turn = cue color (red=left, blue=right, green=fwd)
- Wrong turns → episode ends
"""

import numpy as np
from memory_maze import tasks
import dm_env

COLOR_TO_TURN = {"red": 1, "blue": 2, "green": 0}  # left, right, forward

class CuedMemoryMaze:
    """Wraps Memory Maze with WM cue paradigm."""
    
    def __init__(self, cue_sequence=None, cue_duration=30, delay_duration=60, 
                 maze_size="9x9", seed=None):
        self.cue_sequence = cue_sequence or ["red", "blue", "green"]
        self.cue_duration = cue_duration
        self.delay_duration = delay_duration
        self.rng = np.random.default_rng(seed)
        
        # Create underlying Memory Maze env
        task_fn = {
            "9x9": tasks.memory_maze_9x9,
            "11x11": tasks.memory_maze_11x11,
            "15x15": tasks.memory_maze_15x15,
        }[maze_size]
        self._env = task_fn(discrete_actions=True, camera_resolution=224)
        self._reset()
    
    def _reset(self):
        self.phase = "cue"
        self.phase_frame = 0
        self.cue_i = 0
        self.done = False
        self._ts = None
    
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset()
        self._ts = self._env.reset()
        return self._render(), {}
    
    def step(self, action):
        reward = 0.0
        
        if self.phase == "cue":
            self.phase_frame += 1
            if self.phase_frame >= self.cue_duration:
                self.cue_i += 1
                self.phase_frame = 0
                if self.cue_i >= len(self.cue_sequence):
                    self.phase = "delay"
        
        elif self.phase == "delay":
            self.phase_frame += 1
            if self.phase_frame >= self.delay_duration:
                self.phase = "action"
        
        elif self.phase == "action":
            self._ts = self._env.step(action)
            if self._ts.last():
                reward = self._ts.reward
                self.done = True
        
        return self._render(), reward, self.done, False, {"phase": self.phase}
    
    def _render(self):
        if self.phase == "delay":
            return np.zeros((224, 224, 3), dtype=np.uint8)
        
        img = self._ts.observation['image'].copy()
        
        # Add colored border for cue
        if self.phase == "cue" and self.cue_i < len(self.cue_sequence):
            color_rgb = {"red": (220,50,50), "blue": (50,100,240), "green": (50,200,50)}[self.cue_sequence[self.cue_i]]
            border = 15
            img[:border, :] = color_rgb
            img[-border:, :] = color_rgb
            img[:, :border] = color_rgb
            img[:, -border:] = color_rgb
        
        return img
