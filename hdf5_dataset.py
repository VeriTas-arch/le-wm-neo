"""Minimal HDF5 reader for LeWM datasets.

This avoids stable-worldmodel's optional hdf5plugin dependency.  The generated
maze files use HDF5's built-in LZF compression and only require h5py.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from stable_worldmodel.data.dataset import Dataset


class HDF5Dataset(Dataset):
    def __init__(
        self,
        path,
        frameskip=1,
        num_steps=1,
        transform=None,
        keys_to_load=None,
        keys_to_cache=None,
    ):
        self.path = Path(path)
        self.h5_file = None
        self._cache = {}
        with h5py.File(self.path, "r") as handle:
            lengths = handle["ep_len"][:]
            offsets = handle["ep_offset"][:]
            self._keys = keys_to_load or [
                key for key in handle.keys() if key not in ("ep_len", "ep_offset")
            ]
            for key in keys_to_cache or []:
                self._cache[key] = handle[key][:]
        super().__init__(lengths, offsets, frameskip, num_steps, transform)

    @property
    def column_names(self):
        return self._keys

    def _open(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.path, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5_file"] = None
        return state

    def _load_slice(self, ep_idx, start, end):
        self._open()
        global_start = int(self.offsets[ep_idx] + start)
        global_end = int(self.offsets[ep_idx] + end)
        output = {}
        for key in self._keys:
            source = self._cache if key in self._cache else self.h5_file
            data = source[key][global_start:global_end]
            if key != "action":
                data = data[:: self.frameskip]
            value = torch.from_numpy(np.asarray(data))
            if data.ndim == 4 and data.shape[-1] in (1, 3):
                value = value.permute(0, 3, 1, 2)
            output[key] = value
        return self.transform(output) if self.transform else output

    def get_col_data(self, col):
        if col in self._cache:
            return self._cache[col]
        self._open()
        return self.h5_file[col][:]
