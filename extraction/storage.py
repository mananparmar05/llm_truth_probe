"""
storage.py
----------
Efficient save / load of hidden-state tensors using HDF5 (h5py).

Stored layout inside the HDF5 file
------------------------------------
/{dataset_name}/hidden_states   shape [N, num_layers, hidden_dim]  float32
/{dataset_name}/labels          shape [N]                          int32
/{dataset_name}/metadata        attrs: n_samples, num_layers, hidden_dim, created_at
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False
    logger.warning("h5py not installed — falling back to numpy .npz storage.")


class HiddenStateStore:
    """
    Manages persistence of hidden-state tensors.

    Supports HDF5 (preferred) and numpy .npz (fallback).

    Parameters
    ----------
    directory : str | Path
        Directory where the store file(s) will be created.
    filename  : str
        Base filename without extension.
    format    : "hdf5" | "numpy"
        Storage format.  HDF5 allows in-place append; numpy writes full files.
    """

    def __init__(
        self,
        directory: str | Path,
        filename: str = "hidden_states",
        format: str = "hdf5",
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.format = format if _H5PY_AVAILABLE else "numpy"

        if self.format == "hdf5":
            self.path = self.directory / f"{filename}.h5"
        else:
            self.path = self.directory  # directory; each dataset gets its own .npz

        logger.info("HiddenStateStore — format=%s, path=%s", self.format, self.path)

    # ── Save ─────────────────────────────────────────────────────────

    def save(
        self,
        hidden_states: np.ndarray,
        labels: np.ndarray,
        dataset_name: str,
    ) -> None:
        """
        Persist hidden states and labels for a named dataset.

        Parameters
        ----------
        hidden_states : np.ndarray  [N, num_layers, hidden_dim]
        labels        : np.ndarray  [N]
        dataset_name  : str — key used to retrieve later (e.g. "truthfulqa")
        """
        if self.format == "hdf5":
            self._save_hdf5(hidden_states, labels, dataset_name)
        else:
            self._save_numpy(hidden_states, labels, dataset_name)

    def _save_hdf5(self, hidden_states, labels, dataset_name):
        with h5py.File(self.path, "a") as f:
            grp = f.require_group(dataset_name)
            # Overwrite if exists
            for key in ("hidden_states", "labels"):
                if key in grp:
                    del grp[key]
            grp.create_dataset("hidden_states", data=hidden_states, compression="gzip", compression_opts=4)
            grp.create_dataset("labels", data=labels)
            grp.attrs["n_samples"] = hidden_states.shape[0]
            grp.attrs["num_layers"] = hidden_states.shape[1]
            grp.attrs["hidden_dim"] = hidden_states.shape[2]
            grp.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("Saved [%s] to HDF5 — shape %s", dataset_name, hidden_states.shape)

    def _save_numpy(self, hidden_states, labels, dataset_name):
        path = self.directory / f"{self.filename}_{dataset_name}.npz"
        np.savez_compressed(path, hidden_states=hidden_states, labels=labels)
        logger.info("Saved [%s] to numpy — shape %s", dataset_name, hidden_states.shape)

    # ── Load ─────────────────────────────────────────────────────────

    def load(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load hidden states and labels for a named dataset.

        Returns
        -------
        hidden_states : np.ndarray  [N, num_layers, hidden_dim]
        labels        : np.ndarray  [N]
        """
        if self.format == "hdf5":
            return self._load_hdf5(dataset_name)
        else:
            return self._load_numpy(dataset_name)

    def _load_hdf5(self, dataset_name):
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 store not found: {self.path}")
        with h5py.File(self.path, "r") as f:
            if dataset_name not in f:
                raise KeyError(f"Dataset '{dataset_name}' not found in {self.path}")
            grp = f[dataset_name]
            hidden_states = grp["hidden_states"][:]
            labels = grp["labels"][:]
        logger.info("Loaded [%s] from HDF5 — shape %s", dataset_name, hidden_states.shape)
        return hidden_states, labels

    def _load_numpy(self, dataset_name):
        path = self.directory / f"{self.filename}_{dataset_name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Numpy store not found: {path}")
        data = np.load(path)
        return data["hidden_states"], data["labels"]

    def list_datasets(self) -> list:
        """Return the names of all stored datasets."""
        if self.format == "hdf5" and self.path.exists():
            with h5py.File(self.path, "r") as f:
                return list(f.keys())
        else:
            return [
                p.stem.replace(f"{self.filename}_", "")
                for p in self.directory.glob(f"{self.filename}_*.npz")
            ]
