"""
batch_runner.py
---------------
Runs batched forward passes over the full dataset and collects
hidden-state tensors from every transformer layer via HiddenStateExtractor.

Output
------
all_hidden_states : np.ndarray  shape [N, num_layers, hidden_dim]
all_labels        : np.ndarray  shape [N]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from extraction.hook_extractor import HiddenStateExtractor
from extraction.storage import HiddenStateStore

logger = logging.getLogger(__name__)


# ── Tiny dataset wrapper ──────────────────────────────────────────────

class QADataset(Dataset):
    """Minimal wrapper around a list of (text, label) pairs."""

    def __init__(self, samples: List[Tuple[str, int]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.samples[idx]


# ── Collate function ─────────────────────────────────────────────────

def collate_fn(batch, tokenizer: PreTrainedTokenizer, max_length: int = 512):
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    encoding = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return encoding, labels


# ── Main extraction runner ────────────────────────────────────────────

class BatchRunner:
    """
    Orchestrates batched forward passes and hidden-state collection.

    Parameters
    ----------
    model          : Frozen HuggingFace causal LM
    tokenizer      : Matching tokenizer
    extractor      : HiddenStateExtractor (hooks already registered)
    batch_size     : int — keep ≤16 for A100, ≤4 for T4
    max_length     : int — max token length before truncation
    device         : torch.device — inferred from model if None
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        extractor: HiddenStateExtractor,
        batch_size: int = 8,
        max_length: int = 512,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or next(model.parameters()).device

        # Ensure padding token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.warning("pad_token not set — using eos_token as pad_token.")

    def run(
        self,
        samples: List[Tuple[str, int]],
        store: Optional[HiddenStateStore] = None,
        dataset_name: str = "unknown",
        checkpoint_interval: int = 100,
        resume: bool = True,
        return_in_memory: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run forward passes over all samples and return hidden states.
        Supports incremental checkpointing to disk and resuming on interruption.

        Parameters
        ----------
        samples             : list of (text, label) tuples
        store               : HiddenStateStore — if provided, saves tensors to disk incrementally
        dataset_name        : tag written to the store (e.g. "truthfulqa")
        checkpoint_interval : int — save checkpoint every N samples
        resume              : bool — if True, skip already extracted samples from store
        return_in_memory    : bool — if False and store is provided, do NOT load full dataset into CPU RAM

        Returns
        -------
        all_hidden_states   : np.ndarray | None  [N, num_layers, hidden_dim]
        all_labels          : np.ndarray | None  [N]
        """
        start_idx = 0
        if resume and store is not None:
            already_done = store.get_num_samples(dataset_name)
            if already_done > 0:
                if already_done >= len(samples):
                    logger.info("Dataset [%s] already fully extracted (%d samples).", dataset_name, already_done)
                    if return_in_memory:
                        return store.load(dataset_name)
                    return None, None
                logger.info("Resuming extraction for [%s] from sample index %d / %d", dataset_name, already_done, len(samples))
                start_idx = already_done

        remaining_samples = samples[start_idx:]
        dataset = QADataset(remaining_samples)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, self.tokenizer, self.max_length),
            num_workers=0,   # keep 0 for GPU workloads
        )

        chunk_states: List[np.ndarray] = []
        chunk_labels: List[int] = []

        self.model.eval()
        self.extractor.register_hooks()

        try:
            for batch_idx, (encoding, labels) in enumerate(
                tqdm(loader, desc=f"Extracting [{dataset_name}]", initial=start_idx // self.batch_size, total=(len(samples) + self.batch_size - 1) // self.batch_size)
            ):
                encoding = {k: v.to(self.device) for k, v in encoding.items()}

                with torch.no_grad():
                    _ = self.model(**encoding)

                states_dict = self.extractor.get_states()  # {layer_idx: [B, H]}
                num_layers = len(states_dict)
                batch_size_actual = list(states_dict.values())[0].shape[0]

                batch_states = np.zeros(
                    (batch_size_actual, num_layers, list(states_dict.values())[0].shape[1]),
                    dtype=np.float32,
                )
                for layer_idx, tensor in states_dict.items():
                    batch_states[:, layer_idx, :] = tensor.numpy()

                chunk_states.append(batch_states)
                chunk_labels.extend(labels)
                self.extractor.clear()

                # Checkpoint chunk to disk if threshold reached
                current_chunk_len = sum(s.shape[0] for s in chunk_states)
                if store is not None and current_chunk_len >= checkpoint_interval:
                    c_states = np.concatenate(chunk_states, axis=0)
                    c_labels = np.array(chunk_labels, dtype=np.int32)
                    store.append(c_states, c_labels, dataset_name)
                    chunk_states.clear()
                    chunk_labels.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        finally:
            self.extractor.remove_hooks()

        # Save remaining leftover samples in last chunk
        if store is not None and chunk_states:
            c_states = np.concatenate(chunk_states, axis=0)
            c_labels = np.array(chunk_labels, dtype=np.int32)
            store.append(c_states, c_labels, dataset_name)
            chunk_states.clear()
            chunk_labels.clear()

        if store is not None:
            if return_in_memory:
                return store.load(dataset_name)
            else:
                total_samples = store.get_num_samples(dataset_name)
                logger.info("Extraction complete: [%s] saved to disk (%d samples). CPU RAM conserved.", dataset_name, total_samples)
                return None, None
        else:
            if not chunk_states:
                raise RuntimeError("No samples extracted and no store provided.")
            hidden_states = np.concatenate(chunk_states, axis=0)
            labels_arr = np.array(chunk_labels, dtype=np.int32)
            return hidden_states, labels_arr
