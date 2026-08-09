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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run forward passes over all samples and return hidden states.

        Parameters
        ----------
        samples       : list of (text, label) tuples
        store         : HiddenStateStore — if provided, saves tensors to disk
        dataset_name  : tag written to the store (e.g. "truthfulqa")

        Returns
        -------
        all_hidden_states : np.ndarray  [N, num_layers, hidden_dim]
        all_labels        : np.ndarray  [N]
        """
        dataset = QADataset(samples)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, self.tokenizer, self.max_length),
            num_workers=0,   # keep 0 for GPU workloads
        )

        all_states: List[np.ndarray] = []
        all_labels: List[int] = []

        self.model.eval()
        self.extractor.register_hooks()

        try:
            for batch_idx, (encoding, labels) in enumerate(
                tqdm(loader, desc=f"Extracting [{dataset_name}]")
            ):
                encoding = {k: v.to(self.device) for k, v in encoding.items()}

                with torch.no_grad():
                    _ = self.model(**encoding)

                states_dict = self.extractor.get_states()  # {layer_idx: [B, H]}
                # Stack layers → [B, num_layers, hidden_dim]
                num_layers = len(states_dict)
                batch_size_actual = list(states_dict.values())[0].shape[0]

                batch_states = np.zeros(
                    (batch_size_actual, num_layers, list(states_dict.values())[0].shape[1]),
                    dtype=np.float32,
                )
                for layer_idx, tensor in states_dict.items():
                    batch_states[:, layer_idx, :] = tensor.numpy()

                all_states.append(batch_states)
                all_labels.extend(labels)
                self.extractor.clear()

                if batch_idx % 50 == 0:
                    logger.debug(
                        "Processed %d / %d samples",
                        min((batch_idx + 1) * self.batch_size, len(samples)),
                        len(samples),
                    )
        finally:
            self.extractor.remove_hooks()

        hidden_states = np.concatenate(all_states, axis=0)  # [N, L, H]
        labels_arr = np.array(all_labels, dtype=np.int32)

        logger.info(
            "Extraction complete: %s — shape=%s, labels=%s",
            dataset_name,
            hidden_states.shape,
            np.bincount(labels_arr),
        )

        if store is not None:
            store.save(hidden_states, labels_arr, dataset_name)

        return hidden_states, labels_arr
