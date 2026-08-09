"""
hook_extractor.py
-----------------
Registers PyTorch forward hooks on every transformer layer of a
frozen LLM and captures the last-token hidden state at each layer.

Design notes
------------
* We capture output[0][:, -1, :] — the last-token hidden state —
  because in a causal (decoder-only) transformer every token attends
  to all preceding tokens, so the final token aggregates the richest
  context.
* Hooks are stored as instance state so they can be cleanly removed
  after extraction via `remove_hooks()`.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class HiddenStateExtractor:
    """
    Attaches forward hooks to all (or selected) transformer layers
    of a HuggingFace causal LM and captures the last-token hidden state.

    Usage
    -----
    extractor = HiddenStateExtractor(model)
    extractor.register_hooks()
    _ = model(**inputs)               # forward pass triggers hooks
    states = extractor.get_states()   # dict[layer_idx -> Tensor[hidden_dim]]
    extractor.remove_hooks()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_indices: Optional[List[int]] = None,
        token_position: str = "last",
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Frozen HuggingFace causal LM (Llama / Mistral / GPT-2 etc.)
        layer_indices : list[int] | None
            Which transformer layers to hook.  None → hook all layers.
        token_position : str
            "last"  → hidden state of the final token (recommended)
            "first" → hidden state of the first token (BOS)
            "mean"  → mean-pool over all tokens
        """
        self.model = model
        self.token_position = token_position
        self._captured: Dict[int, Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        # Detect the list of transformer layers
        self._layers: List[torch.nn.Module] = self._detect_layers()
        if layer_indices is None:
            self.layer_indices = list(range(len(self._layers)))
        else:
            self.layer_indices = layer_indices

        logger.info(
            "HiddenStateExtractor initialised — %d layers detected, "
            "hooked on %d of them, token_position=%s",
            len(self._layers),
            len(self.layer_indices),
            token_position,
        )

    # ── Internal helpers ─────────────────────────────────────────────

    def _detect_layers(self) -> List[torch.nn.Module]:
        """Heuristically find the list of transformer decoder layers."""
        # Llama / Mistral: model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        # GPT-2 style: model.transformer.h
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            return list(self.model.transformer.h)
        raise ValueError(
            "Cannot detect transformer layer list for this model architecture. "
            "Supported: Llama/Mistral (model.model.layers), GPT-2 (model.transformer.h)."
        )

    def _make_hook(self, layer_idx: int):
        """Factory that returns a hook closure capturing `layer_idx`."""

        def hook_fn(module: torch.nn.Module, inp, output):
            # output is typically a tuple; output[0] is the hidden state tensor
            # shape: [batch_size, seq_len, hidden_dim]
            hidden = output[0] if isinstance(output, tuple) else output

            if self.token_position == "last":
                captured = hidden[:, -1, :]
            elif self.token_position == "first":
                captured = hidden[:, 0, :]
            elif self.token_position == "mean":
                captured = hidden.mean(dim=1)
            else:
                raise ValueError(f"Unknown token_position: {self.token_position}")

            self._captured[layer_idx] = captured.detach().cpu().float()

        return hook_fn

    # ── Public API ────────────────────────────────────────────────────

    def register_hooks(self) -> None:
        """Register forward hooks on all selected layers."""
        self._captured.clear()
        self._hooks.clear()
        for idx in self.layer_indices:
            handle = self._layers[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(handle)
        logger.debug("Registered %d forward hooks.", len(self._hooks))

    def remove_hooks(self) -> None:
        """Remove all registered hooks (call after extraction to free memory)."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        logger.debug("All forward hooks removed.")

    def get_states(self) -> Dict[int, Tensor]:
        """
        Return the captured hidden states from the last forward pass.

        Returns
        -------
        dict mapping layer_idx (int) → Tensor of shape [batch_size, hidden_dim]
        """
        if not self._captured:
            raise RuntimeError(
                "No hidden states captured yet. "
                "Call register_hooks(), run a forward pass, then call get_states()."
            )
        return dict(self._captured)

    def clear(self) -> None:
        """Clear captured states without removing hooks (use between batches)."""
        self._captured.clear()

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    @property
    def num_hooked_layers(self) -> int:
        return len(self.layer_indices)
