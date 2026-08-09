"""
realtime_detector.py
---------------------
Production inference wrapper: wraps any HuggingFace causal LM
with a trained probe and returns a hallucination probability
and verdict for every (question, answer) pair.

Adapted directly from the HallucinationDetector class in the PDF (Phase 7).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Wraps a frozen HuggingFace LLM with a trained probe for live detection.

    Latency overhead: ~0.5–2 ms per query (single matrix-vector multiply).
    Memory overhead: 4096 × 1 weight matrix = ~32 KB.

    Parameters
    ----------
    model         : frozen HuggingFace causal LM
    tokenizer     : matching tokenizer
    probe         : fitted sklearn probe (must support predict_proba)
    best_layer    : 0-indexed transformer layer where the hook is registered
    hallucination_threshold : probability above which we call HALLUCINATION
    uncertain_threshold     : probability below which we call LIKELY TRUTHFUL
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        probe,
        best_layer: int = 14,
        hallucination_threshold: float = 0.70,
        uncertain_threshold: float = 0.30,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.probe = probe
        self.best_layer = best_layer
        self.hallucination_threshold = hallucination_threshold
        self.uncertain_threshold = uncertain_threshold

        self.captured_state: Optional[torch.Tensor] = None

        # Ensure padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Register hook on best layer only
        self._hook_handle = self._register_hook()
        logger.info(
            "HallucinationDetector ready — layer=%d, thresholds=[%.2f, %.2f]",
            best_layer, uncertain_threshold, hallucination_threshold,
        )

    def _register_hook(self):
        """Register a single forward hook on the chosen best layer."""
        def hook_fn(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Last-token hidden state: [batch=1, seq_len, hidden_dim] → [1, hidden_dim]
            self.captured_state = hidden[:, -1, :].detach().cpu().float()

        # Support Llama/Mistral architecture
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layer = self.model.model.layers[self.best_layer]
        else:
            raise ValueError("Cannot locate transformer layers for hook registration.")

        return layer.register_forward_hook(hook_fn)

    def remove_hook(self) -> None:
        """Remove the registered hook (call when done to avoid memory leaks)."""
        self._hook_handle.remove()

    def detect(self, question: str, answer: str) -> dict:
        """
        Run a single (question, answer) pair through the LLM and return
        the hallucination probability and verdict.

        Parameters
        ----------
        question : str
        answer   : str

        Returns
        -------
        dict with keys: answer, hallucination_probability, verdict
        """
        text = f"Question: {question}\nAnswer: {answer}"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(next(self.model.parameters()).device)

        with torch.no_grad():
            _ = self.model(**inputs)

        if self.captured_state is None:
            raise RuntimeError("No hidden state captured — hook may have failed.")

        h = self.captured_state.numpy()  # [1, hidden_dim]
        prob_hallucination = float(self.probe.predict_proba(h)[0, 1])

        if prob_hallucination > self.hallucination_threshold:
            verdict = "🚫 HALLUCINATION"
        elif prob_hallucination > self.uncertain_threshold:
            verdict = "⚠️  UNCERTAIN"
        else:
            verdict = "✅ LIKELY TRUTHFUL"

        return {
            "answer"                 : answer,
            "hallucination_probability": round(prob_hallucination, 4),
            "verdict"                : verdict,
        }

    def __del__(self):
        try:
            self.remove_hook()
        except Exception:
            pass

    # ── Factory: load from disk ───────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        probe_path: str | Path,
        best_layer: int = 14,
        torch_dtype=torch.float16,
        device_map: str = "auto",
        **kwargs,
    ) -> "HallucinationDetector":
        """
        Convenience factory that loads model, tokenizer, and probe from paths.

        Parameters
        ----------
        model_name  : HuggingFace model ID or local path
        probe_path  : path to joblib-serialised probe (.pkl)
        best_layer  : 0-indexed layer (default 14 = layer 15)
        """
        logger.info("Loading model: %s", model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        probe = joblib.load(probe_path)
        logger.info("Probe loaded from %s", probe_path)
        return cls(model, tokenizer, probe, best_layer=best_layer, **kwargs)
