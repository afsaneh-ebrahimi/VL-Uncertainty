"""Helpers for saving and loading reference embedding sets."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import torch


def save_reference_set(
    path: str,
    embeddings: torch.Tensor,
    metadata: List[Dict[str, Any]] | None = None,
    config: Dict[str, Any] | None = None,
) -> None:
    """Persist a reference embedding set to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "embeddings": embeddings.detach().cpu(),
        "metadata": metadata or [],
        "config": config or {},
    }
    torch.save(payload, path)


def load_reference_set(path: str) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    """Load a reference embedding set from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference set not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if "embeddings" not in payload:
        raise ValueError("Reference payload missing 'embeddings' key.")
    embeddings = payload["embeddings"].float()
    metadata = payload.get("metadata", [])
    config = payload.get("config", {})
    if embeddings.ndim != 2:
        raise ValueError("Reference embeddings must be a 2D tensor.")
    return embeddings, metadata, config


__all__ = ["save_reference_set", "load_reference_set"]
