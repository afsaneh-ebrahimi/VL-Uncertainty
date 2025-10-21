"""Utilities for geometric hallucination detection scores."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

Metric = Literal["cosine", "euclidean"]


def _pairwise_distance(
    query: torch.Tensor,
    reference: torch.Tensor,
    metric: Metric,
) -> torch.Tensor:
    """Compute pairwise distances between query and reference embeddings."""
    if metric == "cosine":
        query_norm = F.normalize(query, dim=-1)
        reference_norm = F.normalize(reference, dim=-1)
        distances = 1.0 - query_norm @ reference_norm.T
    elif metric == "euclidean":
        distances = torch.cdist(query, reference, p=2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return distances


def knn_distance(
    query: torch.Tensor,
    reference: torch.Tensor,
    k: int,
    metric: Metric = "cosine",
) -> torch.Tensor:
    """Return the mean distance to the k nearest reference neighbours for each query."""
    if reference.numel() == 0:
        raise ValueError("Reference embeddings tensor must not be empty.")
    effective_k = min(max(k, 1), reference.shape[0])
    distances = _pairwise_distance(query, reference, metric)
    knn_vals, _ = torch.topk(distances, k=effective_k, largest=False, dim=-1)
    return knn_vals.mean(dim=-1)


def lid_mle(
    query: torch.Tensor,
    reference: torch.Tensor,
    k: int,
    metric: Metric = "cosine",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Estimate Local Intrinsic Dimensionality (LID) for each query embedding."""
    if k < 2:
        raise ValueError("lid_mle requires k >= 2 to compute a valid estimate.")
    if reference.numel() == 0:
        raise ValueError("Reference embeddings tensor must not be empty.")
    effective_k = min(k, reference.shape[0])
    if effective_k < 2:
        raise ValueError("At least two reference embeddings are required for LID.")
    distances = _pairwise_distance(query, reference, metric)
    knn_vals, _ = torch.topk(distances, k=effective_k, largest=False, dim=-1)
    knn_vals = knn_vals.clamp_min(eps)
    r_k = knn_vals[:, -1:].clamp_min(eps)
    log_ratios = torch.log(knn_vals[:, :-1] / r_k)
    lid = -(effective_k - 1) / torch.sum(log_ratios, dim=-1)
    return lid


__all__ = ["knn_distance", "lid_mle"]
