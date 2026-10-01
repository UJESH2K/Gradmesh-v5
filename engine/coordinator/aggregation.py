"""Contribution-weighted FedAvg.

federated_training.average_state_dicts takes an unweighted mean, which is only
correct when every shard is the same size. GradMesh hands workers unequal
shards on purpose, so aggregation weights each state dict by the samples behind
it. That is the original FedAvg estimator:

    w_global = sum_i (n_i / n_total) * w_i

v5 changes two things about how, not what.

* Updates arrive as raw serialised bytes instead of base64 text, a third
  smaller on the wire and with no second copy in memory.
* The average is accumulated one update at a time. v4 decoded every worker's
  state dict before summing, so host memory grew with the number of machines;
  now it holds one model plus one running sum however large the mesh gets.

Backend does not matter here. Workers serialise on CPU (see
federated_training.state_dict_to_bytes), so a state dict trained on Metal, on
an Arc GPU and on a CUDA card average exactly the same way.

torch is imported lazily so the coordinator can boot and serve the dashboard
before the training-plane wheels have finished installing in the background.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

from federated_training import decode_state_dict, state_dict_from_bytes, state_dict_to_bytes

Payload = Union[bytes, bytearray, str]


def torch_ready() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _decode(payload: Payload) -> Dict[str, Any]:
    if isinstance(payload, (bytes, bytearray)):
        return state_dict_from_bytes(bytes(payload))
    return decode_state_dict(payload)


def weighted_average_state_dicts(payloads: Sequence[Payload], weights: Sequence[float]) -> Dict[str, Any]:
    """Weighted mean of serialised state dicts. Weights are normalised here."""
    if not payloads:
        raise ValueError("At least one state dict is required")
    if len(payloads) != len(weights):
        raise ValueError("Each state dict needs exactly one weight")

    total = float(sum(weights))
    normalised = [float(weight) / total for weight in weights] if total > 0 else [1.0 / len(weights)] * len(weights)

    reference: Dict[str, Any] = {}
    sums: Dict[str, Any] = {}
    weight_sums: Dict[str, float] = {}

    for index, (payload, weight) in enumerate(zip(payloads, normalised)):
        state = _decode(payload)
        if index == 0:
            reference = state
        for key, reference_value in reference.items():
            if not hasattr(reference_value, "detach"):
                continue
            value = state.get(key)
            if value is None or not hasattr(value, "detach"):
                continue
            if tuple(value.shape) != tuple(reference_value.shape):
                # A head reshaped by a different class count. Skip rather than
                # crash, exactly as the worker does when loading weights.
                continue
            contribution = value.detach().float().cpu() * weight
            sums[key] = contribution if key not in sums else sums[key] + contribution
            weight_sums[key] = weight_sums.get(key, 0.0) + weight
        if index > 0:
            del state

    averaged: Dict[str, Any] = {}
    for key, reference_value in reference.items():
        if key in sums and weight_sums.get(key, 0.0) > 0:
            # Renormalise in case some workers were missing this tensor.
            averaged[key] = (sums[key] / weight_sums[key]).to(reference_value.dtype)
        else:
            averaged[key] = reference_value
    return averaged


def aggregate(results: Sequence[dict], weights: Dict[str, float]) -> bytes:
    """Aggregate finished round results into new serialised global weights."""
    ordered: List[dict] = [result for result in results if result.get("weights")]
    if not ordered:
        raise ValueError("No completed shards carried weights")
    payloads = [result["weights"] for result in ordered]
    weight_vector = [weights.get(result["batch_id"], 0.0) for result in ordered]
    return state_dict_to_bytes(weighted_average_state_dicts(payloads, weight_vector))
