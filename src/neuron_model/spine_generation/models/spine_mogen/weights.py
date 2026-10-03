"""Flax (MoGen checkpoint) <-> PyTorch (``PointInfinity``) weight conversion.

Input: an ``.npz`` produced by ``scripts/neuron_model/export_mogen_weights.py``
(run in the separate JAX environment) whose keys are ``/``-joined Flax paths,
e.g. ``ema_params/PointInfinityBlock_0/MultiHeadDotProductAttention_0/query/kernel``.

Layout differences handled here:

- ``Dense.kernel [in, out]``                      -> ``Linear.weight [out, in]`` (transpose)
- ``DenseGeneral`` q/k/v ``kernel [in, H, D]``      -> ``[H*D, in]``; ``bias [H, D]`` -> ``[H*D]``
- ``DenseGeneral`` out ``kernel [H, D, out]``       -> ``[out, H*D]``
- ``Embed.embedding [n, d]``                      -> ``Embedding.weight [n, d]``
- ``PointInfinityBlock_<i>``                      -> ``blocks.<i>``

The model configuration (dims, heads, k_nn, cond width) is inferred from the
array shapes, so a checkpoint is never loaded into a mismatched architecture.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch

from .pointinfinity import PointInfinity, PointInfinityConfig

_BLOCK_RE = re.compile(r"^PointInfinityBlock_(\d+)$")
_LEAF = {"kernel": "weight", "bias": "bias", "scale": "scale", "embedding": "weight", "param": "param"}


def load_flax_npz(path: Path, collection: str = "ema_params") -> Dict[str, np.ndarray]:
    """``{flax path without the collection prefix: array}`` for one collection (``params`` / ``ema_params``)."""
    prefix = collection + "/"
    with np.load(path) as data:
        params = {key[len(prefix):]: np.asarray(data[key]) for key in data.files if key.startswith(prefix)}
    if not params:
        raise KeyError(f"No '{collection}' arrays in {path}")
    return params


def infer_config(flax_params: Mapping[str, np.ndarray]) -> PointInfinityConfig:
    tokenizer = flax_params["Dense_1/kernel"]
    embed = flax_params["Embed_0/embedding"]
    blocks = sorted({int(m.group(1)) for key in flax_params if (m := _BLOCK_RE.match(key.split("/")[0]))})
    attn = {key.split("/")[1] for key in flax_params if key.startswith("PointInfinityBlock_0/MultiHeadDotProductAttention_")}
    n_heads = flax_params["PointInfinityBlock_0/MultiHeadDotProductAttention_0/query/kernel"].shape[1]
    coord_dim = flax_params["Dense_2/kernel"].shape[1]
    token_in = tokenizer.shape[0]
    if token_in % coord_dim:
        raise ValueError(f"Tokenizer input width {token_in} is not a multiple of coord_dim {coord_dim} (point_cond / features not supported)")
    return PointInfinityConfig(
        point_dim=int(tokenizer.shape[1]),
        latent_dim=int(embed.shape[1]),
        n_latents=int(embed.shape[0]),
        n_blocks=len(blocks),
        n_subblocks=len(attn) - 2,
        n_heads=int(n_heads),
        k_nn=token_in // coord_dim - 1,
        coord_dim=int(coord_dim),
        cond_dim=int(flax_params["Dense_0/kernel"].shape[0]) if "Dense_0/kernel" in flax_params else 0,
    )


def torch_name(flax_key: str) -> str:
    parts = flax_key.split("/")
    match = _BLOCK_RE.match(parts[0])
    head = [f"blocks.{match.group(1)}"] if match else [parts[0]]
    return ".".join(head + parts[1:-1] + [_LEAF[parts[-1]]])


def flax_to_torch_array(flax_key: str, array: np.ndarray) -> np.ndarray:
    leaf = flax_key.rsplit("/", 1)[-1]
    parent = flax_key.split("/")[-2]
    if leaf == "kernel":
        if array.ndim == 2:
            return array.T
        if array.ndim == 3 and parent in ("query", "key", "value"):
            return array.reshape(array.shape[0], -1).T
        if array.ndim == 3 and parent == "out":
            return array.reshape(-1, array.shape[-1]).T
        raise ValueError(f"Unexpected kernel shape {array.shape} at {flax_key}")
    if leaf == "bias" and array.ndim == 2:
        return array.reshape(-1)
    return array


def flax_to_state_dict(flax_params: Mapping[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    return {torch_name(key): torch.from_numpy(np.ascontiguousarray(flax_to_torch_array(key, value)).astype(np.float32)) for key, value in flax_params.items()}


def load_pointinfinity_from_flax(flax_params: Mapping[str, np.ndarray]) -> PointInfinity:
    """Build a ``PointInfinity`` matching the checkpoint and load it strictly (every tensor accounted for)."""
    model = PointInfinity(infer_config(flax_params))
    state = flax_to_state_dict(flax_params)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise KeyError(f"Flax->torch name mismatch. missing={missing[:10]} unexpected={unexpected[:10]}")
    for name, tensor in state.items():
        if tuple(tensor.shape) != tuple(expected[name].shape):
            raise ValueError(f"Shape mismatch for {name}: checkpoint {tuple(tensor.shape)} vs model {tuple(expected[name].shape)}")
    model.load_state_dict(state, strict=True)
    return model


def state_dict_to_flax(model: PointInfinity) -> Dict[str, np.ndarray]:
    """Inverse conversion (torch -> Flax layout); used to test that the mapping is a bijection."""
    cfg = model.cfg
    result: Dict[str, np.ndarray] = {}
    for name, tensor in model.state_dict().items():
        parts = name.split(".")
        if parts[0] == "blocks":
            parts = [f"PointInfinityBlock_{parts[1]}"] + parts[2:]
        leaf = parts[-1]
        array = tensor.detach().cpu().numpy()
        flax_leaf = {"weight": "embedding" if parts[0] == "Embed_0" else "kernel", "bias": "bias", "scale": "scale", "param": "param"}[leaf]
        parent = parts[-2]
        if flax_leaf == "kernel":
            if parent in ("query", "key", "value"):
                array = array.T.reshape(array.shape[1], cfg.n_heads, -1)
            elif parent == "out":
                array = array.T.reshape(cfg.n_heads, -1, array.shape[0])
            else:
                array = array.T
        elif flax_leaf == "bias" and parent in ("query", "key", "value"):
            array = array.reshape(cfg.n_heads, -1)
        result["/".join(parts[:-1] + [flax_leaf])] = array
    return result
