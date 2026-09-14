"""A minimal, dependency-free stand-in for ComfyUI's ``Llama2_`` decoder stack.

The CVRR driver only relies on the calling convention ComfyUI uses:

    x, kv = layer(x=..., attention_mask=..., freqs_cis=...,
                  optimized_attention=..., past_key_value=None)

plus ``model.compute_freqs_cis(position_ids, device)`` and ``model.layers``.

``tests/test_comfy_integration.py`` additionally runs the driver on the *real*
ComfyUI Qwen3-VL text encoder with tiny dimensions; this module keeps the pure
algorithmic tests fast and readable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class TinyConfig:
    vocab_size: int = 64
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 8
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 8
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e4
    rope_dims: Optional[list] = None
    interleaved_mrope: bool = False
    qkv_bias: bool = False
    mlp_activation: str = "silu"


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(dtype)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Standard rotate-half RoPE; ``x`` is ``[B, H, L, D]`` and freqs ``[L, D/2]``."""
    cos = freqs.cos()[None, None]
    sin = freqs.sin()[None, None]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class TinyRMS(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):  # noqa: D102
        return rms_norm(x, self.weight, self.eps)


class TinyAttention(nn.Module):
    def __init__(self, config: TinyConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        inner = self.num_heads * self.head_dim
        kv = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, inner, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, kv, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv, bias=config.qkv_bias)
        self.o_proj = nn.Linear(inner, config.hidden_size, bias=False)
        self.q_norm = TinyRMS(self.head_dim, config.rms_norm_eps)
        self.k_norm = TinyRMS(self.head_dim, config.rms_norm_eps)
        self.scaling = self.head_dim ** -0.5

    def project(self, hidden_states: torch.Tensor):
        b, l, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, l, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, l, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return self.q_norm(q), self.k_norm(k), v

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        optimized_attention=None,
        past_key_value: Optional[Tuple] = None,
        sliding_window: Optional[int] = None,
    ):
        b, l, _ = hidden_states.shape
        q, k, v = self.project(hidden_states)
        if freqs_cis is not None:
            q = apply_rope(q, freqs_cis)
            k = apply_rope(k, freqs_cis)
        out = optimized_attention(q, k, v, self.num_heads, mask=attention_mask, skip_reshape=True,
                                  enable_gqa=self.num_heads != self.num_kv_heads)
        out = out.transpose(1, 2).reshape(b, l, -1)
        return self.o_proj(out), None


class TinyMLP(nn.Module):
    def __init__(self, config: TinyConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = torch.nn.functional.silu

    def forward(self, x):  # noqa: D102
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))


class TinyBlock(nn.Module):
    """Same call signature as ``comfy.text_encoders.llama.TransformerBlock``."""

    def __init__(self, config: TinyConfig, index: int = 0):
        super().__init__()
        self.index = index
        self.input_layernorm = TinyRMS(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = TinyRMS(config.hidden_size, config.rms_norm_eps)
        self.self_attn = TinyAttention(config)
        self.mlp = TinyMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        optimized_attention=None,
        past_key_value: Optional[Tuple] = None,
    ):
        residual = x
        hidden = self.input_layernorm(x)
        hidden, kv = self.self_attn(
            hidden_states=hidden,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            optimized_attention=optimized_attention,
            past_key_value=past_key_value,
        )
        x = residual + hidden
        residual = x
        x = residual + self.mlp(self.post_attention_layernorm(x))
        return x, kv


class TinyQwen3LM(nn.Module):
    """Stands in for ``comfy.text_encoders.llama.Llama2_``."""

    def __init__(self, config: TinyConfig, seed: int = 0):
        super().__init__()
        self.config = config
        torch.manual_seed(seed)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TinyBlock(config, i) for i in range(config.num_hidden_layers)])
        self.norm = TinyRMS(config.hidden_size, config.rms_norm_eps)

    def compute_freqs_cis(self, position_ids: torch.Tensor, device) -> torch.Tensor:
        """Plain (non-M-RoPE) frequencies: enough to exercise the driver.

        Accepts ComfyUI's ``(3, batch, seq)`` M-RoPE layout and reduces it to the
        first axis (the tests use identical positions on all three axes).
        """
        if position_ids.ndim == 3:
            position_ids = position_ids[0]
        keys = position_ids.reshape(-1, position_ids.shape[-1])
        position_ids = keys[0].reshape(-1)
        head_dim = self.config.head_dim
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        freqs = torch.outer(position_ids.float(), inv_freq)
        return freqs


def tiny_sequence(seq_len: int, hidden: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, seq_len, hidden, generator=generator) * scale
