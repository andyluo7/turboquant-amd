"""Walsh-Hadamard Transform rotation and weight fusion.

The WHT rotation transforms vectors into a space where coordinates are
approximately i.i.d. Gaussian, enabling near-optimal scalar quantization.
This module provides:
  - WHT matrix construction
  - Weight fusion (eliminating rotation overhead at inference time)

Key property: WHT is orthogonal and self-inverse (H @ H = I), so:
  - Forward rotation: x_rot = x @ H
  - Inverse rotation: x = x_rot @ H

Weight fusion folds H into the QKV and output projection weights:
  - W_q_new = PiT.T @ W_q  (Q comes out pre-rotated)
  - W_k_new = PiT.T @ W_k  (K comes out pre-rotated)
  - W_o_new = W_o @ Pi.T   (output de-rotates through O projection)

After fusion, rotation cost = 0.

References:
    - TurboQuant paper (arXiv: 2504.19874), Section 3.2
"""

import math
import torch


def make_wht_matrix(d: int, device: torch.device = None) -> torch.Tensor:
    """Construct a normalized Walsh-Hadamard Transform matrix.

    Args:
        d: Dimension (must be a power of 2).
        device: Target device.

    Returns:
        H: Orthogonal WHT matrix of shape [d, d], where H @ H = I.
    """
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < d:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0) / math.sqrt(2)
    return H[:d, :d]


def fuse_rotation_into_qkv_proj(
    qkv_weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> torch.Tensor:
    """Fuse WHT rotation into packed QKV projection weight.

    Args:
        qkv_weight: [Q | K | V] packed weight, shape [(nh + 2*nkv)*D, hidden].
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (for GQA).
        head_size: Head dimension.

    Returns:
        Modified qkv_weight with rotation fused into Q, K, and V portions.
    """
    PiT = make_wht_matrix(head_size, device=qkv_weight.device).to(qkv_weight.dtype)

    q_size = num_heads * head_size
    k_size = num_kv_heads * head_size
    v_size = num_kv_heads * head_size

    # Split and reshape
    q_weight = qkv_weight[:q_size].view(num_heads, head_size, -1)
    k_weight = qkv_weight[q_size:q_size + k_size].view(num_kv_heads, head_size, -1)
    v_weight = qkv_weight[q_size + k_size:].view(num_kv_heads, head_size, -1)

    # Fuse PiT.T into output dimension of each head
    for h in range(num_heads):
        q_weight[h] = PiT.T @ q_weight[h]
    for h in range(num_kv_heads):
        k_weight[h] = PiT.T @ k_weight[h]
        v_weight[h] = PiT.T @ v_weight[h]

    return torch.cat([
        q_weight.view(q_size, -1),
        k_weight.view(k_size, -1),
        v_weight.view(v_size, -1),
    ], dim=0)


def fuse_rotation_into_o_proj(
    o_weight: torch.Tensor,
    num_heads: int,
    head_size: int,
) -> torch.Tensor:
    """Fuse WHT de-rotation into output projection weight.

    Args:
        o_weight: Output projection, shape [hidden_size, num_heads * head_size].
        num_heads: Number of attention heads.
        head_size: Head dimension.

    Returns:
        Modified o_weight with de-rotation fused in.
    """
    Pi = make_wht_matrix(head_size, device=o_weight.device).to(o_weight.dtype)
    hidden_size = o_weight.shape[0]
    o_reshaped = o_weight.view(hidden_size, num_heads, head_size)

    for h in range(num_heads):
        o_reshaped[:, h, :] = o_reshaped[:, h, :] @ Pi.T

    return o_reshaped.view(hidden_size, num_heads * head_size)


def apply_weight_fusion(
    model_state_dict: dict,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> dict:
    """Apply rotation fusion to all attention layers in a model state dict.

    Modifies qkv_proj and o_proj weights in-place.

    Returns:
        Modified state dict.
    """
    modified = 0
    for key in list(model_state_dict.keys()):
        if "qkv_proj.weight" in key:
            model_state_dict[key] = fuse_rotation_into_qkv_proj(
                model_state_dict[key], num_heads, num_kv_heads, head_size
            )
            modified += 1
        elif "o_proj.weight" in key:
            model_state_dict[key] = fuse_rotation_into_o_proj(
                model_state_dict[key], num_heads, head_size
            )
            modified += 1

    return model_state_dict
