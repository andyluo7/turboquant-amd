"""TurboQuant FP4 KV Cache Allocation Patch for vLLM v0.18.

Monkey-patches vLLM v0.18's KV cache allocation pipeline to use FP4
(68 bytes per K/V head per token position) instead of standard FP8/BF16
(128+ bytes). This yields ~1.88x KV cache capacity with the same GPU memory.

FP4 layout per head per position:
  - 64 bytes: packed FP4 E2M1 data (head_dim=128, 2 values per byte)
  - 4 bytes:  E8M0 scales (one per 32-element group, 128/32=4)
  - Total: 68 bytes vs 128 bytes (FP8) or 256 bytes (BF16)

What gets patched:
  1. FullAttentionSpec.real_page_size_bytes → FP4-aware page size calculation
  2. AttentionSpec.real_page_size_bytes    → FP4-aware page size (base class)
  3. _reshape_kv_cache()                  → view as int8 (not model dtype) for FP4
  4. Integration with TurboQuantFP4Backend.get_kv_cache_shape()

Activation:
  Set TQ_FP4_ENABLE=1 environment variable before launching vLLM.
  Without it, all patches are no-ops (original behavior preserved).

Usage:
  # From .pth autoloader (runs in ALL Python processes):
  from turboquant.integration.vllm_fp4_cache_patch import apply_fp4_cache_patch
  apply_fp4_cache_patch()

  # Or from patch_vllm_fp4() in tq_fp4_backend.py:
  from turboquant.integration.vllm_fp4_cache_patch import apply_fp4_cache_patch
  apply_fp4_cache_patch()

Architecture note:
  This module patches at the allocation level (how many bytes per page,
  how the raw tensor is reshaped into the KV cache tensor). The actual
  FP4 compression/decompression happens in tq_fp4_backend.py's
  TurboQuantFP4Impl.forward() and do_kv_cache_update().
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

logger = logging.getLogger("turboquant.fp4_cache_patch")

# ============================================================================
# Constants
# ============================================================================

# FP4 E2M1 packed bytes per head position:
#   head_dim // 2 = 64 bytes (packed data, 2 FP4 values per byte)
#   head_dim // 32 = 4 bytes (E8M0 scales, one per 32-element group)
#   Total = 68 bytes for head_dim=128
#
# This is parameterized below so it works for any head_dim that's a
# multiple of 32, but the canonical value for MiniMax-M2.5 etc. is 68.
FP4_BYTES_PER_HEAD = 68  # for head_dim=128
_DEFAULT_HEAD_DIM = 128


def fp4_bytes_per_head(head_dim: int = _DEFAULT_HEAD_DIM) -> int:
    """Compute FP4 packed bytes per head for a given head_dim.

    Layout: (head_dim // 2) packed data + (head_dim // 32) E8M0 scales.
    """
    assert head_dim % 32 == 0, f"head_dim must be multiple of 32, got {head_dim}"
    return head_dim // 2 + head_dim // 32


def is_fp4_enabled() -> bool:
    """Check if TurboQuant FP4 mode is active."""
    return os.environ.get("TQ_FP4_ENABLE", "0") == "1"


# ============================================================================
# Patch State
# ============================================================================

_FP4_CACHE_PATCHED = False

# Store original functions/properties for clean unpatching
_originals: dict[str, object] = {}


# ============================================================================
# Patch 1: AttentionSpec.real_page_size_bytes
# ============================================================================

def _make_fp4_page_size_property(original_property):
    """Create a new property that returns FP4-aware page size.

    For AttentionSpec (base class), the formula is:
      Original: 2 * block_size * num_kv_heads * head_size * dtype_size
      FP4:      2 * block_size * num_kv_heads * fp4_bytes_per_head(head_size)

    For FullAttentionSpec, the formula is:
      Original: block_size * num_kv_heads * (head_size + head_size_v) * dtype_size
      FP4:      block_size * num_kv_heads * (fp4_bytes + fp4_bytes_v)
           where fp4_bytes = fp4_bytes_per_head(head_size)
    """

    def _fp4_real_page_size_bytes(self):
        if not is_fp4_enabled():
            # Fall through to original
            return original_property.fget(self)

        fp4_bytes = fp4_bytes_per_head(self.head_size)

        # Check if this is FullAttentionSpec (has head_size_v)
        head_size_v = getattr(self, "head_size_v", None)
        if head_size_v is not None and head_size_v != self.head_size:
            # Different K/V head sizes (rare but possible)
            fp4_bytes_v = fp4_bytes_per_head(head_size_v)
            page_size = (
                self.block_size * self.num_kv_heads * (fp4_bytes + fp4_bytes_v)
            )
        else:
            # Same K/V head sizes (common case)
            # 2 * block_size * num_kv_heads * fp4_bytes
            page_size = 2 * self.block_size * self.num_kv_heads * fp4_bytes

        logger.debug(
            "[TQ-FP4] page_size_bytes: %d (block_size=%d, kv_heads=%d, "
            "head_size=%d, fp4_bytes=%d)",
            page_size,
            self.block_size,
            self.num_kv_heads,
            self.head_size,
            fp4_bytes,
        )
        return page_size

    return property(_fp4_real_page_size_bytes)


# ============================================================================
# Patch 2: _reshape_kv_cache — use int8 view for FP4 cache tensors
# ============================================================================

def _make_patched_reshape_kv_cache(original_fn):
    """Wrap _reshape_kv_cache to handle FP4 dtype correctly.

    The original function does:
      dtype = kv_cache_spec.dtype   # e.g., bfloat16 (2 bytes)
      raw_tensor = raw_tensor.view(dtype)
      raw_tensor = raw_tensor.view(kv_cache_shape)

    For FP4, the raw tensor is int8 and should stay as int8 (1 byte per
    element), since the KV cache shape already accounts for the packed
    FP4 byte layout. Viewing as bfloat16 would halve the element count
    and break the shape calculation.

    Strategy: When FP4 is enabled, we intercept the reshape to use
    torch.int8 as the view dtype instead of the spec's model dtype.
    """
    import torch

    def _patched_reshape_kv_cache(kv_cache_config, kv_cache_raw_tensors,
                                  attn_backends):
        if not is_fp4_enabled():
            return original_fn(
                kv_cache_config, kv_cache_raw_tensors, attn_backends
            )

        # FP4 path: manually reshape with int8 view
        from vllm.v1.kv_cache_interface import AttentionSpec

        kv_caches = {}
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group_spec.kv_cache_spec
            if not isinstance(kv_cache_spec, AttentionSpec):
                # Non-attention specs (Mamba etc.) — use original
                for layer_name in kv_cache_group_spec.layer_names:
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    # Fall back to the original for non-attention layers
                    # (This is a simplified path; full models with mixed
                    #  attention+mamba layers would need more care)
                    kv_caches[layer_name] = raw_tensor
                continue

            for layer_name in kv_cache_group_spec.layer_names:
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0, (
                    f"[TQ-FP4] Raw tensor size {raw_tensor.numel()} not "
                    f"divisible by page_size_bytes {kv_cache_spec.page_size_bytes}"
                )
                num_blocks = (
                    raw_tensor.numel() // kv_cache_spec.page_size_bytes
                )

                attn_backend = attn_backends[layer_name]
                kv_cache_shape = attn_backend.get_kv_cache_shape(
                    num_blocks,
                    kv_cache_spec.block_size,
                    kv_cache_spec.num_kv_heads,
                    kv_cache_spec.head_size,
                )

                # Handle stride ordering (same as original)
                try:
                    kv_cache_stride_order = (
                        attn_backend.get_kv_cache_stride_order()
                    )
                    assert len(kv_cache_stride_order) == len(kv_cache_shape)
                except (AttributeError, NotImplementedError):
                    kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

                kv_cache_shape = tuple(
                    kv_cache_shape[i] for i in kv_cache_stride_order
                )
                inv_order = [
                    kv_cache_stride_order.index(i)
                    for i in range(len(kv_cache_stride_order))
                ]

                # KEY DIFFERENCE: For FP4, view as int8 (same as raw dtype)
                # instead of the model's compute dtype (bfloat16/float16).
                # The raw tensor is already int8 with 1 byte per element.
                # FP4 packed data is uint8-compatible (same byte size as int8).
                #
                # We keep int8 view because:
                #   - raw_tensor is allocated as torch.int8
                #   - FP4 packed data is byte-level (uint8/int8 interchangeable)
                #   - The shape already accounts for FP4 byte packing
                #   - Viewing as bfloat16 would halve element count and break
                view_dtype = torch.int8  # NOT kv_cache_spec.dtype

                viewed = raw_tensor.view(view_dtype)
                viewed = viewed.view(kv_cache_shape)
                kv_caches[layer_name] = viewed.permute(*inv_order)

                logger.info(
                    "[TQ-FP4] Reshaped KV cache for %s: shape=%s, "
                    "dtype=%s, num_blocks=%d",
                    layer_name,
                    tuple(kv_caches[layer_name].shape),
                    kv_caches[layer_name].dtype,
                    num_blocks,
                )

        return kv_caches

    return _patched_reshape_kv_cache


# ============================================================================
# Patch 3: get_kv_cache_shape override for FP4
# ============================================================================

def fp4_get_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype_str: str = "auto",
) -> tuple[int, ...]:
    """KV cache shape for FP4 packing.

    Shape: (2, num_blocks, block_size, num_kv_heads, fp4_bytes_per_head)
    where:
      - 2 = K and V caches stacked
      - fp4_bytes_per_head = head_size//2 + head_size//32
                           = 64 + 4 = 68 for head_size=128

    The tensor dtype is int8/uint8, so each element is one byte.
    This interleaves packed FP4 data and E8M0 scales in the last dimension:
      [0:head_size//2]              = packed FP4 E2M1 data
      [head_size//2:fp4_bytes]      = E8M0 scale bytes
    """
    if block_size % 16 != 0:
        raise ValueError("Block size must be a multiple of 16.")
    fp4_bytes = fp4_bytes_per_head(head_size)
    return (2, num_blocks, block_size, num_kv_heads, fp4_bytes)


# ============================================================================
# Main Patch Application
# ============================================================================

def apply_fp4_cache_patch() -> bool:
    """Apply FP4 KV cache allocation patches to vLLM v0.18.

    This function monkey-patches three components:
      1. AttentionSpec.real_page_size_bytes (property on frozen dataclass)
      2. FullAttentionSpec.real_page_size_bytes (overridden property)
      3. _reshape_kv_cache() function in attn_utils

    All patches check TQ_FP4_ENABLE=1 at runtime, so they're safe to
    apply unconditionally — when disabled, they're complete no-ops.

    Returns:
        True if patches were applied, False if already applied or failed.
    """
    global _FP4_CACHE_PATCHED

    if _FP4_CACHE_PATCHED:
        logger.debug("[TQ-FP4] Cache patch already applied, skipping")
        return False

    try:
        import torch
        from vllm.v1.kv_cache_interface import (
            AttentionSpec,
            FullAttentionSpec,
        )
        from vllm.v1.worker.gpu import attn_utils
    except ImportError as e:
        logger.debug(
            "[TQ-FP4] vLLM not available, skipping cache patch: %s", e
        )
        return False

    # ── Patch 1a: AttentionSpec.real_page_size_bytes ──
    # It's a @property on a frozen dataclass. We can override it on the
    # class because property descriptors live on the class, not instances.
    orig_base_prop = AttentionSpec.__dict__.get("real_page_size_bytes")
    if orig_base_prop is not None and isinstance(orig_base_prop, property):
        _originals["AttentionSpec.real_page_size_bytes"] = orig_base_prop
        AttentionSpec.real_page_size_bytes = _make_fp4_page_size_property(
            orig_base_prop
        )
        logger.info(
            "[TQ-FP4] Patched AttentionSpec.real_page_size_bytes"
        )
    else:
        logger.warning(
            "[TQ-FP4] AttentionSpec.real_page_size_bytes is not a property "
            "(%s) — skipping base class patch",
            type(orig_base_prop),
        )

    # ── Patch 1b: FullAttentionSpec.real_page_size_bytes ──
    # FullAttentionSpec overrides real_page_size_bytes with its own formula
    # that uses (head_size + head_size_v). We need to patch this too.
    orig_full_prop = FullAttentionSpec.__dict__.get("real_page_size_bytes")
    if orig_full_prop is not None and isinstance(orig_full_prop, property):
        _originals["FullAttentionSpec.real_page_size_bytes"] = orig_full_prop
        FullAttentionSpec.real_page_size_bytes = _make_fp4_page_size_property(
            orig_full_prop
        )
        logger.info(
            "[TQ-FP4] Patched FullAttentionSpec.real_page_size_bytes"
        )
    else:
        logger.warning(
            "[TQ-FP4] FullAttentionSpec.real_page_size_bytes not found "
            "or not a property — skipping"
        )

    # ── Patch 1c: Also patch any other AttentionSpec subclasses ──
    # MLAAttentionSpec, SlidingWindowSpec, ChunkedLocalAttentionSpec all
    # inherit or override real_page_size_bytes. For now, we only target
    # the common FullAttentionSpec path. Others can be added as needed.
    try:
        from vllm.v1.kv_cache_interface import MLAAttentionSpec
        orig_mla_prop = MLAAttentionSpec.__dict__.get("real_page_size_bytes")
        if orig_mla_prop is not None and isinstance(orig_mla_prop, property):
            _originals["MLAAttentionSpec.real_page_size_bytes"] = orig_mla_prop
            # MLA has different head_size semantics — for now, patch with
            # the same FP4 logic. The property getter checks head_size_v.
            MLAAttentionSpec.real_page_size_bytes = (
                _make_fp4_page_size_property(orig_mla_prop)
            )
            logger.info(
                "[TQ-FP4] Patched MLAAttentionSpec.real_page_size_bytes"
            )
    except ImportError:
        pass

    # ── Patch 2: _reshape_kv_cache ──
    orig_reshape = attn_utils._reshape_kv_cache
    _originals["_reshape_kv_cache"] = orig_reshape
    attn_utils._reshape_kv_cache = _make_patched_reshape_kv_cache(orig_reshape)
    logger.info("[TQ-FP4] Patched _reshape_kv_cache in attn_utils")

    # ── Patch 3: Backend get_kv_cache_shape ──
    # Patch the ROCm AITER FA backend's get_kv_cache_shape so it returns
    # FP4-compatible shapes. This is what _reshape_kv_cache calls via
    # attn_backend.get_kv_cache_shape().
    try:
        from vllm.v1.attention.backends.rocm_aiter_fa import (
            AiterFlashAttentionBackend,
        )
        _originals["AiterFlashAttentionBackend.get_kv_cache_shape"] = (
            AiterFlashAttentionBackend.get_kv_cache_shape
        )

        @staticmethod
        def _fp4_get_kv_cache_shape(
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
            cache_dtype_str: str = "auto",
        ) -> tuple[int, ...]:
            if is_fp4_enabled():
                return fp4_get_kv_cache_shape(
                    num_blocks, block_size, num_kv_heads, head_size,
                    cache_dtype_str,
                )
            # Original path
            if block_size % 16 != 0:
                raise ValueError("Block size must be a multiple of 16.")
            return (2, num_blocks, block_size, num_kv_heads, head_size)

        AiterFlashAttentionBackend.get_kv_cache_shape = _fp4_get_kv_cache_shape
        logger.info(
            "[TQ-FP4] Patched AiterFlashAttentionBackend.get_kv_cache_shape"
        )
    except ImportError:
        logger.debug(
            "[TQ-FP4] AiterFlashAttentionBackend not available, "
            "skipping get_kv_cache_shape patch"
        )

    # Also patch RocmAttnBackend (used when VLLM_ROCM_USE_AITER=0)
    try:
        from vllm.v1.attention.backends.rocm_attn import RocmAttnBackend
        _originals["RocmAttnBackend.get_kv_cache_shape"] = (
            RocmAttnBackend.get_kv_cache_shape
        )

        @staticmethod
        def _fp4_rocm_get_kv_cache_shape(
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
            cache_dtype_str: str = "auto",
        ) -> tuple[int, ...]:
            if is_fp4_enabled():
                return fp4_get_kv_cache_shape(
                    num_blocks, block_size, num_kv_heads, head_size,
                    cache_dtype_str,
                )
            if block_size % 16 != 0:
                raise ValueError("Block size must be a multiple of 16.")
            return (2, num_blocks, block_size, num_kv_heads, head_size)

        RocmAttnBackend.get_kv_cache_shape = _fp4_rocm_get_kv_cache_shape
        logger.info(
            "[TQ-FP4] Patched RocmAttnBackend.get_kv_cache_shape"
        )
    except ImportError:
        logger.debug(
            "[TQ-FP4] RocmAttnBackend not available, "
            "skipping get_kv_cache_shape patch"
        )

    # ── Patch 4: GPUModelRunner._reshape_kv_cache_tensors dtype override ──
    # The method does `.view(dtype).view(shape)` where dtype=bfloat16.
    # For FP4, we need `.view(torch.int8).view(shape)` since the raw
    # tensor is int8 and FP4 packed data is byte-level.
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        import torch

        _orig_reshape = GPUModelRunner._reshape_kv_cache_tensors
        _originals["GPUModelRunner._reshape_kv_cache_tensors"] = _orig_reshape

        def _fp4_reshape_kv_cache_tensors(self, kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes):
            """Patched reshape that uses int8 view for FP4 cache tensors."""
            if not is_fp4_enabled():
                return _orig_reshape(self, kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes)

            from vllm.v1.kv_cache_interface import AttentionSpec
            import itertools

            kv_caches = {}
            for group in self._kv_cache_spec_attn_group_iterator():
                kv_cache_spec = group.kv_cache_spec
                attn_backend = group.backend
                if group.kv_cache_group_id == len(kernel_block_sizes):
                    continue
                kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
                for layer_name in group.layer_names:
                    if layer_name in self.runner_only_attn_layers:
                        continue
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                    if isinstance(kv_cache_spec, AttentionSpec):
                        num_blocks_per_kv_block = (
                            kv_cache_spec.block_size // kernel_block_size
                        )
                        kernel_num_blocks = num_blocks * num_blocks_per_kv_block
                        kv_cache_shape = attn_backend.get_kv_cache_shape(
                            kernel_num_blocks,
                            kernel_block_size,
                            kv_cache_spec.num_kv_heads,
                            kv_cache_spec.head_size,
                            cache_dtype_str=self.cache_config.cache_dtype,
                        )
                        try:
                            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
                            assert len(kv_cache_stride_order) == len(kv_cache_shape)
                        except (AttributeError, NotImplementedError):
                            kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
                        kv_cache_shape = tuple(
                            kv_cache_shape[i] for i in kv_cache_stride_order
                        )
                        inv_order = [
                            kv_cache_stride_order.index(i)
                            for i in range(len(kv_cache_stride_order))
                        ]
                        # FP4: view as int8 (not bfloat16) — byte-level packing
                        kv_caches[layer_name] = (
                            raw_tensor
                            .view(torch.int8)  # NOT kv_cache_spec.dtype
                            .view(kv_cache_shape)
                            .permute(*inv_order)
                        )
                        logger.info(
                            "[TQ-FP4] Reshaped %s: shape=%s dtype=int8 blocks=%d",
                            layer_name, tuple(kv_caches[layer_name].shape), num_blocks,
                        )
                    else:
                        # Non-attention layers (Mamba etc.) — use original logic
                        # This shouldn't happen for our use case
                        logger.warning("[TQ-FP4] Non-attention layer %s, using original reshape", layer_name)
            return kv_caches

        GPUModelRunner._reshape_kv_cache_tensors = _fp4_reshape_kv_cache_tensors
        logger.info("[TQ-FP4] Patched GPUModelRunner._reshape_kv_cache_tensors")
    except ImportError as e:
        logger.debug("[TQ-FP4] GPUModelRunner not available: %s", e)

    _FP4_CACHE_PATCHED = True

    # Log summary
    if is_fp4_enabled():
        fp4_bytes = fp4_bytes_per_head()
        std_fp8_bytes = _DEFAULT_HEAD_DIM  # FP8: 1 byte per element
        std_bf16_bytes = _DEFAULT_HEAD_DIM * 2  # BF16: 2 bytes per element
        logger.info(
            "[TQ-FP4] ✓ FP4 cache patch applied and ACTIVE\n"
            "  FP4 bytes per head: %d (vs FP8: %d, BF16: %d)\n"
            "  Capacity vs FP8: %.2fx\n"
            "  Capacity vs BF16: %.2fx",
            fp4_bytes, std_fp8_bytes, std_bf16_bytes,
            std_fp8_bytes / fp4_bytes,
            std_bf16_bytes / fp4_bytes,
        )
    else:
        logger.info(
            "[TQ-FP4] Cache patch applied but INACTIVE "
            "(set TQ_FP4_ENABLE=1 to activate)"
        )

    return True


def revert_fp4_cache_patch() -> bool:
    """Revert all FP4 cache patches (for testing/cleanup).

    Returns:
        True if patches were reverted, False if nothing to revert.
    """
    global _FP4_CACHE_PATCHED

    if not _FP4_CACHE_PATCHED:
        return False

    try:
        from vllm.v1.kv_cache_interface import (
            AttentionSpec,
            FullAttentionSpec,
        )
        from vllm.v1.worker.gpu import attn_utils

        # Restore AttentionSpec property
        if "AttentionSpec.real_page_size_bytes" in _originals:
            AttentionSpec.real_page_size_bytes = _originals[
                "AttentionSpec.real_page_size_bytes"
            ]

        # Restore FullAttentionSpec property
        if "FullAttentionSpec.real_page_size_bytes" in _originals:
            FullAttentionSpec.real_page_size_bytes = _originals[
                "FullAttentionSpec.real_page_size_bytes"
            ]

        # Restore MLAAttentionSpec property
        try:
            from vllm.v1.kv_cache_interface import MLAAttentionSpec

            if "MLAAttentionSpec.real_page_size_bytes" in _originals:
                MLAAttentionSpec.real_page_size_bytes = _originals[
                    "MLAAttentionSpec.real_page_size_bytes"
                ]
        except ImportError:
            pass

        # Restore _reshape_kv_cache
        if "_reshape_kv_cache" in _originals:
            attn_utils._reshape_kv_cache = _originals["_reshape_kv_cache"]

        # Restore AiterFlashAttentionBackend.get_kv_cache_shape
        try:
            from vllm.v1.attention.backends.rocm_aiter_fa import (
                AiterFlashAttentionBackend,
            )

            if "AiterFlashAttentionBackend.get_kv_cache_shape" in _originals:
                AiterFlashAttentionBackend.get_kv_cache_shape = _originals[
                    "AiterFlashAttentionBackend.get_kv_cache_shape"
                ]
        except ImportError:
            pass

        _originals.clear()
        _FP4_CACHE_PATCHED = False
        logger.info("[TQ-FP4] Cache patches reverted")
        return True

    except ImportError:
        _FP4_CACHE_PATCHED = False
        _originals.clear()
        return False


# ============================================================================
# Helpers for the .pth autoloader
# ============================================================================

def autoload_fp4_cache_patch():
    """Entry point for .pth autoloader.

    Called from tq_autoload.pth in site-packages. This runs in ALL
    Python processes (main server + worker subprocesses).

    Only applies the patch if vLLM is importable — otherwise silently
    skips (so non-vLLM Python processes aren't affected).
    """
    if not is_fp4_enabled():
        return

    try:
        apply_fp4_cache_patch()
    except Exception as e:
        # Don't crash non-vLLM processes
        logger.debug("[TQ-FP4] Autoload skipped: %s", e)


# ============================================================================
# Diagnostic utilities
# ============================================================================

def get_fp4_cache_stats(
    num_kv_heads: int = 4,
    head_size: int = 128,
    block_size: int = 16,
    total_gpu_memory_gb: float = 288.0,
    kv_cache_fraction: float = 0.5,
) -> dict:
    """Compute expected FP4 cache capacity statistics.

    Args:
        num_kv_heads: Number of KV attention heads
        head_size: Attention head dimension
        block_size: KV cache block size
        total_gpu_memory_gb: Total GPU memory in GB
        kv_cache_fraction: Fraction of GPU memory for KV cache

    Returns:
        Dict with capacity statistics for FP4 vs FP8 vs BF16
    """
    fp4_bytes = fp4_bytes_per_head(head_size)
    fp8_bytes = head_size      # 1 byte per element
    bf16_bytes = head_size * 2  # 2 bytes per element

    # Page sizes (K + V combined)
    fp4_page = 2 * block_size * num_kv_heads * fp4_bytes
    fp8_page = 2 * block_size * num_kv_heads * fp8_bytes
    bf16_page = 2 * block_size * num_kv_heads * bf16_bytes

    kv_memory = total_gpu_memory_gb * kv_cache_fraction * 1024**3

    fp4_blocks = int(kv_memory / fp4_page)
    fp8_blocks = int(kv_memory / fp8_page)
    bf16_blocks = int(kv_memory / bf16_page)

    return {
        "fp4_bytes_per_head": fp4_bytes,
        "fp8_bytes_per_head": fp8_bytes,
        "bf16_bytes_per_head": bf16_bytes,
        "fp4_page_size": fp4_page,
        "fp8_page_size": fp8_page,
        "bf16_page_size": bf16_page,
        "fp4_num_blocks": fp4_blocks,
        "fp8_num_blocks": fp8_blocks,
        "bf16_num_blocks": bf16_blocks,
        "fp4_tokens": fp4_blocks * block_size,
        "fp8_tokens": fp8_blocks * block_size,
        "bf16_tokens": bf16_blocks * block_size,
        "capacity_vs_fp8": fp4_blocks / max(fp8_blocks, 1),
        "capacity_vs_bf16": fp4_blocks / max(bf16_blocks, 1),
    }


def verify_patch_integrity() -> dict[str, str]:
    """Verify that all patches are correctly applied.

    Returns:
        Dict mapping patch names to status strings.
    """
    results = {}

    # Check 1: AttentionSpec.real_page_size_bytes
    try:
        from vllm.v1.kv_cache_interface import AttentionSpec

        prop = AttentionSpec.__dict__.get("real_page_size_bytes")
        if prop and isinstance(prop, property):
            # Check if it's our patched version
            if "_fp4_real_page_size_bytes" in str(prop.fget):
                results["AttentionSpec.real_page_size_bytes"] = "PATCHED"
            else:
                results["AttentionSpec.real_page_size_bytes"] = "ORIGINAL"
        else:
            results["AttentionSpec.real_page_size_bytes"] = "MISSING"
    except ImportError:
        results["AttentionSpec.real_page_size_bytes"] = "UNAVAILABLE"

    # Check 2: FullAttentionSpec.real_page_size_bytes
    try:
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        prop = FullAttentionSpec.__dict__.get("real_page_size_bytes")
        if prop and isinstance(prop, property):
            if "_fp4_real_page_size_bytes" in str(prop.fget):
                results["FullAttentionSpec.real_page_size_bytes"] = "PATCHED"
            else:
                results["FullAttentionSpec.real_page_size_bytes"] = "ORIGINAL"
        else:
            results["FullAttentionSpec.real_page_size_bytes"] = "MISSING"
    except ImportError:
        results["FullAttentionSpec.real_page_size_bytes"] = "UNAVAILABLE"

    # Check 3: _reshape_kv_cache
    try:
        from vllm.v1.worker.gpu import attn_utils

        fn = attn_utils._reshape_kv_cache
        if "_patched_reshape_kv_cache" in str(fn):
            results["_reshape_kv_cache"] = "PATCHED"
        else:
            results["_reshape_kv_cache"] = "ORIGINAL"
    except ImportError:
        results["_reshape_kv_cache"] = "UNAVAILABLE"

    # Check 4: Backend get_kv_cache_shape
    try:
        from vllm.v1.attention.backends.rocm_aiter_fa import (
            AiterFlashAttentionBackend,
        )

        fn = AiterFlashAttentionBackend.get_kv_cache_shape
        if "_fp4_get_kv_cache_shape" in str(fn):
            results["AiterFlashAttentionBackend.get_kv_cache_shape"] = "PATCHED"
        else:
            results["AiterFlashAttentionBackend.get_kv_cache_shape"] = "ORIGINAL"
    except ImportError:
        results["AiterFlashAttentionBackend.get_kv_cache_shape"] = "UNAVAILABLE"

    results["TQ_FP4_ENABLE"] = "1" if is_fp4_enabled() else "0"
    results["_FP4_CACHE_PATCHED"] = str(_FP4_CACHE_PATCHED)

    return results


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TurboQuant FP4 KV Cache Patch for vLLM v0.18"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the FP4 cache patch",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify patch integrity",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print FP4 cache capacity statistics",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=4,
        help="Number of KV heads (default: 4)",
    )
    parser.add_argument(
        "--head-size",
        type=int,
        default=128,
        help="Head dimension (default: 128)",
    )
    parser.add_argument(
        "--gpu-memory-gb",
        type=float,
        default=288.0,
        help="GPU memory in GB (default: 288 for MI355X)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    if args.stats:
        stats = get_fp4_cache_stats(
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            total_gpu_memory_gb=args.gpu_memory_gb,
        )
        print("FP4 KV Cache Capacity Statistics")
        print("=" * 50)
        print(f"Head dimension: {args.head_size}")
        print(f"KV heads: {args.num_kv_heads}")
        print(f"GPU memory: {args.gpu_memory_gb} GB")
        print()
        print(f"Bytes per head per position:")
        print(f"  FP4:  {stats['fp4_bytes_per_head']}")
        print(f"  FP8:  {stats['fp8_bytes_per_head']}")
        print(f"  BF16: {stats['bf16_bytes_per_head']}")
        print()
        print(f"Total KV cache tokens (50% GPU memory):")
        print(f"  FP4:  {stats['fp4_tokens']:,}")
        print(f"  FP8:  {stats['fp8_tokens']:,}")
        print(f"  BF16: {stats['bf16_tokens']:,}")
        print()
        print(f"Capacity improvement:")
        print(f"  FP4 vs FP8:  {stats['capacity_vs_fp8']:.2f}x")
        print(f"  FP4 vs BF16: {stats['capacity_vs_bf16']:.2f}x")

    if args.apply:
        os.environ["TQ_FP4_ENABLE"] = "1"
        result = apply_fp4_cache_patch()
        if result:
            print("✓ FP4 cache patch applied successfully")
        else:
            print("✗ FP4 cache patch failed or already applied")

    if args.verify:
        status = verify_patch_integrity()
        print("Patch Integrity Check")
        print("=" * 50)
        for name, state in status.items():
            symbol = "✓" if state == "PATCHED" else "✗"
            print(f"  {symbol} {name}: {state}")
