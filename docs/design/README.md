# Design Documents

## MXFP4 PA Kernel (`MXFP4_PA_KERNEL.md`)
Native FP4 MFMA paged attention kernel design for MI355X (gfx950).
Uses `mfma_scale_f32_16x16x128_f8f6f4` for hardware FP4 decode.
Target: ≤ AITER FP8 latency with 2x KV cache capacity.

## AITER Fusion Plan (`AITER_FUSION_PLAN.md`)
Strategy for fusing TQ 3-bit dequant into AITER's paged attention inner loop.
Compares AOT HIP kernel vs Triton PA fork approaches.

## AITER turbo4 Design (`AITER_TURBO4_DESIGN.md`)
Detailed kernel structure analysis for integrating turbo4 (4-bit nibble PolarQuant)
into AITER's `paged_attention_ll4mi_QKV_mfma16_kernel`.
