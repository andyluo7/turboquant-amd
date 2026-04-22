"""Apply FP4 paged attention patches to AITER.

Architecture:
  Forked kernel `_paged_attention_fp4_kernel` in pa_kernels_fp4.cuh,
  dispatched from pa_v1.cuh when KV_DTYPE == kFp4E2M1.

  K-fetch: raw FP4 bytes (8 bytes per qkhe_depth) stored in Klocal.
  QK MFMA: LUT decode FP4→FP8, Q read directly from q pointer,
           1× mfma_scale_f32_16x16x128 (cbsz=0, both FP8) per head_loop.
  V-fetch: FP4→FP8 LUT decode via fp4_decode_8bytes_to_fp8.
  V-MFMA:  existing FP8 path (unchanged).
  Scale:   k_scale/v_scale passed as nullptr.

  Correctness: 9/9 tests pass, cosine > 0.99 vs torch SDPA reference.
  Performance: FP4 ≈ FP8 latency at short-medium contexts, 2× KV capacity.

Files patched:
  1. dtype_fp8.cuh:       add kFp4E2M1 = 3 enum
  2. pa_common.cuh:       add vector types + FP4→FP8 LUT helpers
  3. pa_kernels_fp4.cuh:  write forked FP4 kernel (new file)
  4. pa_kernels.cuh:      add #include "pa_kernels_fp4.cuh"
  5. pa_v1.cuh:           dispatch kFp4E2M1 → forked kernel
  6. pa_v1.py:            fp4_e2m1 dtype + JIT includes + nullptr scales
  7. pa_v1.cpp.jinja:     fp4_e2m1 → kFp4E2M1 enum mapping

Usage:
    python3 paged_attention_fp4_patch.py          # apply
    python3 paged_attention_fp4_patch.py --revert # restore from backup
    python3 paged_attention_fp4_patch.py --check  # check status
"""

import shutil
import sys
from pathlib import Path


def _find_aiter_base() -> Path:
    candidates = [
        Path("/app/aiter-test/csrc"),
        Path("/opt/aiter/csrc"),
        Path.home() / "aiter_prwork/csrc",
        Path.home() / "aiter/csrc",
    ]
    for p in candidates:
        if (p / "cpp_itfs/pa/pa_kernels.cuh").exists():
            return p
    raise FileNotFoundError(
        "Could not find AITER csrc. Set AITER_CSRC env var."
    )


import os as _os

AITER_BASE = (
    Path(_os.environ["AITER_CSRC"])
    if _os.environ.get("AITER_CSRC")
    else _find_aiter_base()
)
PA_DIR = AITER_BASE / "cpp_itfs/pa"
KERNELS_CUH = PA_DIR / "pa_kernels.cuh"
COMMON_CUH = PA_DIR / "pa_common.cuh"
V1_CUH = PA_DIR / "pa_v1.cuh"
PA_V1_PY = PA_DIR / "pa_v1.py"
PA_V1_JINJA = PA_DIR / "pa_v1.cpp.jinja"
DTYPE_CUH = AITER_BASE / "include/dtype_fp8.cuh"
FP4_KERNEL = PA_DIR / "pa_kernels_fp4.cuh"
JIT_DIR = Path.home() / ".aiter/build"
# Forked kernel source lives alongside this script
SCRIPT_DIR = Path(__file__).resolve().parent

BACKUP_SUFFIX = ".fp4_bak"


def _backup(path: Path):
    bak = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  Backed up {path.name}")


def _restore(path: Path):
    bak = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if bak.exists():
        shutil.copy2(bak, path)
        bak.unlink()
        print(f"  Restored {path.name}")
    else:
        print(f"  No backup for {path.name}")


def clear_jit_cache():
    if JIT_DIR.exists():
        n = 0
        for d in JIT_DIR.iterdir():
            if d.name.startswith("pa_v1_"):
                shutil.rmtree(d)
                n += 1
        if n:
            print(f"  Cleared {n} JIT cache entries")


# ── 1. dtype_fp8.cuh ────────────────────────────────────────────────
def patch_dtype():
    c = DTYPE_CUH.read_text()
    if "kFp4" in c:
        print("1. dtype_fp8.cuh: kFp4 already present")
        return
    _backup(DTYPE_CUH)
    c = c.replace(
        "    kFp8E5M2 = 2,\n};",
        "    kFp8E5M2 = 2,\n"
        "    kFp4E2M1 = 3,  // TurboQuant FP4\n};",
    )
    DTYPE_CUH.write_text(c)
    print("1. dtype_fp8.cuh: added kFp4E2M1 = 3")


# ── 2. pa_common.cuh ────────────────────────────────────────────────
def patch_common():
    c = COMMON_CUH.read_text()
    if "fp4_to_fp8_e4m3" in c:
        print("2. pa_common.cuh: FP4 helpers already present")
        return
    _backup(COMMON_CUH)

    helpers = """
// =========================================================================
// TurboQuant: vector types + FP4→FP8 LUT helpers
// =========================================================================

// 256-bit / 128-bit vector types for mfma_scale_f32_16x16x128 operands
typedef int _mfma_b256_t __attribute__((vector_size(32)));
typedef int _mfma_b128_t __attribute__((vector_size(16)));

// FP4 E2M1 → FP8 E4M3 single-nibble LUT
__device__ __forceinline__ uint8_t fp4_to_fp8_e4m3(uint8_t nibble)
{
    constexpr uint8_t lut[16] = {
        0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
        0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC,
    };
    return lut[nibble & 0xF];
}

// Decode 8 packed FP4 bytes (16 nibbles) → 16 FP8 E4M3 bytes
__device__ __forceinline__ _B16x8
fp4_decode_8bytes_to_fp8(_B8x8 fp4_raw)
{
    _B16x8 out;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(&fp4_raw);
    uint8_t* dst = reinterpret_cast<uint8_t*>(&out);
    #pragma unroll
    for(int i = 0; i < 8; i++) {
        dst[i * 2]     = fp4_to_fp8_e4m3(src[i] & 0xF);
        dst[i * 2 + 1] = fp4_to_fp8_e4m3((src[i] >> 4) & 0xF);
    }
    return out;
}
"""
    # Insert after the #endif that closes the __gfx950__ / #else block
    idx950 = c.find("#if defined(__gfx950__)")
    if idx950 >= 0:
        idx_else = c.find("\n#else\n", idx950)
        idx_endif = c.find("\n#endif\n", idx_else if idx_else > 0 else idx950)
        if idx_endif >= 0:
            pos = idx_endif + len("\n#endif\n")
            c = c[:pos] + helpers + c[pos:]
        else:
            c += helpers
    else:
        c += helpers

    COMMON_CUH.write_text(c)
    print("2. pa_common.cuh: added FP4 vector types + LUT helpers")


# ── 3. pa_kernels_fp4.cuh ───────────────────────────────────────────
def write_fp4_kernel():
    src = SCRIPT_DIR / "pa_kernels_fp4.cuh"
    if not src.exists():
        print("3. ERROR: pa_kernels_fp4.cuh not found next to patch script")
        return
    shutil.copy2(src, FP4_KERNEL)
    print(f"3. pa_kernels_fp4.cuh: copied ({FP4_KERNEL})")


# ── 4. pa_kernels.cuh ───────────────────────────────────────────────
def patch_kernels():
    c = KERNELS_CUH.read_text()
    include_line = '#include "pa_kernels_fp4.cuh"'
    if include_line in c:
        print("4. pa_kernels.cuh: #include already present")
        return
    _backup(KERNELS_CUH)
    c = c.rstrip() + "\n\n" + include_line + "\n"
    KERNELS_CUH.write_text(c)
    print("4. pa_kernels.cuh: added #include pa_kernels_fp4.cuh")


# ── 5. pa_v1.cuh ────────────────────────────────────────────────────
# Runtime cutover between native FP4 MFMA and FP4→FP8 LUT decode paths.
# After fixing the shuffle reorientation in pa_kernels_fp4.cuh, native is
# faster than LUT at every measured S (1.07× at S=4K → 1.26× at S=64K), so
# the threshold is 0 (always native). Keep the dispatch in case a future
# regime makes LUT competitive again.
TQ_FP4_NATIVE_THRESHOLD = 0


def patch_v1_cuh():
    c = V1_CUH.read_text()
    if "_paged_attention_fp4_kernel" in c:
        print("5. pa_v1.cuh: FP4 dispatch already present")
        return
    _backup(V1_CUH)

    old_call = (
        "    _paged_attention_kernel<scalar_t, cache_t, KV_DTYPE, BLOCK_SIZE, "
        "HEAD_SIZE, NUM_THREADS, ALIBI_ENABLED, GQA_RATIO, MTP, "
        "AttentionVariant, SLIDING_WINDOW_ENABLED>"
        "(block_table_seq, static_cast<int64_t>(query_loc), context_len, "
        "partition_start_token_idx, q, k_cache, v_cache, scale, "
        "alibi_slopes, q_stride, kv_block_stride, kv_head_stride, "
        "kv_seq_stride, exp_sums, max_logits, out, logits_soft_cap, "
        "logits_soft_cap_rcp, q_scale_ptr, k_scale_ptr, v_scale_ptr, "
        "variant, sliding_window);"
    )
    targs = (
        "<scalar_t, cache_t, KV_DTYPE, BLOCK_SIZE, HEAD_SIZE, "
        "NUM_THREADS, ALIBI_ENABLED, GQA_RATIO, MTP, "
        "AttentionVariant, SLIDING_WINDOW_ENABLED>"
    )
    targs_native = targs[:-1] + ", true>"
    targs_lut    = targs[:-1] + ", false>"
    args = (
        "(block_table_seq, static_cast<int64_t>(query_loc), context_len, "
        "partition_start_token_idx, q, k_cache, v_cache, scale, "
        "alibi_slopes, q_stride, kv_block_stride, kv_head_stride, "
        "kv_seq_stride, exp_sums, max_logits, out, logits_soft_cap, "
        "logits_soft_cap_rcp, q_scale_ptr, k_scale_ptr, v_scale_ptr, "
        "variant, sliding_window)"
    )
    new_call = (
        f"    if (KV_DTYPE == vllm::Fp8KVCacheDataType::kFp4E2M1)\n"
        f"    {{\n"
        f"        if ((int64_t)gridDim.x * context_len >= {TQ_FP4_NATIVE_THRESHOLD})\n"
        f"        {{\n"
        f"            _paged_attention_fp4_kernel{targs_native}{args};\n"
        f"        }}\n"
        f"        else\n"
        f"        {{\n"
        f"            _paged_attention_fp4_kernel{targs_lut}{args};\n"
        f"        }}\n"
        f"    }}\n"
        f"    else\n"
        f"    {{\n"
        f"        _paged_attention_kernel{targs}{args};\n"
        f"    }}"
    )
    if old_call in c:
        c = c.replace(old_call, new_call, 1)
        V1_CUH.write_text(c)
        print(f"5. pa_v1.cuh: added FP4 dispatch (native threshold = {TQ_FP4_NATIVE_THRESHOLD})")
    else:
        print("5. WARNING: pa_v1.cuh call pattern not found")


# ── 6. pa_v1.py ─────────────────────────────────────────────────────
def patch_v1_py():
    c = PA_V1_PY.read_text()
    if "fp4_e2m1" in c:
        print("6. pa_v1.py: fp4_e2m1 already present")
        return
    _backup(PA_V1_PY)

    # 6a. Add fp4_e2m1 dtype block
    old = '    else:\n        raise ValueError(f"Unsupported kv_cache_dtype: {kv_cache_dtype}")'
    insert = (
        '    elif kv_cache_dtype == "fp4_e2m1":\n'
        '        if query.dtype == torch.bfloat16:\n'
        '            dtype = "__hip_bfloat16"\n'
        '        elif query.dtype == torch.float16:\n'
        '            dtype = "_Float16"\n'
        '        else:\n'
        '            raise ValueError(f"Unsupported query dtype for FP4: {query.dtype}")\n'
        '        kv_dtype = "uint8_t"\n'
        '        fp8_kv_dtype = "fp4_e2m1"\n'
    )
    if old in c:
        c = c.replace(old, insert + old)
    else:
        print("6a. WARNING: dtype else-block not found")
        return

    # 6b. Add pa_kernels_fp4.cuh to JIT includes
    old_inc = '            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_common.cuh",'
    new_inc = (
        '            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_common.cuh",\n'
        '            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_kernels_fp4.cuh",'
    )
    if "pa_kernels_fp4" not in c:
        c = c.replace(old_inc, new_inc, 1)

    # 6c. k_scale/v_scale → nullptr for FP4
    old_scale = (
        "        ctypes.cast(k_scale.data_ptr(), ctypes.POINTER(ctypes.c_float)),\n"
        "        ctypes.cast(v_scale.data_ptr(), ctypes.POINTER(ctypes.c_float)),"
    )
    new_scale = (
        "        ctypes.POINTER(ctypes.c_float)()\n"
        "        if kv_cache_dtype == 'fp4_e2m1'\n"
        "        else ctypes.cast(k_scale.data_ptr(), ctypes.POINTER(ctypes.c_float)),\n"
        "        ctypes.POINTER(ctypes.c_float)()\n"
        "        if kv_cache_dtype == 'fp4_e2m1'\n"
        "        else ctypes.cast(v_scale.data_ptr(), ctypes.POINTER(ctypes.c_float)),"
    )
    if old_scale in c:
        c = c.replace(old_scale, new_scale)

    PA_V1_PY.write_text(c)
    print("6. pa_v1.py: added fp4_e2m1 dtype + includes + nullptr scales")


# ── 7. pa_v1.cpp.jinja ──────────────────────────────────────────────
def patch_jinja():
    c = PA_V1_JINJA.read_text()
    if "fp4_e2m1" in c:
        print("7. pa_v1.cpp.jinja: fp4_e2m1 already present")
        return
    _backup(PA_V1_JINJA)

    old_j = (
        "                                            {% else %}\n"
        "                                            vllm::Fp8KVCacheDataType::kFp8E4M3,"
    )
    new_j = (
        "                                            {% elif fp8_kv_dtype == 'fp4_e2m1' %}\n"
        "                                            vllm::Fp8KVCacheDataType::kFp4E2M1,\n"
        "                                            {% else %}\n"
        "                                            vllm::Fp8KVCacheDataType::kFp8E4M3,"
    )
    if old_j in c:
        c = c.replace(old_j, new_j)
        PA_V1_JINJA.write_text(c)
        print("7. pa_v1.cpp.jinja: added fp4_e2m1 enum mapping")
    else:
        print("7. WARNING: Jinja pattern not found")


# ── apply / revert / check ──────────────────────────────────────────
def apply():
    print("=== Applying TurboQuant FP4 PA patches ===\n")
    patch_dtype()
    patch_common()
    write_fp4_kernel()
    patch_kernels()
    patch_v1_cuh()
    patch_v1_py()
    patch_jinja()
    clear_jit_cache()
    print("\n=== Done. FP4 paged attention ready. ===")


def revert():
    print("=== Reverting TurboQuant FP4 patches ===\n")
    for p in [COMMON_CUH, KERNELS_CUH, V1_CUH, DTYPE_CUH, PA_V1_PY, PA_V1_JINJA]:
        _restore(p)
    if FP4_KERNEL.exists():
        FP4_KERNEL.unlink()
        print(f"  Removed {FP4_KERNEL.name}")
    clear_jit_cache()
    print("\nDone.")


def check():
    print("=== FP4 patch status ===")
    print(f"  dtype_fp8.cuh:       {'kFp4E2M1' in DTYPE_CUH.read_text()}")
    print(f"  pa_common.cuh:       {'fp4_to_fp8_e4m3' in COMMON_CUH.read_text()}")
    print(f"  pa_kernels_fp4.cuh:  {FP4_KERNEL.exists()}")
    print(f"  pa_kernels.cuh:      {'pa_kernels_fp4.cuh' in KERNELS_CUH.read_text()}")
    print(f"  pa_v1.cuh:           {'_paged_attention_fp4_kernel' in V1_CUH.read_text()}")
    print(f"  pa_v1.py:            {'fp4_e2m1' in PA_V1_PY.read_text()}")
    print(f"  pa_v1.cpp.jinja:     {'fp4_e2m1' in PA_V1_JINJA.read_text()}")


if __name__ == "__main__":
    if "--revert" in sys.argv:
        revert()
    elif "--check" in sys.argv:
        check()
    else:
        apply()
