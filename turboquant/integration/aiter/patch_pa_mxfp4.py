"""Patch pa_kernels.cuh to support MXFP4 KV cache.

Modifies the K and V fetch loops to unpack FP4 nibbles + apply E8M0 scales.
Uses FP16 MFMA path (same as kAuto) for QK and PV computation.
"""
import shutil
from pathlib import Path

AITER_META = Path("/usr/local/lib/python3.12/dist-packages/aiter_meta")


def patch_kernels_cuh():
    """Patch pa_kernels.cuh with MXFP4 K/V dequant."""
    p = AITER_META / "csrc/cpp_itfs/pa/pa_kernels.cuh"
    bak = p.with_suffix(".cuh.mxfp4_bak")
    
    content = p.read_text()
    
    if "kMxfp4" in content:
        print("pa_kernels.cuh already patched")
        return True
    
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f"Backed up {p.name}")
    
    # PATCH 1: K scale — don't apply k_scale for MXFP4
    old1 = "    if constexpr(KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto)\n    {\n        // multiply by k_scale if fp8 kv cache\n        scale2 *= *k_scale_ptr;\n    }"
    new1 = "    if constexpr(KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto && KV_DTYPE != vllm::Fp8KVCacheDataType::kMxfp4)\n    {\n        // multiply by k_scale if fp8 kv cache (not for MXFP4 — scale applied during unpack)\n        scale2 *= *k_scale_ptr;\n    }"
    
    if old1 in content:
        content = content.replace(old1, new1)
        print("  Patched k_scale guard")
    else:
        print("  WARNING: k_scale pattern not found")
    
    # PATCH 2: QK MFMA — use FP16 path for MXFP4
    old2 = "                        if constexpr(KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto)"
    new2 = "                        if constexpr(KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto || KV_DTYPE == vllm::Fp8KVCacheDataType::kMxfp4)"
    
    # This pattern appears multiple times (QK and PV sections)
    count = content.count(old2)
    content = content.replace(old2, new2)
    print(f"  Patched {count} MFMA path guards (FP16 for MXFP4)")
    
    # PATCH 3: K fetch — add MXFP4 unpack before the load
    # Find the K fetch loop and add MXFP4 branch
    k_fetch_old = """                const cache_t* k_fetch_ptr    = k_ptr3 + offset1 * KX + offset2;
                const _B16x8* k_fetch_ptr_16B = reinterpret_cast<const _B16x8*>(k_fetch_ptr);
                if constexpr(NT_KV_LOAD)
                {
                    Klocal[head_loop][token_depth][qkhe_depth] =
                        load_ntmprl_16Byte(k_fetch_ptr_16B);
                }
                else
                {
                    Klocal[head_loop][token_depth][qkhe_depth] = *k_fetch_ptr_16B;
                }"""
    
    k_fetch_new = """                if constexpr(KV_DTYPE == vllm::Fp8KVCacheDataType::kMxfp4)
                {
                    // MXFP4: unpack nibble-packed FP4 + apply E8M0 block scale
                    const int nibble_byte = head_elem / 2;
                    const uint8_t* k_fp4 = reinterpret_cast<const uint8_t*>(k_ptr3) + nibble_byte;
                    // Load 8 packed bytes = 16 FP4 values
                    uint64_t packed8 = *reinterpret_cast<const uint64_t*>(k_fp4);
                    // E8M0 scale at offset 64 (after 64 packed bytes)
                    const uint8_t* sc = reinterpret_cast<const uint8_t*>(k_ptr3) + 64 + head_elem / 32;
                    float bscale = exp2f((float)(*sc) - 127.0f);
                    // FP4 E2M1 LUT
                    constexpr float lut[16] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,
                                               -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};
                    _B16x8 k_unpacked;
                    _Float16* kf = reinterpret_cast<_Float16*>(&k_unpacked);
                    for(int ii = 0; ii < 8; ii++)
                    {
                        uint8_t b = (packed8 >> (ii*8)) & 0xFF;
                        kf[2*ii]   = (_Float16)(lut[b & 0xF] * bscale);
                        kf[2*ii+1] = (_Float16)(lut[(b>>4) & 0xF] * bscale);
                    }
                    Klocal[head_loop][token_depth][qkhe_depth] = k_unpacked;
                }
                else
                {
                    const cache_t* k_fetch_ptr    = k_ptr3 + offset1 * KX + offset2;
                    const _B16x8* k_fetch_ptr_16B = reinterpret_cast<const _B16x8*>(k_fetch_ptr);
                    if constexpr(NT_KV_LOAD)
                    {
                        Klocal[head_loop][token_depth][qkhe_depth] =
                            load_ntmprl_16Byte(k_fetch_ptr_16B);
                    }
                    else
                    {
                        Klocal[head_loop][token_depth][qkhe_depth] = *k_fetch_ptr_16B;
                    }
                }"""
    
    if k_fetch_old in content:
        content = content.replace(k_fetch_old, k_fetch_new)
        print("  Patched K fetch with MXFP4 unpack")
    else:
        print("  WARNING: K fetch pattern not found")
    
    # PATCH 4: V scale — same as K scale
    old4 = "                // apply post Softmax V mfma v_scale\n                if constexpr(KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto)"
    new4 = "                // apply post Softmax V mfma v_scale\n                if constexpr(KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto && KV_DTYPE != vllm::Fp8KVCacheDataType::kMxfp4)"
    
    if old4 in content:
        content = content.replace(old4, new4)
        print("  Patched v_scale guard")
    else:
        print("  WARNING: v_scale pattern not found")
    
    # PATCH 5: V fetch — add MXFP4 unpack (first kernel variant, line ~340)
    v_fetch_old = """                const cache_t* v_fetch_ptr = v_ptr2 + (vblock_number * kv_block_stride);

                Vlocal[vtoken_depth][vhe_depth][vblock_depth] =
                    *reinterpret_cast<const _B16x8*>(v_fetch_ptr);"""
    
    v_fetch_new = """                const cache_t* v_fetch_ptr = v_ptr2 + (vblock_number * kv_block_stride);

                if constexpr(KV_DTYPE == vllm::Fp8KVCacheDataType::kMxfp4)
                {
                    // MXFP4: unpack nibble-packed FP4 + apply E8M0 block scale for V
                    const int v_nibble_byte = vhead_elem / 2;
                    const uint8_t* v_fp4 = reinterpret_cast<const uint8_t*>(v_fetch_ptr) + v_nibble_byte - vhead_elem;
                    // Actually: v_fetch_ptr already points to head start + vhead_elem offset
                    // We need to read from the base position, offset by nibble byte index
                    const uint8_t* v_base = reinterpret_cast<const uint8_t*>(v_ptr2 + (vblock_number * kv_block_stride) - vhead_elem);
                    const uint8_t* v_fp4_data = v_base + vhead_elem / 2;
                    uint64_t v_packed8 = *reinterpret_cast<const uint64_t*>(v_fp4_data);
                    const uint8_t* v_sc = v_base + 64 + vhead_elem / 32;
                    float v_bscale = exp2f((float)(*v_sc) - 127.0f);
                    constexpr float vlut[16] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,
                                               -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};
                    _B16x8 v_unpacked;
                    _Float16* vf = reinterpret_cast<_Float16*>(&v_unpacked);
                    for(int ii = 0; ii < 8; ii++)
                    {
                        uint8_t b = (v_packed8 >> (ii*8)) & 0xFF;
                        vf[2*ii]   = (_Float16)(vlut[b & 0xF] * v_bscale);
                        vf[2*ii+1] = (_Float16)(vlut[(b>>4) & 0xF] * v_bscale);
                    }
                    Vlocal[vtoken_depth][vhe_depth][vblock_depth] = v_unpacked;
                }
                else
                {
                    Vlocal[vtoken_depth][vhe_depth][vblock_depth] =
                        *reinterpret_cast<const _B16x8*>(v_fetch_ptr);
                }"""
    
    if v_fetch_old in content:
        content = content.replace(v_fetch_old, v_fetch_new, 1)  # Only first occurrence
        print("  Patched V fetch with MXFP4 unpack")
    else:
        print("  WARNING: V fetch pattern not found")

    # Write
    p.write_text(content)
    print(f"  Written {p}")
    
    # Clear JIT cache
    jit_dir = Path("/root/.aiter/build")
    if jit_dir.exists():
        for d in jit_dir.iterdir():
            if d.name.startswith("pa_v1_"):
                shutil.rmtree(d)
                print(f"  Cleared JIT: {d.name}")
    
    return True


def revert():
    p = AITER_META / "csrc/cpp_itfs/pa/pa_kernels.cuh"
    bak = p.with_suffix(".cuh.mxfp4_bak")
    if bak.exists():
        shutil.copy2(bak, p)
        bak.unlink()
        print("Reverted pa_kernels.cuh")


if __name__ == "__main__":
    import sys
    if "--revert" in sys.argv:
        revert()
    else:
        patch_kernels_cuh()
