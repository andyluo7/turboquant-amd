"""MXFP4 PA kernel: FP4→FP8 byte LUT in K/V fetch.

Strategy:
- Keep cache_t = uint8, HEAD_SIZE = 128 (same as FP8)
- K/V fetch: read 8 packed FP4 bytes → 16 FP8 bytes via LUT → store in _B16x8
- Use existing FP8 MFMA path (no tiling changes needed!)
- E8M0 scale applied via k_scale/v_scale (per-position)

The LUT is only 16 bytes — fits in registers.
Each lane processes 8 packed bytes → 16 FP8 output bytes.
Total cost: 16 LUT lookups + 16 byte writes per fetch = minimal overhead.
"""
from pathlib import Path
import shutil

AITER = Path("/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc")

def patch():
    p = AITER / "cpp_itfs/pa/pa_kernels.cuh"
    
    # Restore from clean backup
    bak = p.with_suffix(".cuh.bak")
    if bak.exists():
        shutil.copy2(bak, p)
        print("Restored clean kernel from backup")
    
    c = p.read_text()
    
    # 1. Add kMxfp4 enum if not present
    enum_file = AITER / "include/dtype_fp8.cuh"
    ec = enum_file.read_text()
    if "kMxfp4" not in ec:
        ec = ec.replace("kFp8E5M2  = 2,", "kFp8E5M2  = 2,\n    kMxfp4   = 3,")
        enum_file.write_text(ec)
        print("1. Added kMxfp4 enum")
    else:
        print("1. kMxfp4 enum exists")
    
    # 2. K fetch: add FP4→FP8 LUT conversion before existing fetch
    old_k = """                const cache_t* k_fetch_ptr    = k_ptr3 + offset1 * KX + offset2;
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
    
    new_k = """                if constexpr(KV_DTYPE == vllm::Fp8KVCacheDataType::kMxfp4)
                {
                    // FP4→FP8 via 16-entry byte LUT (no floating point!)
                    // FP4 E2M1 → FP8 E4M3FNUZ mapping
                    constexpr uint8_t FP4_TO_FP8[16] = {
                        0x00, 0x38, 0x40, 0x44, 0x48, 0x4C, 0x50, 0x54,
                        0x00, 0xB8, 0xC0, 0xC4, 0xC8, 0xCC, 0xD0, 0xD4
                    };
                    
                    // head_elem indexes bytes (cache_t=uint8)
                    // Read 8 packed FP4 bytes → 16 FP8 bytes
                    const uint8_t* k_fp4 = reinterpret_cast<const uint8_t*>(k_ptr3) + head_elem / 2;
                    
                    _B16x8 k_fp8;
                    uint8_t* k_out = reinterpret_cast<uint8_t*>(&k_fp8);
                    
                    #pragma unroll
                    for(int ii = 0; ii < 8; ii++)
                    {
                        uint8_t packed = k_fp4[ii];
                        k_out[2*ii]   = FP4_TO_FP8[packed & 0xF];
                        k_out[2*ii+1] = FP4_TO_FP8[(packed >> 4) & 0xF];
                    }
                    
                    Klocal[head_loop][token_depth][qkhe_depth] = k_fp8;
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
    
    if old_k in c:
        c = c.replace(old_k, new_k)
        print("2. K fetch: FP4→FP8 LUT conversion")
    else:
        print("2. K pattern not found!")
    
    # 3. V fetch: same LUT conversion
    old_v = """                    Vlocal[vtoken_depth][vhe_depth][vblock_depth] =
                        *reinterpret_cast<const _B16x8*>(v_fetch_ptr);"""
    
    new_v = """                    if constexpr(KV_DTYPE == vllm::Fp8KVCacheDataType::kMxfp4)
                    {
                        constexpr uint8_t FP4_TO_FP8_V[16] = {
                            0x00, 0x38, 0x40, 0x44, 0x48, 0x4C, 0x50, 0x54,
                            0x00, 0xB8, 0xC0, 0xC4, 0xC8, 0xCC, 0xD0, 0xD4
                        };
                        const uint8_t* v_base = reinterpret_cast<const uint8_t*>(
                            v_ptr2 + vblock_number * kv_block_stride - vhead_elem);
                        const uint8_t* v_fp4 = v_base + vhead_elem / 2;
                        
                        _B16x8 v_fp8;
                        uint8_t* v_out = reinterpret_cast<uint8_t*>(&v_fp8);
                        
                        #pragma unroll
                        for(int ii = 0; ii < 8; ii++)
                        {
                            uint8_t packed = v_fp4[ii];
                            v_out[2*ii]   = FP4_TO_FP8_V[packed & 0xF];
                            v_out[2*ii+1] = FP4_TO_FP8_V[(packed >> 4) & 0xF];
                        }
                        
                        Vlocal[vtoken_depth][vhe_depth][vblock_depth] = v_fp8;
                    }
                    else
                    {
                    Vlocal[vtoken_depth][vhe_depth][vblock_depth] =
                        *reinterpret_cast<const _B16x8*>(v_fetch_ptr);
                    }"""
    
    if old_v in c:
        c = c.replace(old_v, new_v)
        print("3. V fetch: FP4→FP8 LUT conversion")
    else:
        print("3. V pattern not found!")
    
    # 4. K scale: MXFP4 uses per-position E8M0 scale via k_scale_ptr
    # The existing FP8 path already multiplies by k_scale.
    # For MXFP4, we pass the E8M0-derived scale as k_scale.
    # No kernel change needed — Python caller sets k_scale appropriately.
    print("4. k_scale: handled by Python caller (no kernel change)")
    
    p.write_text(c)
    print(f"\nWritten {p}")
    
    # 5. Update pa_v1.py for MXFP4
    p2 = AITER / "cpp_itfs/pa/pa_v1.py"
    c2 = p2.read_text()
    if "mxfp4" not in c2:
        old = '        fp8_kv_dtype = "fp8_e4m3"'
        idx = c2.rindex(old)
        end = c2.index("\n\n", idx)
        insert = '''
    elif kv_cache_dtype == "mxfp4":
        if query.dtype == torch.bfloat16:
            dtype = "__hip_bfloat16"
            kv_dtype = "uint8_t"
        elif query.dtype == torch.float16:
            dtype = "_Float16"
            kv_dtype = "uint8_t"
        else:
            raise ValueError(f"Unsupported data type: {query.dtype}")
        fp8_kv_dtype = "mxfp4"
'''
        c2 = c2[:end] + insert + c2[end:]
        p2.write_text(c2)
        print("5. pa_v1.py: added mxfp4 (uint8 cache_t)")
    else:
        # Ensure it's uint8, not bf16
        c2 = c2.replace('kv_dtype = "__hip_bfloat16"  # bf16 cache_t for correct MFMA tiling', 'kv_dtype = "uint8_t"')
        c2 = c2.replace('kv_dtype = "_Float16"  # fp16 cache_t for correct MFMA tiling', 'kv_dtype = "uint8_t"')
        p2.write_text(c2)
        print("5. pa_v1.py: ensured uint8 cache_t for mxfp4")
    
    # 6. Jinja template
    p3 = AITER / "cpp_itfs/pa/pa_v1.cpp.jinja"
    c3 = p3.read_text()
    if "mxfp4" not in c3:
        old_j = '{% elif fp8_kv_dtype == "fp8_e5m2" %}'
        new_j = '{% elif fp8_kv_dtype == "mxfp4" %}\nvllm::Fp8KVCacheDataType::kMxfp4,\n{% elif fp8_kv_dtype == "fp8_e5m2" %}'
        c3 = c3.replace(old_j, new_j)
        p3.write_text(c3)
        print("6. Jinja: added mxfp4")
    else:
        print("6. Jinja: already has mxfp4")
    
    # Clear JIT
    jit = Path("/root/.aiter/build")
    if jit.exists():
        for d in jit.iterdir():
            if d.name.startswith("pa_v1_"):
                shutil.rmtree(d)
                print(f"Cleared {d.name}")

if __name__ == "__main__":
    import sys
    if "--revert" in sys.argv:
        p = AITER / "cpp_itfs/pa/pa_kernels.cuh"
        bak = p.with_suffix(".cuh.bak")
        if bak.exists():
            shutil.copy2(bak, p)
            print("Reverted")
    else:
        patch()
