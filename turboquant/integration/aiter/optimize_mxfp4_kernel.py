"""MXFP4 v4: Batch 4 FP4x2 conversions using uint32 + minimize register pressure."""
import shutil
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/cpp_itfs/pa/pa_kernels.cuh")
content = p.read_text()

# Instead of loop of 8 byte-by-byte conversions, process 4 bytes → 8 half values
# by accessing packed8 as uint8_t array and using a tighter loop

old_k = """                    // Optimized FP4 unpack: cast bytes directly, use __half2 scale
                    _B16x8 k_unpacked;
                    __half2* kh2 = reinterpret_cast<__half2*>(&k_unpacked);
                    __half2 bscale_h2 = __half2(__half(bscale), __half(bscale));
                    const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed8);
                    #pragma unroll
                    for(int ii = 0; ii < 8; ii++)
                    {
                        __half2_raw h2 = __hip_cvt_fp4x2_to_halfraw2(pb[ii], __HIP_E2M1);
                        kh2[ii] = __hmul2(*reinterpret_cast<__half2*>(&h2), bscale_h2);
                    }"""

# Try: move scale multiplication AFTER the unpack loop (batch multiply)
# Unpack 16 values first, then multiply all at once
new_k = """                    // Unpack all 16 FP4 values first, then batch-scale
                    _B16x8 k_unpacked;
                    __half2* kh2 = reinterpret_cast<__half2*>(&k_unpacked);
                    const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed8);
                    #pragma unroll
                    for(int ii = 0; ii < 8; ii++)
                    {
                        __half2_raw h2 = __hip_cvt_fp4x2_to_halfraw2(pb[ii], __HIP_E2M1);
                        kh2[ii] = *reinterpret_cast<__half2*>(&h2);
                    }
                    // Batch scale: 4x half2 multiplies
                    __half2 bscale_h2 = __half2(__half(bscale), __half(bscale));
                    #pragma unroll
                    for(int ii = 0; ii < 8; ii++)
                    {
                        kh2[ii] = __hmul2(kh2[ii], bscale_h2);
                    }"""

if old_k in content:
    content = content.replace(old_k, new_k)
    print("K: separated unpack from scale multiply (batch)")

# Same for V
old_v = """                    _B16x8 v_unpacked;
                    __half2* vh2_arr = reinterpret_cast<__half2*>(&v_unpacked);
                    __half2 v_bscale_h2 = __half2(__half(v_bscale), __half(v_bscale));
                    const uint8_t* vpb = reinterpret_cast<const uint8_t*>(&v_packed8);
                    #pragma unroll
                    for(int ii = 0; ii < 8; ii++)
                    {
                        __half2_raw vh2 = __hip_cvt_fp4x2_to_halfraw2(vpb[ii], __HIP_E2M1);
                        vh2_arr[ii] = __hmul2(*reinterpret_cast<__half2*>(&vh2), v_bscale_h2);
                    }"""

new_v = """                    _B16x8 v_unpacked;
                    __half2* vh2_arr = reinterpret_cast<__half2*>(&v_unpacked);
                    const uint8_t* vpb = reinterpret_cast<const uint8_t*>(&v_packed8);
                    #pragma unroll
                    for(int ii = 0; ii < 8; ii++)
                    {
                        __half2_raw vh2 = __hip_cvt_fp4x2_to_halfraw2(vpb[ii], __HIP_E2M1);
                        vh2_arr[ii] = *reinterpret_cast<__half2*>(&vh2);
                    }
                    __half2 v_bscale_h2 = __half2(__half(v_bscale), __half(v_bscale));
                    #pragma unroll
                    for(int ii = 0; ii < 8; ii++)
                    {
                        vh2_arr[ii] = __hmul2(vh2_arr[ii], v_bscale_h2);
                    }"""

if old_v in content:
    content = content.replace(old_v, new_v)
    print("V: separated unpack from scale multiply (batch)")

p.write_text(content)
print(f"Written {p}")

jit = Path("/root/.aiter/build")
if jit.exists():
    for d in jit.iterdir():
        if d.name.startswith("pa_v1_"):
            shutil.rmtree(d)
            print(f"Cleared {d.name}")
