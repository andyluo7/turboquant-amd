# HIP C++ Tests

## WHT Roundtrip Test (`test_wht_roundtrip.cpp`)

Standalone HIP test for Walsh-Hadamard Transform correctness on AMD GPUs.
Tests forward WHT → inverse WHT roundtrip error.

### Build & Run
```bash
hipcc -o test_wht test_wht_roundtrip.cpp --offload-arch=gfx942 -O2
./test_wht
```

### Expected Output
```
=== TurboQuant WHT Roundtrip Test (HIP/gfx942) ===
Total elements: 512 (4 heads x 128 dim)
Forward WHT zeros: 0 / 512
Roundtrip max error: 2.980232e-07
Roundtrip RMSE:      6.816018e-08
Result: PASS ✅
```

Tested on MI300X (gfx942, ROCm 7.0.2) and MI355X (gfx950, ROCm 7.0.1).
