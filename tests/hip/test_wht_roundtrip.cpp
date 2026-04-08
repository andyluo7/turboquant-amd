// Standalone test: TurboQuant WHT kernel correctness on HIP/ROCm
// Tests: forward WHT → inverse WHT should recover original data

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

// ---- WHT sign arrays (must match turbo-quant.cuh exactly) ----
__constant__ float TURBO_WHT_SIGNS1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

__constant__ float TURBO_WHT_SIGNS2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f
};

// ---- WHT kernel (identical to turbo-wht.cu) ----
template <int direction, int group_size>
__global__ void k_turbo_wht_f32(const float * __restrict__ src,
                                 float * __restrict__ dst,
                                 const float * __restrict__ scale_inv,
                                 int64_t n_groups,
                                 int64_t head_dim,
                                 int64_t groups_per_head) {
    const int64_t g = blockIdx.x;
    if (g >= n_groups) return;

    const int t = threadIdx.x;
    const int64_t head_idx     = g / groups_per_head;
    const int64_t grp_in_head  = g % groups_per_head;
    const int64_t base         = head_idx * head_dim + grp_in_head * group_size;

    __shared__ float x[group_size];

    x[t] = src[base + t];
    __syncthreads();

    if (direction == 0 && scale_inv != nullptr) {
        x[t] *= scale_inv[t % group_size];
        __syncthreads();
    }

    x[t] *= (direction == 0) ? TURBO_WHT_SIGNS1[t] : TURBO_WHT_SIGNS2[t];
    __syncthreads();

#define WHT_STAGE(h) \
    if (t % (2*(h)) < (h)) { float a = x[t], b = x[t+(h)]; x[t] = a+b; x[t+(h)] = a-b; } \
    __syncthreads();

    WHT_STAGE(1)
    WHT_STAGE(2)
    WHT_STAGE(4)
    WHT_STAGE(8)
    WHT_STAGE(16)
    WHT_STAGE(32)
    WHT_STAGE(64)
#undef WHT_STAGE

    constexpr float inv_sqrt = 0.08838834764831845f; // 1/sqrt(128)
    float result = x[t] * inv_sqrt *
        ((direction == 0) ? TURBO_WHT_SIGNS2[t] : TURBO_WHT_SIGNS1[t]);

    if (direction == 1 && scale_inv != nullptr) {
        result *= scale_inv[t % group_size];
    }

    dst[base + t] = result;
}

int main() {
    const int N = 128;  // head_dim = group_size
    const int n_heads = 4;
    const int total = N * n_heads;

    // Host data
    float *h_input  = (float*)malloc(total * sizeof(float));
    float *h_fwd    = (float*)malloc(total * sizeof(float));
    float *h_inv    = (float*)malloc(total * sizeof(float));

    // Initialize with known values
    srand(42);
    for (int i = 0; i < total; i++) {
        h_input[i] = (float)(rand() % 1000 - 500) / 500.0f;
    }

    // Device buffers
    float *d_input, *d_fwd, *d_inv;
    hipMalloc(&d_input, total * sizeof(float));
    hipMalloc(&d_fwd,   total * sizeof(float));
    hipMalloc(&d_inv,   total * sizeof(float));

    hipMemcpy(d_input, h_input, total * sizeof(float), hipMemcpyHostToDevice);

    // Forward WHT
    dim3 blocks(n_heads); // 1 group per head since head_dim = group_size
    dim3 threads(128);
    k_turbo_wht_f32<0, 128><<<blocks, threads>>>(d_input, d_fwd, nullptr, n_heads, N, 1);
    hipDeviceSynchronize();

    // Inverse WHT
    k_turbo_wht_f32<1, 128><<<blocks, threads>>>(d_fwd, d_inv, nullptr, n_heads, N, 1);
    hipDeviceSynchronize();

    // Copy back
    hipMemcpy(h_fwd, d_fwd, total * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_inv, d_inv, total * sizeof(float), hipMemcpyDeviceToHost);

    // Check roundtrip: input → forward → inverse should recover input
    float max_err = 0.0f;
    double sum_sq_err = 0.0;
    int n_zero_fwd = 0;
    for (int i = 0; i < total; i++) {
        float err = fabsf(h_input[i] - h_inv[i]);
        max_err = fmaxf(max_err, err);
        sum_sq_err += (double)err * err;
        if (h_fwd[i] == 0.0f) n_zero_fwd++;
    }
    double rmse = sqrt(sum_sq_err / total);

    printf("=== TurboQuant WHT Roundtrip Test (HIP/gfx942) ===\n");
    printf("Total elements: %d (%d heads x %d dim)\n", total, n_heads, N);
    printf("Forward WHT zeros: %d / %d\n", n_zero_fwd, total);
    printf("Roundtrip max error: %.6e\n", max_err);
    printf("Roundtrip RMSE:      %.6e\n", rmse);
    printf("Result: %s\n", (max_err < 1e-4f) ? "PASS ✅" : "FAIL ❌");

    // Print first 8 values of each stage for one head
    printf("\nHead 0 first 8 values:\n");
    printf("Input:   ");
    for (int i = 0; i < 8; i++) printf("%.4f ", h_input[i]);
    printf("\nForward: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", h_fwd[i]);
    printf("\nInverse: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", h_inv[i]);
    printf("\n");

    // Check that forward WHT actually changed values (not identity)
    int n_changed = 0;
    for (int i = 0; i < total; i++) {
        if (fabsf(h_input[i] - h_fwd[i]) > 1e-6f) n_changed++;
    }
    printf("Forward changed %d / %d values (should be ~all)\n", n_changed, total);

    hipFree(d_input);
    hipFree(d_fwd);
    hipFree(d_inv);
    free(h_input);
    free(h_fwd);
    free(h_inv);

    return (max_err < 1e-4f) ? 0 : 1;
}
