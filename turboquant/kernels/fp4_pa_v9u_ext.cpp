#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Forward declaration of the HIP kernel launcher
extern "C" void launch_fp4_pa_v9u(
    const void* Q, const void* KV_cache,
    const void* BT, const void* CL, void* O,
    void* ws_m, void* ws_l, void* ws_acc,
    int ns, int nh, int nkv, int hd, int bs, int mnb,
    float sc, int num_splits, hipStream_t st
);

void fp4_paged_attention_v9u(
    torch::Tensor query,       // [batch, num_heads, head_dim] fp16
    torch::Tensor kv_cache,    // [num_blocks, 2, block_size, num_kv_heads, 68] int8
    torch::Tensor block_table, // [batch, max_blocks] int32
    torch::Tensor context_lens,// [batch] int32
    torch::Tensor output,      // [batch, num_heads, head_dim] fp16
    torch::Tensor ws_m,        // workspace
    torch::Tensor ws_l,
    torch::Tensor ws_acc,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_num_blocks,
    float scale,
    int num_splits
) {
    auto stream = at::cuda::getCurrentHIPStream().stream();
    
    launch_fp4_pa_v9u(
        query.data_ptr(),
        kv_cache.data_ptr(),
        block_table.data_ptr(),
        context_lens.data_ptr(),
        output.data_ptr(),
        ws_m.data_ptr(),
        ws_l.data_ptr(),
        ws_acc.data_ptr(),
        query.size(0),  // batch
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        max_num_blocks,
        scale,
        num_splits,
        stream
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp4_paged_attention_v9u", &fp4_paged_attention_v9u,
          "FP4 Paged Attention v9u (unified cache)");
}
