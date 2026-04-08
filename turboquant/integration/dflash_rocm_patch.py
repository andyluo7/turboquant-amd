"""
DFLASH ROCm Port Patch
=====================
Enables DFLASH speculative decoding on AMD GPUs by adding Triton attention
backend support to the draft worker.

Changes:
1. dflash_worker.py: Add "triton" to supported_draft_backends
2. dflash_worker.py: Default to "triton" on ROCm instead of "flashinfer"

Apply: python3 dflash_rocm_patch.py <sglang_root>
Example: python3 dflash_rocm_patch.py /sgl-workspace/sglang/python/sglang
"""

import sys
import os

def patch_file(filepath, old_text, new_text):
    if not os.path.exists(filepath):
        print(f"SKIP: {filepath} not found")
        return False
    content = open(filepath).read()
    if old_text not in content:
        if new_text in content:
            print(f"ALREADY PATCHED: {filepath}")
            return True
        print(f"MISMATCH: expected text not found in {filepath}")
        return False
    content = content.replace(old_text, new_text)
    open(filepath, 'w').write(content)
    print(f"PATCHED: {filepath}")
    return True

def main():
    if len(sys.argv) < 2:
        sglang_root = "/sgl-workspace/sglang/python/sglang"
    else:
        sglang_root = sys.argv[1]

    # Patch 1: Add "triton" to supported draft backends and default to it on ROCm
    worker_file = os.path.join(sglang_root, "srt/speculative/dflash_worker.py")
    patch_file(
        worker_file,
        '        supported_draft_backends = ("flashinfer", "fa3", "fa4")',
        '        supported_draft_backends = ("flashinfer", "fa3", "fa4", "triton")'
    )

    # Patch 2: Default to "triton" on ROCm instead of "flashinfer"
    patch_file(
        worker_file,
        '''        if draft_backend is None:
            draft_backend, _ = draft_server_args.get_attention_backends()
        if draft_backend is None:
            draft_backend = "flashinfer"''',
        '''        if draft_backend is None:
            draft_backend, _ = draft_server_args.get_attention_backends()
        if draft_backend is None:
            # Use triton on ROCm (no FlashInfer), flashinfer on CUDA
            import importlib
            _hip = getattr(importlib.import_module("torch").version, "hip", None)
            draft_backend = "triton" if _hip else "flashinfer"'''
    )

    # Patch 3: Fall back to "triton" instead of "flashinfer" for unsupported backends on ROCm
    patch_file(
        worker_file,
        '''        elif draft_backend == "trtllm_mha":
            logger.warning(
                "DFLASH draft worker does not support 'trtllm_mha' because the "
                "draft path requires non-causal attention. Falling back to "
                "'flashinfer'."
            )
            draft_backend = "flashinfer"
        elif draft_backend not in supported_draft_backends:
            logger.warning(
                "DFLASH draft worker only supports attention_backend in %s for now, "
                "but got %r. Falling back to 'flashinfer'.",
                supported_draft_backends,
                draft_backend,
            )
            draft_backend = "flashinfer"''',
        '''        elif draft_backend == "trtllm_mha":
            import importlib as _il
            fallback = "triton" if getattr(_il.import_module("torch").version, "hip", None) else "flashinfer"
            logger.warning(
                "DFLASH draft worker does not support 'trtllm_mha' because the "
                "draft path requires non-causal attention. Falling back to "
                "'%s'.", fallback
            )
            draft_backend = fallback
        elif draft_backend not in supported_draft_backends:
            import importlib as _il2
            fallback = "triton" if getattr(_il2.import_module("torch").version, "hip", None) else "flashinfer"
            logger.warning(
                "DFLASH draft worker only supports attention_backend in %s for now, "
                "but got %r. Falling back to '%s'.",
                supported_draft_backends,
                draft_backend,
                fallback,
            )
            draft_backend = fallback'''
    )

    print("\nDone! DFLASH ROCm patch applied.")
    print("\nTo test:")
    print("  python3 -m sglang.launch_server \\")
    print("    --model-path Qwen/Qwen3-8B \\")
    print("    --speculative-algorithm DFLASH \\")
    print("    --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 \\")
    print("    --attention-backend triton \\")
    print("    --tp 1")


if __name__ == "__main__":
    main()
