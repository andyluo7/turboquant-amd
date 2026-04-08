#!/usr/bin/env python3
"""Compact KV cache allocator patch for TurboQuant in vLLM.

TurboQuant stores only 50 bytes per KV slot (48 packed 3-bit + 2 norm bytes)
but vLLM allocates 128-byte slots (head_size=128). This wastes 61% of KV cache
memory. This patch makes the allocator aware of the compressed size.

Changes:
  1. turboquant_attn.py: get_kv_cache_shape uses compressed slot size (50)
  2. attention.py: get_kv_cache_spec passes compressed head_size for TQ layers

The decode kernel (tq_paged_attention_v12.py) auto-adapts because it uses
strides from kv_cache.stride(0..3), not hardcoded offsets. The compression
kernel (tq_compress_paged.py) similarly uses strides.

Usage:
  python3 compact_kv_patch.py --apply    # Apply patch (backs up originals)
  python3 compact_kv_patch.py --revert   # Revert to original files
  python3 compact_kv_patch.py --check    # Check current state
  python3 compact_kv_patch.py --dry-run  # Show what would change

Memory savings for MiniMax-M2.5 (4 KV heads, D=128, block_size=16):
  Before: 2 * 16 * 4 * 128 = 16384 bytes/page
  After:  2 * 16 * 4 * 50  = 6400  bytes/page
  Savings: 61% -> 2.56x more KV cache capacity
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

# Base path for vLLM installation
VLLM_BASE = Path("/usr/local/lib/python3.12/dist-packages/vllm")

# Compressed slot size constants
_D = 128
_DQ3 = _D * 3 // 8  # 48
_NORM_BYTES = 2
_PERF_SLOT = _DQ3 + _NORM_BYTES  # 50

# ============================================================================
# Patch definitions: (file, old_text, new_text)
# ============================================================================

PATCHES = [
    # -----------------------------------------------------------------------
    # Patch 1: turboquant_attn.py — get_kv_cache_shape
    # Change slot_bytes from head_size (128) to compressed size (50)
    # -----------------------------------------------------------------------
    (
        "v1/attention/backends/turboquant_attn.py",
        # OLD: uses head_size directly (wastes 78 bytes per slot)
        '''\
    ) -> tuple[int, ...]:
        """Packed KV cache layout.
        Shape: [num_blocks, 2, block_size, num_kv_heads, head_size]
        where 2 = {K, V}. Only first 50 bytes (perf) or 60 bytes (quality)
        of each slot are used; rest is padding for allocator alignment.
        """
        slot_bytes = head_size
        return (num_blocks, 2, block_size, num_kv_heads, slot_bytes)''',
        # NEW: compute compressed slot size from head_size
        '''\
    ) -> tuple[int, ...]:
        """Compact packed KV cache layout.
        Shape: [num_blocks, 2, block_size, num_kv_heads, compressed_slot]
        where 2 = {K, V}, compressed_slot = D*3//8 + 2 = 50 bytes for D=128.
        No padding — allocator uses exact compressed size for 2.56x capacity.
        """
        # Compressed slot: 3-bit packed indices + fp16 norm
        DQ3 = head_size * 3 // 8  # 48 for D=128
        NORM_BYTES = 2
        slot_bytes = DQ3 + NORM_BYTES  # 50 for D=128
        return (num_blocks, 2, block_size, num_kv_heads, slot_bytes)''',
    ),

    # -----------------------------------------------------------------------
    # Patch 2: attention.py — get_kv_cache_spec
    # For TurboQuant, pass compressed head_size to FullAttentionSpec so
    # page_size_bytes matches the actual compact allocation.
    # -----------------------------------------------------------------------
    (
        "model_executor/layers/attention/attention.py",
        # OLD: always uses self.head_size
        '''\
        else:
            return FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
            )''',
        # NEW: TurboQuant uses compressed head_size
        '''\
        else:
            # TurboQuant: use compressed slot size as head_size for correct
            # page_size_bytes calculation (50 bytes vs 128 -> 2.56x capacity)
            spec_head_size = self.head_size
            spec_head_size_v = self.head_size_v
            if self.kv_cache_dtype == "turboquant":
                DQ3 = self.head_size * 3 // 8  # 48
                NORM_BYTES = 2
                compressed_slot = DQ3 + NORM_BYTES  # 50
                spec_head_size = compressed_slot
                spec_head_size_v = compressed_slot
            return FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=spec_head_size,
                head_size_v=spec_head_size_v,
                dtype=self.kv_cache_torch_dtype,
            )''',
    ),
]


def backup_path(filepath: Path) -> Path:
    return filepath.with_suffix(filepath.suffix + ".compact_kv_bak")


def check_state():
    """Check which patches are applied."""
    results = {}
    for rel_path, old_text, new_text in PATCHES:
        filepath = VLLM_BASE / rel_path
        bak = backup_path(filepath)
        if not filepath.exists():
            results[rel_path] = "MISSING"
            continue
        content = filepath.read_text()
        if new_text in content:
            results[rel_path] = "PATCHED"
        elif old_text in content:
            results[rel_path] = "ORIGINAL"
        else:
            results[rel_path] = "UNKNOWN (text not found — manual edit?)"
    return results


def apply_patches(dry_run=False):
    """Apply all patches, backing up originals."""
    state = check_state()
    
    for rel_path, old_text, new_text in PATCHES:
        filepath = VLLM_BASE / rel_path
        status = state[rel_path]
        
        if status == "PATCHED":
            print(f"  SKIP {rel_path} (already patched)")
            continue
        elif status == "MISSING":
            print(f"  ERROR {rel_path} (file not found)")
            return False
        elif status == "ORIGINAL":
            if dry_run:
                print(f"  WOULD PATCH {rel_path}")
                # Show diff
                lines_old = old_text.split('\n')
                lines_new = new_text.split('\n')
                print(f"    - {len(lines_old)} lines -> {len(lines_new)} lines")
                continue
            
            # Backup
            bak = backup_path(filepath)
            if not bak.exists():
                shutil.copy2(filepath, bak)
                print(f"  BACKUP {rel_path} -> {bak.name}")
            
            # Patch
            content = filepath.read_text()
            new_content = content.replace(old_text, new_text)
            filepath.write_text(new_content)
            print(f"  PATCHED {rel_path}")
        else:
            print(f"  ERROR {rel_path}: {status}")
            return False
    
    if not dry_run:
        # Clear __pycache__ for patched modules
        for rel_path, _, _ in PATCHES:
            cache_dir = (VLLM_BASE / rel_path).parent / "__pycache__"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                print(f"  CLEARED {cache_dir}")
    
    return True


def revert_patches():
    """Revert all patches from backups."""
    for rel_path, old_text, new_text in PATCHES:
        filepath = VLLM_BASE / rel_path
        bak = backup_path(filepath)
        
        if bak.exists():
            shutil.copy2(bak, filepath)
            bak.unlink()
            print(f"  REVERTED {rel_path}")
        else:
            # Try text-based revert
            if filepath.exists():
                content = filepath.read_text()
                if new_text in content:
                    new_content = content.replace(new_text, old_text)
                    filepath.write_text(new_content)
                    print(f"  REVERTED {rel_path} (text-based)")
                else:
                    print(f"  SKIP {rel_path} (not patched)")
            else:
                print(f"  SKIP {rel_path} (file missing)")
    
    # Clear __pycache__
    for rel_path, _, _ in PATCHES:
        cache_dir = (VLLM_BASE / rel_path).parent / "__pycache__"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"  CLEARED {cache_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compact KV cache allocator patch for TurboQuant")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true",
                       help="Apply compact KV cache patches")
    group.add_argument("--revert", action="store_true",
                       help="Revert to original files")
    group.add_argument("--check", action="store_true",
                       help="Check current patch state")
    group.add_argument("--dry-run", action="store_true",
                       help="Show what --apply would change")
    args = parser.parse_args()

    if args.check:
        print("Compact KV cache patch status:")
        for path, status in check_state().items():
            print(f"  {path}: {status}")
        
        # Show memory calculation
        print(f"\nMemory layout (D={_D}):")
        print(f"  Original slot:   {_D} bytes/slot")
        print(f"  Compact slot:    {_PERF_SLOT} bytes/slot")
        print(f"  Savings:         {100*(1 - _PERF_SLOT/_D):.0f}%")
        print(f"  Capacity boost:  {_D/_PERF_SLOT:.2f}x")
        
    elif args.dry_run:
        print("Dry run — changes that would be applied:")
        apply_patches(dry_run=True)
        
    elif args.apply:
        print("Applying compact KV cache patches...")
        if apply_patches():
            print("\nDone! Restart vLLM server to use compact allocation.")
            print(f"Expected capacity boost: {_D/_PERF_SLOT:.2f}x")
        else:
            print("\nFailed — some patches could not be applied.")
            sys.exit(1)
            
    elif args.revert:
        print("Reverting compact KV cache patches...")
        revert_patches()
        print("\nDone! Restart vLLM server to use original allocation.")


if __name__ == "__main__":
    main()
