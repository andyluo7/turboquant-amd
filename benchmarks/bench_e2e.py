#!/usr/bin/env python3
"""End-to-end serving benchmark for TurboQuant vs vanilla FP8.

Measures throughput, TTFT, and ITL using vLLM's benchmark_serving.py.
Compares TurboQuant (turbo3/turbo4) against vanilla FP8 KV cache.

Usage:
    # Start TQ server
    python -m vllm.entrypoints.openai.api_server \\
        --model MiniMax-Text-01 --tensor-parallel-size 2 \\
        --kv-cache-dtype turboquant --port 8000

    # Start vanilla server
    python -m vllm.entrypoints.openai.api_server \\
        --model MiniMax-Text-01 --tensor-parallel-size 2 \\
        --kv-cache-dtype fp8_e4m3 --port 8001

    # Run benchmark
    python bench_e2e.py --tq-port 8000 --vanilla-port 8001 \\
        --isl 31000 --osl 1024 --concurrency 64
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def run_benchmark(port: int, isl: int, osl: int, concurrency: int, output_file: str):
    """Run vLLM benchmark_serving against a server."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--host", "localhost",
        "--port", str(port),
        "--backend", "vllm",
        "--dataset-name", "sharegpt",
        "--num-prompts", str(concurrency * 10),
        "--request-rate", str(concurrency),
        "--max-concurrency", str(concurrency),
        "--input-len", str(isl),
        "--output-len", str(osl),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    with open(output_file, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="E2E TurboQuant benchmark")
    parser.add_argument("--tq-port", type=int, default=8000)
    parser.add_argument("--vanilla-port", type=int, default=8001)
    parser.add_argument("--isl", type=int, default=31000)
    parser.add_argument("--osl", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"E2E Benchmark: ISL={args.isl}, OSL={args.osl}, Conc={args.concurrency}")
    print("=" * 60)

    # TQ benchmark
    print("\n--- TurboQuant ---")
    tq_out = run_benchmark(
        args.tq_port, args.isl, args.osl, args.concurrency,
        str(output_dir / f"tq_isl{args.isl}_conc{args.concurrency}.txt"),
    )

    # Vanilla benchmark
    print("\n--- Vanilla FP8 ---")
    vanilla_out = run_benchmark(
        args.vanilla_port, args.isl, args.osl, args.concurrency,
        str(output_dir / f"vanilla_isl{args.isl}_conc{args.concurrency}.txt"),
    )

    print("\n" + "=" * 60)
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
