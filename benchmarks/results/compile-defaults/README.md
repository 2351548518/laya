# Classic compile default measurements

The 2026-10-03 runs used the classic English checkpoint at revision
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, an RTX 4070 Ti SUPER 16 GiB,
PyTorch 2.11.0+cu130 and NVIDIA driver 615.71.09. The code was based on
`fa9a2a7`. Each process sent three passes over the same ten request shapes
and checked within-process answer equality. Request 8 is the first single-row request.
Memory columns show peak PyTorch allocator memory. Warm-up runs shared the GPU
with other jobs. FX cache hits record disk reuse across processes.

## Constructor warm-up

| Run | Load s | Request 1 ms | Request 8 ms | Peak alloc / reserved MiB | FX hits / misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Lazy cold](baseline-cold.log) | 3.078 | 50154.96 | 40468.15 | 1680.0 / 1820.0 | 0 / 2 |
| [Lazy restart](baseline-restart.log) | 3.124 | 23281.17 | 18366.73 | 1671.0 / 1794.0 | 2 / 0 |
| [Warm-up cold](warmup-cold.log) | 108.498 | 26.61 | 10.91 | 1685.6 / 1820.0 | 0 / 2 |
| [Warm-up restart](warmup-restart.log) | 61.314 | 27.42 | 10.69 | 1671.0 / 1796.0 | 2 / 0 |

Warm-up moves compilation into checkpoint load. The restart reused both compiled FX graphs.

## Persistent cache

| Run | Load s | Request 1 ms | Request 8 ms | Peak alloc / reserved MiB | FX hits / misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Verified cold](cache-verified-cold.log) | 111.643 | 22.86 | 8.69 | 1685.6 / 1820.0 | 0 / 2 |
| [Verified restart](cache-verified-restart.log) | 51.292 | 24.43 | 10.84 | 1671.0 / 1796.0 | 2 / 0 |

The verified processes used the same isolated XDG cache location. Earlier attempts
initialized Dynamo counters before configuring the Laya cache and selected another directory.
The benchmark now checks the selected path and imports counters after model load.

## CUDA graph mode

| Run | Load s | Request 1 ms | Request 8 ms | Pass 2 median ms | Pass 3 median ms | Peak alloc / reserved MiB | CUDA graphs | FX hits / misses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [Default reference](mode-default-reference.log) | 46.659 | 22.27 | 8.67 | 12.64 | 12.65 | 1671.0 / 1796.0 | 0 | 2 / 0 |
| [Reduce-overhead cold](mode-cold.log) | 99.089 | 25.73 | 14.69 | 26.94 | 12.41 | 1687.5 / 1892.0 | 10 | 0 / 2 |
| [Reduce-overhead restart](mode-restart.log) | 48.100 | 25.00 | 13.86 | 28.72 | 13.83 | 1669.5 / 1818.0 | 10 | 2 / 0 |

The two CUDA graph runs matched the default reference with a maximum decoded-answer
delta of zero. They recorded ten CUDA graphs in each process. Their cold-run recording
pass took 26.94 ms median. The results describe this checkpoint, device and request
sequence. [Decoded parity report](mode-parity.log)

## Reproduce

Set `MODEL` to the pinned checkpoint directory. Run the following twice as separate
processes with the same absolute cache root to compare cold load and restart. Use a fresh
root for each variant. Pass `--no-warmup` for lazy compilation or add
`--mode reduce-overhead` for CUDA graph recording.

```bash
env -u TORCHINDUCTOR_CACHE_DIR XDG_CACHE_HOME="$PWD/_local/compile-defaults" \
  HF_HUB_OFFLINE=1 python benchmarks/bench_compile_defaults.py --model "$MODEL" --cache
```

The logs linked above record request timings, packed answers, cache counters and device
snapshots. Compare the three saved mode logs with the original parity checker.

```bash
python benchmarks/compare_compile_defaults.py \
  benchmarks/results/compile-defaults/mode-default-reference.log \
  benchmarks/results/compile-defaults/mode-cold.log \
  benchmarks/results/compile-defaults/mode-restart.log
```

Use `python -m pytest tests/test_compile.py -q` for weight-free compilation checks and
`python tests/test_compile_cuda.py` for CUDA graph replay and two-thread parity. The older
trial files and CI transcripts remain in Git history.
[Original run archive](https://github.com/2351548518/laya/tree/643419567b25711662ead312f7d7a1e5bb03b234/benchmarks/results/compile-defaults)
