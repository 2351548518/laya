# compile=True and the TileLang fast path: engineering notes

These settings apply to the classic text encoder. The measurements come from #472, #576
and #718 on an RTX 4070 Ti SUPER with PyTorch 2.11 and TileLang 0.1.14.
[Native request latency](benchmarks.md#native-multimodal-checkpoints)

## Backend selection

`Agent(..., backend="auto")` and `laya.load(..., backend="auto")` opt into the backend
class layer. The default remains eager. Explicit `backend=` takes precedence over `compile`
and `fast`; omitting it preserves both flags' existing behaviour.

- `eager`: the stock PyTorch forward, on any supported device.
- `compile`: CUDA-only `torch.compile` with dynamic shapes and `reduce-overhead` mode,
  bucket padding, persistent inductor cache, and warmup at installation. It reuses the same
  independent-dimension scope as `compile=True`, which keeps its existing default mode and
  CPU support. Set `LAYA_COMPILE_WARMUP=0` to defer backend warmup and `LAYA_INDUCTOR_CACHE_DIR`
  to choose its cache directory (default `~/.cache/laya/inductor`).
- `tilelang`: an adapter around the current fast path, using the agent's bf16 or fp16 dtype.
- `auto`: TileLang on CUDA with a supported ModernBERT encoder and dtype when TileLang is
  installed, otherwise compile on CUDA; eager on other devices. One check comes before the device:
  a checkpoint on the parallel option layout (`Agent.parallel_options`, set by `option_layout` in
  its config) is eager on CUDA too, because the accelerated forwards do not take the parallel
  masks. Nothing flags the choice — `auto` answered with a backend, just not the one the clause
  above implies — so `agent.backend` is what says which arm ran, and `laya.backends.auto_policy` is
  the authority on the order of those checks.
- `onnx`: `laya.load(..., backend="onnx", onnx_path="model.onnx")` returns the existing
  `ONNXAgent`. Without `onnx_path`, it uses `laya.onnx`.

An unavailable backend emits a `RuntimeWarning` naming the resolved backend and falls back to
eager. To require a backend, use `agent.set_backend("tilelang", strict=True)`. Switching waits
for active inference; `agent.backend` reports the active name and `agent.backend_object`
exposes the installed object. `agent.set_backend("compile", warmup=False)` defers compilation
until inference, so compilation errors then surface on the request. `agent.warmup()` remains
available. `agent.deaccelerate()` removes a backend installed through the class layer.

Routers forward an explicit selection through `Router(agent_kwargs={"backend": "auto"})`.
They pass no backend argument by default, preserving compatibility with existing Agent-like
constructors. Scoped CPU OOM retries detach the backend and restore it when the model returns
to its original device.

## `compile=True` materialises the attention mask

Eager SDPA takes ModernBERT's `(rows, 1, L, L)` attention mask as a broadcast view.
`compile_model` (`laya/_compile.py:42`) passes `dynamic=True` to PyTorch. Under these dynamic
shapes, Inductor cannot prove that the last dimension is aligned. It expands the mask to every
head and pads it into a real buffer of `rows x heads x L x L`. In bf16 with 12 heads, that is
**0.8 GB at 32 rows x 1024 tokens**.

- **GPU with headroom.** The buffer costs bandwidth, tens of ms per long batch.
- **GPU nearly full.** The caching allocator thrashes, and the same call can take tens of seconds.

If you compile with long batches on a busy GPU, cap the batch size (`predict_batch(...,
batch_size=)`) or use `fast=True`. The TileLang attention reads the packed QKV buffer and masks
by sequence length, so it has no such buffer.

## Cold start

- **First compile.** It takes tens of seconds per graph. `compile=True` needs two graphs: one for
  batches and one for a single row, which torch specialises. `compile=True` now calls `agent.warmup()`
  during load. `compile_warmup=False` restores lazy compilation, and `agent.warmup(shapes=...)`
  remains available manually. Eager and TileLang loads do not warm automatically. These shapes
  cover common requests, not every possible shape guard.
- **Warm-up failure.** Automatic warm-up is best effort: a failure emits a `RuntimeWarning`
  naming the error (including the underlying compiler error) and load returns with the
  `torch.compile` wrapper and compile settings intact. For example, Windows without MSVC can
  load with `compile=True` even though warm-up fails. Later requests still use the compiled
  model and surface compilation failures; Laya does not switch them to eager execution.
  Explicit `agent.warmup()` calls also propagate failures, including after a failed automatic
  warm-up. A successful load therefore does not guarantee that compiled inference is ready.
- **Across restarts.** Inductor's FX-graph cache keeps compiled graphs under
  `TORCHINDUCTOR_CACHE_DIR`. The default is under `/tmp`, which does not survive a reboot or a
  container restart. Set it to a persistent directory, or a volume in a container, and a second
  process loads the graphs instead of compiling them. In #472's measurement that took warm-up from
  about 120 s to about 50 s.
- **Laya cache opt-in.** `laya.load(..., compile=True, compile_cache=True)` sets the process-wide
  `TORCHINDUCTOR_CACHE_DIR` only when absent, to `$XDG_CACHE_HOME/laya/torchinductor` or
  `~/.cache/laya/torchinductor` when XDG is unset or not absolute. An existing setting, including
  one set by an earlier PyTorch compile, wins. The directory is created at load; filesystem
  errors propagate. `compile_cache=False` (default), eager, and TileLang loads leave the
  environment alone. This does not move or delete old caches. Containers still need a persistent
  home/volume. Cache compatibility and invalidation are managed by PyTorch; a GPU, torch,
  compiler, model, or input guard change can require compilation again.

## Opt-in CUDA graphs

```python
agent = laya.load("convaiinnovations/laya", compile=True,
                  compile_cache=True, compile_mode="reduce-overhead")
```

`compile_mode` defaults to `"default"`; only `"default"` and `"reduce-overhead"` are accepted
on the active compiled path. Eager and TileLang loads ignore the compile options. CPU compilation
still works, but CUDA graph recording only applies on CUDA. The CUDA mode requires PyTorch's
`torch.compiler.cudagraph_mark_step_begin` API; older builds without it raise an explicit error.

Dynamic Dynamo graphs do not imply shape-independent CUDA graphs: new concrete shapes may
require warm-up and recording again, without a new Dynamo graph. The two default synthetic
warm-up shapes do not pre-record every request shape. Repeated shapes can benefit, but varying
shapes can pay extra latency and retain graph pools. PyTorch may skip CUDA graphs for unsupported
operations or configurations; setting this mode is not a guarantee of capture.

On `compile=True` with `compile_mode="reduce-overhead"`, Laya marks each compiled CUDA forward as
a new step, serializes these forwards across its agents, and clones both output tensors outside
the compiled graph before releasing the lock. `Agent._infer` enters `laya._compile.cuda_graph_step`
only when `_reduce_overhead` is set (`laya/agent.py:700`, read at `:1237`, entered at `:1238`), and
nothing outside this branch sets it; the clone of every output is at `:1250`. The lock
`cuda_graph_step` takes is a module-level `threading.Lock` (`laya/_compile.py:39`, taken at `:63`),
so every agent in the process shares it: that is what "across its agents" means. Together these
keep retained outputs valid across replays, at the cost of two copies and serialized forward
execution.

`backend="compile"` is a different route. It still copies its outputs out before releasing its
lock, but it marks no step and locks per agent rather than across agents. `CompileBackend.forward`
holds an `RLock` built per instance (`laya/backends/compile.py:79`, taken at `:114`) and clones the
sliced outputs inside it (`:117`); it does not import `cuda_graph_step`, so it makes no
`torch.compiler.cudagraph_mark_step_begin` call, and it never raises on a torch build without that
API. Two agents on `backend="compile"` therefore replay their graphs concurrently, with no step
marked between their forwards. The backend's default mode is `"reduce-overhead"`
(`laya/backends/compile.py:70`) and `backend="auto"` resolves to it on a CUDA machine without
tilelang (`laya.backends.auto_policy`), so this is the CUDA graph path that runs by default, not
the opt-in one. `agent.backend` names the route a load took; `agent.backend_object._lock` is the
specific lock a second `backend="compile"` agent would not be sharing.

In both routes the lock does not coordinate unrelated application-owned compiled models; callers
sharing CUDA graph iterations or using custom streams must manage their own coordination. Disk
caches reuse compiled code, not live CUDA graph recordings or their device memory, across
processes.

Reproduce cold/restart timings, memory, and cache counters with
`benchmarks/bench_compile_defaults.py`; see
[the recorded measurements](https://github.com/2351548518/laya/blob/main/benchmarks/results/compile-defaults/README.md).

## AOTInductor

Export and compile a classic checkpoint for a supported CUDA runtime with
`scripts/check_aoti.py`. The export uses an eval copy in an explicit dtype and declares
the token dimension in multiples of 16. The accelerator guide includes the package
recipe, numerical checks and measured timings. [AOTInductor](fast-backends.md#aotinductor)

## TileLang portability: the GPU specialisation is CUDA-only

The five TileLang kernels have a scalar FP32 path for reference checks. The
`compile_cpu` helper selects the C target and local buffers.
[CPU specialization and numerical checks](fast-backends.md#tilelang-portability)
