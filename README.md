# RTX 5060 Ti + Gemma 4 26B: mixed KV cache `q8_0/q4_0` prefill hang in llama.cpp CUDA

A small reproducible note from local testing: on this setup, matched KV cache types work, but the mixed pair

```text
--cache-type-k q8_0 --cache-type-v q4_0
```

can hang during long-prefill / prompt evaluation, even at a small 12K context.

This is interesting because both matched configurations tested fine:

```text
q4_0 / q4_0  -> works, even at 48K / 64K context in local testing
q8_0 / q8_0  -> works at 12K and 32K context
q8_0 / q4_0  -> hangs/reproduces failure at 12K context
```

So this does **not** look like a simple VRAM-capacity problem. It looks like a bug/pathology in the mixed K/V cache dtype path for this model/build.

## TL;DR conclusion

For this environment, avoid:

```bat
--cache-type-k q8_0 --cache-type-v q4_0
```

Use matched KV types instead:

```bat
--cache-type-k q4_0 --cache-type-v q4_0
```

or, if quality/VRAM budget allows:

```bat
--cache-type-k q8_0 --cache-type-v q8_0
```

## Environment

Observed on:

| Item | Value |
|---|---|
| OS | Windows 10 |
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB |
| NVIDIA-SMI | 610.47 |
| CUDA UMD | 13.3 |
| llama.cpp | `version: 9596 (18ef86ece)` |
| llama.cpp backend | CUDA build, `ggml-cuda.dll` |
| Model | `gemma-4-26B-A4B-it-QAT-Q4_0.gguf` |
| Model source path locally | LM Studio community GGUF path |
| Server mode | `llama-server.exe` |
| Flash attention | `-fa on` |
| GPU offload | `-ngl 99 --device CUDA0` |
| Parallel slots | `-np 1` |

From server log:

```text
CUDA0   : NVIDIA GeForce RTX 5060 Ti (16310 MiB, 15173 MiB free)
CPU     : Intel(R) Xeon(R) CPU E5-2666 v3 @ 2.90GHz (65376 MiB, 58888 MiB free)
system_info: n_threads = 8 (n_threads_batch = 12) / 20 | CUDA : ARCHS = 750,800,860,890,900,1200,1210 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | BLACKWELL_NATIVE_FP4 = 1
```

## Minimal A/B

Both tests use:

```bat
-c 12288
-fa on
-b 4096
-ub 1024
--n-cpu-moe 0
--threads-batch 12
-ngl 99
--device CUDA0
-np 1
```

Only the KV cache dtype changes.

### Passing: 12K `q8_0/q8_0`

```bat
--cache-type-k q8_0 --cache-type-v q8_0
```

Log reached normal timing output:

```text
slot print_timing: id  0 | task 0 | prompt eval time =    3206.06 ms /  9798 tokens (    0.33 ms per token,  3056.09 tokens per second)
slot print_timing: id  0 | task 0 |        eval time =   11334.23 ms /   928 tokens (   12.21 ms per token,    81.88 tokens per second)
slot print_timing: id  0 | task 0 |       total time =   14540.29 ms / 10726 tokens
slot      release: id  0 | task 0 | stop processing: n_tokens = 10725, truncated = 0
```

User also confirmed 32K `q8_0/q8_0` passed.

### Failing/hanging: 12K mixed `q8_0/q4_0`

```bat
--cache-type-k q8_0 --cache-type-v q4_0
```

The server loads, `/health` returns OK, but prompt evaluation stalls after creating the first tiny checkpoint:

```text
slot launch_slot_: id  0 | task 0 | processing task, is_child = 0
slot create_check: id  0 | task 0 | created context checkpoint 1 of 32 (pos_min = 0, pos_max = 0, n_tokens = 1, size = 0.080 MiB)
```

No normal `prompt eval time` line appears.

During this bad state:

- `/health` still returns `{"status":"ok"}`
- `/slots` times out after 5 seconds with 0 bytes returned
- GPU memory is high, around 15.1 GiB, but GPU utilization is low
- This does not look like a normal OOM

Example runtime state during the bad mixed-KV run:

```text
GPU memory: 15150 MiB used
GPU util:   ~1%
Power:      ~28 W
```

## Earlier misleading hypothesis

At first this looked like plain VRAM pressure:

- 32K / 64K with `q8_0/q4_0` had very bad prefill behavior.
- GPU utilization stayed low during prefill.
- VRAM was close to full.

However, later tests contradicted a simple VRAM explanation:

- `q4_0/q4_0` worked smoothly at 48K and reportedly 64K.
- `q8_0/q8_0` worked at 12K and 32K.
- The same 12K context and `ub1024` failed only with `q8_0/q4_0`.

That strongly points to the mixed K/V dtype combination rather than q8 itself or total VRAM usage.

## Reproduction scripts

See:

- `scripts/server-12k-q8q8-good.bat`
- `scripts/server-12k-Kq8-Vq4-bad.bat`

Edit `LLAMA_DIR` and `MODEL` paths for your machine.

## Practical recommendation

For this model/build, treat this as unsafe:

```bat
--cache-type-k q8_0 --cache-type-v q4_0
```

Prefer matched KV cache types:

```bat
--cache-type-k q4_0 --cache-type-v q4_0
```

or:

```bat
--cache-type-k q8_0 --cache-type-v q8_0
```

## Status

This is a local empirical finding, not yet a confirmed upstream llama.cpp bug. The evidence is strong enough to avoid mixed `K=q8_0, V=q4_0` for this setup and worth reporting upstream if others can reproduce it.
