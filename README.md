# Asymmetric KV Cache Can Stall Prefill on CUDA (RTX 5060 Ti + Gemma 4 26B)

<!-- Search keywords: llama.cpp prefill stuck, prompt processing hangs, GPU 1% utilization, llama-server no response, Gemma 4 CUDA bug, KV cache q8_0 q4_0 -->

> **TL;DR —** On CUDA, using different quant types for K cache and V cache (e.g. `K=q8_0, V=q4_0`) can cause prompt evaluation to stall indefinitely.  
> The server looks healthy, VRAM is high, but GPU utilization drops to near zero. No OOM. No crash. It just hangs.  
> **Fix:** use matched KV types (`q4_0/q4_0` or `q8_0/q8_0`).

---

## Table of Contents

1. [Are You Hitting This Bug?](#are-you-hitting-this-bug)
2. [The Fix](#the-fix)
3. [What's Going On](#whats-going-on)
4. [Reproduction Matrix](#reproduction-matrix)
5. [Why This Is Not a VRAM Problem](#why-this-is-not-a-vram-problem)
6. [Diagnostic Flowchart](#diagnostic-flowchart)
7. [Environment](#environment)
8. [Log Evidence](#log-evidence)
9. [Reproduction Scripts](#reproduction-scripts)
10. [Status](#status)

---

## Are You Hitting This Bug?

Check every item. If all boxes match your situation, this is very likely the same issue.

- [ ] llama-server starts and loads the model without errors
- [ ] `GET /health` returns `{"status":"ok"}`
- [ ] You send an API request (chat completion, etc.)
- [ ] Prompt processing starts but **never finishes**
- [ ] There is **no** `prompt eval time` line in the server log
- [ ] GPU memory is high (near capacity)
- [ ] GPU utilization is near **0–5%** (should be 90%+ during prefill)
- [ ] GPU power draw is very low (~28 W on a 150 W card)
- [ ] `GET /slots` times out or returns stale data
- [ ] You are using **different** `--cache-type-k` and `--cache-type-v`

> **Key distinction:** In a real VRAM OOM, you'd see an explicit allocation failure or the process crashes.  
> Here the server is *alive* — it just never finishes processing the prompt. That's the tell.

---

## The Fix

Use **matched** KV cache types. Pick one:

```bash
# Option A — lower VRAM, works fine
--cache-type-k q4_0 --cache-type-v q4_0

# Option B — higher quality, also works fine
--cache-type-k q8_0 --cache-type-v q8_0
```

**Avoid** this combination:

```bash
# Hangs prefill — DO NOT USE until upstream fixes it
--cache-type-k q8_0 --cache-type-v q4_0
```

> Other asymmetric combos (e.g. `K=q4_0, V=q8_0`) were not tested. Treat any mixed K/V pair with caution.

---

## What's Going On

On this setup, llama.cpp's CUDA backend supports three KV cache configurations:

| Config | Behaviour |
|---|---|
| `K=q4_0, V=q4_0` | Works — up to 64K context tested |
| `K=q8_0, V=q8_0` | Works — up to 32K context tested |
| `K=q8_0, V=q4_0` | **Hangs** — even at 12K context |

The failure is **not** about running out of memory. `q8_0/q8_0` uses **more** VRAM than `q8_0/q4_0` — yet it completes normally.

This points to a bug in the asymmetric execution path: when K and V use different quantization formats, something in the CUDA kernel dispatch or memory layout stalls the prefill pipeline.

---

## Reproduction Matrix

All runs share identical flags — only the KV cache types change:

```
-c 12288 -fa on -b 4096 -ub 1024 -ngl 99 --device CUDA0 -np 1
```

| Context | K Cache | V Cache | Result | Performance |
|---|---|---|---|---|
| 12K | `q8_0` | `q8_0` | **PASS** | 3,056 t/s prompt eval, 82 t/s gen |
| 12K | `q8_0` | `q4_0` | **FAIL** | Stalls at checkpoint 1 of 32 |
| 12K | `q4_0` | `q4_0` | **PASS** | Normal |
| 32K | `q8_0` | `q8_0` | **PASS** | Normal |
| 48K | `q4_0` | `q4_0` | **PASS** | Normal |
| 64K | `q4_0` | `q4_0` | **PASS** | Normal |

> The problem is the **asymmetric pair**, not the context length.

---

## Why This Is Not a VRAM Problem

This is the counter-intuitive part.

**Intuition says:** `q8_0/q4_0` sits between `q4_0/q4_0` and `q8_0/q8_0` in VRAM usage. If anything, it should be the "safest middle ground."

**Reality:**

```
VRAM usage:     q4_0/q4_0  <  q8_0/q4_0  <  q8_0/q8_0
Prefill result:    PASS         FAIL           PASS
                  (lowest)    (middle)       (highest)
```

The **middle** option fails while the **highest**-VRAM option works. That rules out "not enough memory" and points to a code-path bug in how llama.cpp handles mixed K/V quantization on CUDA.

---

## Diagnostic Flowchart

```
Prefill stuck? Server looks fine but never returns?
│
├─ GPU VRAM high, GPU util ~0%, no OOM error?
│  │
│  ├─ YES → Check your KV cache flags
│  │        │
│  │        ├─ --cache-type-k == --cache-type-v ?
│  │        │  ├─ YES → Probably a different issue
│  │        │  └─ NO  → You found this bug. Use matched types.
│  │        │
│  │        └─ Not using --cache-type-k at all?
│  │           └─ You're using defaults → different issue
│  │
│  └─ NO (GPU util high, or OOM error, or crash)
│     └─ Different problem — not this bug
│
└─ Server returns errors / won't start?
   └─ Different problem — not this bug
```

---

## Environment

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti 16 GB |
| Driver | NVIDIA-SMI 610.47 |
| CUDA UMD | 13.3 |
| llama.cpp | `v9596 (18ef86ece)` — CUDA build (`ggml-cuda.dll`) |
| Model | `gemma-4-26B-A4B-it-QAT-Q4_0.gguf` |
| OS | Windows 10 |
| Server | `llama-server.exe` |
| Flash Attention | `-fa on` |
| GPU Offload | `-ngl 99 --device CUDA0` |
| Slots | `-np 1` |
| Batch / Micro-batch | `-b 4096 -ub 1024` |
| CPU | Intel Xeon E5-2666 v3 @ 2.90 GHz |

Server hardware detection:

```
CUDA0   : NVIDIA GeForce RTX 5060 Ti (16310 MiB, 15173 MiB free)
CPU     : Intel(R) Xeon(R) CPU E5-2666 v3 @ 2.90GHz (65376 MiB, 58888 MiB free)
system_info: CUDA : ARCHS = 750,800,860,890,900,1200,1210 | USE_GRAPHS = 1 | BLACKWELL_NATIVE_FP4 = 1
```

---

## Log Evidence

Full raw excerpts: [`logs/log-excerpts.md`](logs/log-excerpts.md)

### PASS — `q8_0/q8_0` (12K context)

```text
slot launch_slot_: id  0 | task 0 | processing task
slot create_check: id  0 | task 0 | created context checkpoint 1 of 32 (0.104 MiB)
slot create_check: id  0 | task 0 | created context checkpoint 2 of 32 (106.262 MiB)
slot create_check: id  0 | task 0 | created context checkpoint 3 of 32 (106.262 MiB)
slot print_timing: id  0 | task 0 | prompt eval time = 3206.06 ms / 9798 tokens (3056.09 t/s)
slot print_timing: id  0 | task 0 |        eval time = 11334.23 ms /  928 tokens (  81.88 t/s)
slot print_timing: id  0 | task 0 |       total time = 14540.29 ms / 10726 tokens
slot      release: id  0 | task 0 | stop processing: n_tokens = 10725
```

3 checkpoints → timing output → slot released. Normal.

### FAIL — `q8_0/q4_0` (12K context)

```text
srv  llama_server: model loaded
srv  llama_server: server is listening on http://0.0.0.0:8080
srv  update_slots: all slots are idle
slot launch_slot_: id  0 | task 0 | processing task
slot create_check: id  0 | task 0 | created context checkpoint 1 of 32 (0.080 MiB)

  ← STALLS HERE. No further output. Ever.

GET /health  → {"status":"ok"}
GET /slots   → timeout (5s, 0 bytes)
GPU memory   → 15150 MiB used
GPU util     → ~1%
Power        → ~28 W
```

Server is alive. Slot is stuck. GPU is idle. No crash, no error.

---

## Reproduction Scripts

Two `.bat` scripts for quick A/B testing. Edit `LLAMA_DIR` and `MODEL` paths, then run back-to-back.

| Script | KV Config | Expected |
|---|---|---|
| [`scripts/server-12k-q8q8-good.bat`](scripts/server-12k-q8q8-good.bat) | `K=q8_0, V=q8_0` | Normal |
| [`scripts/server-12k-Kq8-Vq4-bad.bat`](scripts/server-12k-Kq8-Vq4-bad.bat) | `K=q8_0, V=q4_0` | Hang |

---

## Status

| Aspect | State |
|---|---|
| Reproducible | Yes — consistently on this environment |
| Root cause | Suspected asymmetric KV cache kernel path issue in CUDA backend |
| Upstream report | Not yet filed — additional reproduction data welcome |
| Workaround | Use matched KV types (`q4_0/q4_0` or `q8_0/q8_0`) |

If you can reproduce this on your hardware, please open an issue with your environment details and logs. Extra data points would help get this reported upstream to llama.cpp.

---

<sub>Found this useful? A ⭐ helps others find it when they're debugging the same problem.</sub>
