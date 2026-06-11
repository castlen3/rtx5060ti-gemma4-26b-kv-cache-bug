# Log excerpts

These are cleaned excerpts from local llama-server logs.

## Passing control: 12K `q8_0/q8_0`

Config:

```text
-c 12288 -fa on --cache-type-k q8_0 --cache-type-v q8_0 -b 4096 -ub 1024 --n-cpu-moe 0
```

Result:

```text
slot launch_slot_: id  0 | task 0 | processing task, is_child = 0
slot create_check: id  0 | task 0 | created context checkpoint 1 of 32 (pos_min = 0, pos_max = 0, n_tokens = 1, size = 0.104 MiB)
slot create_check: id  0 | task 0 | created context checkpoint 2 of 32 (pos_min = 6722, pos_max = 8769, n_tokens = 8770, size = 106.262 MiB)
slot create_check: id  0 | task 0 | created context checkpoint 3 of 32 (pos_min = 7746, pos_max = 9793, n_tokens = 9794, size = 106.262 MiB)
slot print_timing: id  0 | task 0 | n_decoded =    100, tg =  81.30 t/s
slot print_timing: id  0 | task 0 | n_decoded =    349, tg =  82.39 t/s
slot print_timing: id  0 | task 0 | n_decoded =    593, tg =  81.91 t/s
slot print_timing: id  0 | task 0 | n_decoded =    837, tg =  81.71 t/s
slot print_timing: id  0 | task 0 | prompt eval time =    3206.06 ms /  9798 tokens (    0.33 ms per token,  3056.09 tokens per second)
slot print_timing: id  0 | task 0 |        eval time =   11334.23 ms /   928 tokens (   12.21 ms per token,    81.88 tokens per second)
slot print_timing: id  0 | task 0 |       total time =   14540.29 ms / 10726 tokens
slot print_timing: id  0 | task 0 |    graphs reused =        924
slot      release: id  0 | task 0 | stop processing: n_tokens = 10725, truncated = 0
srv  update_slots: all slots are idle
```

## Failing mixed KV: 12K `q8_0/q4_0`

Config:

```text
-c 12288 -fa on --cache-type-k q8_0 --cache-type-v q4_0 -b 4096 -ub 1024 --n-cpu-moe 0
```

Result:

```text
srv  llama_server: model loaded
srv  llama_server: server is listening on http://0.0.0.0:8080
srv  update_slots: all slots are idle
srv  params_from_: Chat format: peg-gemma4
slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1
srv  get_availabl: updating prompt cache
srv          load:  - looking for better prompt, base f_keep = -1.000, sim = 0.000
srv        update:  - cache state: 0 prompts, 0.000 MiB (limits: 8192.000 MiB, 12288 tokens, 8589934592 est)
srv  get_availabl: prompt cache update took 0.02 ms
reasoning-budget: activated, budget=2147483647 tokens
reasoning-budget: deactivated (natural end)
slot launch_slot_: id  0 | task 0 | processing task, is_child = 0
slot create_check: id  0 | task 0 | created context checkpoint 1 of 32 (pos_min = 0, pos_max = 0, n_tokens = 1, size = 0.080 MiB)
```

No normal `prompt eval time` line was produced. In this state, `/health` still returned OK, but `/slots` timed out after 5 seconds with 0 bytes returned.
