# LLM Inference System Simulator

I've tried to make a C++ simulator for how LLM inference servers actually schedule and serve requests. It models **continuous batching**, **paged KV cache memory**, **tiered SLO scheduling**, and **prefix caching with eviction**, which is the stuff that vLLM/ SGLang deal with under the hood. Note that its just a simulation so no GPU's or real models are needed here. The idea is simply to simulate the two things that actually bottleneck an inference server (compute time and KV cache memory) and use that to test how different scheduling and caching strategies behave. 

---

## How it works

Everything runs through one worker loop (`worker_continuous`) that re-evaluates the batch at every single generation step, meaning a request can join or leave the batch mid flight without waiting on anyone else to finish. That's the "continuous batching" part, as opposed to static batching where you wait for a full batch and run it to completion together (which is also implemented here in `worker_static`, mostly as a baseline to compare against and advance my understanding in chronological order).

```mermaid
graph TD
    subgraph Client [Client Workload]
        Req1[Request 1: Pro Tier]
        Req2[Request 2: Enterprise Tier]
        Req3[Request 3: Free Tier]
    end

    Req1 --> Q[Thread-Safe Queue]
    Req2 --> Q
    Req3 --> Q

    subgraph LLM_Engine [LLM Engine Simulator]
        Q --> Adm[Admission Controller\n FCFS / ShortPrompt / SLO]
        Adm -->|Admit| Active[Active Batch Slots]

        subgraph Memory [Memory Subsystem]
            Active <-->|Alloc/Release| KVA[KV Block Allocator]
            KVA -.->|Evict/Lookup| PC[Prefix Cache\n LRU / Cost-Ratio / GDS]
        end

        Active -->|Process| Tick[Iteration Tick]
        Tick -->|Output| Stream[Token Generation]
    end
```

Couple more aspects of it worth knowing are:

- **KV block allocator** -> memory is modelled as fixed-size blocks (16 tokens/block by default), the same way vLLM's PagedAttention works. Slots grab blocks as they generate tokens instead of pre-reserving a worst-case chunk of memory.
- **Chunked prefill** -> a long prompt doesn't get prefilled all at once, its split into chunks (`--chunk_size`, 256 tokens by default) so one huge prompt can't hog the compute budget for an entire tick and starve everyone else in the batch.
- **Preemption** -> if the allocator runs out of blocks mid-generation, the lowest-priority active request gets kicked back to the waiting queue (its blocks freed) so a more urgent one can keep going. This is the main source of "thrashing" you'll see in the stats.

## Admission policies

The admission controller decides who gets into the active batch each tick. Three are implemented:

- **`fcfs`** -> first come first served.
- **`shortprompt`** -> admits whichever waiting requests have the cheapest prefill cost first (shortest-job-first, basically). Good for throughput, bad for anyone unlucky enough to send a long prompt.
- **`slo`** -> every request is tagged with a tier (Enterprise / Pro / Free), each with its own time-to-first-token target. This policy sorts by *urgency*, meaning how close a request is to blowing its deadline and admits the most urgent ones first. 

Tier SLOs and traffic mix are currently hardcoded in `get_tier_slo()` and the trace generator: Enterprise gets 200ms, Pro gets 600ms, Free gets 2000ms, and the synthetic traffic is generated as 20% Enterprise / 30% Pro / 50% Free. 

## Prefix caching

If a bunch of requests share a common prefix (a system prompt, a few-shot template, etc.), there's no reason to recompute the KV states for that shared chunk every single time. When prefix caching is on (`--prefix_cache=true`), the first request through a given template pays the full prefill cost and the resulting KV blocks get cached; every later request against that same template skips straight to prefilling just its unique suffix.

```mermaid
sequenceDiagram
    participant Req as Request (Template A)
    participant Engine as Worker Loop
    participant Cache as Prefix Cache
    participant KV as KV Allocator

    Req->>Engine: Arrives (shared prefix + unique suffix)
    Engine->>Cache: Lookup Template A

    alt Cache Hit
        Cache-->>Engine: Hit (N blocks already cached)
        Engine->>Engine: Skip prefix prefill
        Engine->>KV: Allocate blocks for suffix only
    else Cache Miss
        Cache-->>Engine: Miss
        Engine->>KV: Allocate blocks for full prompt
        Engine->>Engine: Compute full prefill
        Engine->>Cache: Insert Template A into cache
    end
```

Because memory is tight, cached prefixes eventually have to get evicted to make room. Three eviction policies are implemented:

1. **`lru`** -> evict whichever cached prefix has sat idle the longest.
2. **`cost_ratio`** -> evict based on `idle_time / compute_cost`, trying to protect expensive to recompute prefixes. In practice this tends to let heavy prefixes squat in the cache forever(cache pollution), which hurts overall hit rate.
3. **`gds`** -> Greedy-Dual-Size. Scores each entry as `cost/size + L`, where `L` is a running clock that tracks the priority of the last-evicted item. This is the classic fix for the cache-pollution problem `cost_ratio` runs into, and it shows up in the numbers below.

## Benchmarks

Ran with `N=500` requests under a deliberately tight memory budget (100 total KV blocks, so eviction actually has to do work). Numbers are reproducible

**1. Does prefix caching actually help?**

| Metric | No cache | Prefix caching (LRU) | Change |
| :--- | :--- | :--- | :--- |
| Total duration | 188.5s | **146.7s** | 22% faster |
| Tokens processed (prefill + decode) | 176,704 | **92,994** | 47% less compute |
| TTFT (p95) | 85.3s | **43.5s** | ~49% lower |
| Preemptions | 82 | **40** | ~51% less thrashing |

![Impact of Prefix Caching](data/caching_impact.png)

**2. Which eviction policy wins under pressure?**

| Metric | LRU | Cost-Ratio | GDS |
| :--- | :--- | :--- | :--- |
| Cache hit rate | 82.78% | 82.22% | **83.33%** |
| Duration | 146.7s | 149.9s | **146.9s** |
| TTFT (p95) | **43.5s** | 46.9s | **43.5s** |

![Eviction Policy Performance](data/eviction_policies.png)

Cost-Ratio ends up hoarding expensive prefixes for too long and loses on both hit rate and speed as a result. GDS balances cost against size and recency and comes out ahead. Same idea as the real GreedyDual-Size algorithm, just applied to KV blocks instead of HTTP objects.

## Building and running

Only a standard C++17 compiler, no dependencies.

```bash
g++ -O3 -std=c++17 main.cpp -o main
```
(If your linker complains about pthreads, add `-pthread`.)

**Baseline, no caching:**
```bash
./main --policy=slo --n=500 --num_templates=5 --prefix_cache=false --total_kv_blocks=100
```

**With GDS prefix caching:**
```bash
./main --policy=slo --n=500 --num_templates=5 --prefix_cache=true --eviction=gds --total_kv_blocks=100
```

By default the simulator paces itself in real time to mimic actual request arrivals, so a run that reports 188s of simulated duration will also take roughly 188 real seconds to finish. Pass `--pace=0` to skip the sleeping and run the sim as fast as your CPU allows for busy, memory-constrained workloads (like the benchmark above) this gives identical numbers, but for light/low-traffic workloads it can produce unrealistic (even negative) wait times, since it removes the real-time sync between when requests "arrive" and when the scheduler is actually free to look at them. Use it for quick iteration, not for final numbers on lightly-loaded configs.

## CLI flags

`--help` only documents a handful of the flags, so here's the full list of what `parse_args` actually accepts:

| Flag | Default | What it does |
| :--- | :--- | :--- |
| `--mode` | `continuous` | `continuous` (iteration-level scheduling) or `static` (classic batch-and-wait) |
| `--policy` | `fcfs` | `fcfs`, `shortprompt`, or `slo` |
| `--n` | `200` | number of requests to simulate |
| `--lambda` | `5.0` | Poisson arrival rate (requests/sec) for the synthetic trace |
| `--seed` | `42` | RNG seed, for reproducible traces |
| `--max_batch_tokens` | `512` | per-tick compute budget, in tokens |
| `--chunk_size` | `256` | max tokens of a single prompt prefilled per tick |
| `--kv_block_size` | `16` | tokens per KV cache block |
| `--total_kv_blocks` | `256` | total KV blocks available (the memory budget) |
| `--num_templates` | `5` | how many of the 5 built-in prompt templates to sample from (0 = no shared prefixes at all) |
| `--prefix_cache` | `false` | turn prefix caching on/off |
| `--eviction` | `lru` | `lru`, `cost_ratio`, or `gds` |
| `--quadratic_cost` | `false` | adds a quadratic term to prefill cost, to simulate attention scaling with sequence length |
| `--pace` | `1` | sleep in real time to match simulated arrivals (see note above) |
| `--out` | `results.csv` | per-request CSV output; `none` to skip |
| `--summary` | `summary.csv` | one summary row gets *appended* per run, so you can point many runs (different seeds/policies) at the same file for sweeps |
| `--verbose` | `false` | also print a line per request to stdout |
| `--ttft_slo` | `500.0` | parsed but currently unused — per-tier SLOs are hardcoded in `get_tier_slo()` instead |



