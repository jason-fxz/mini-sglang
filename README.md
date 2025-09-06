# mini-sglang

A lightweight SGLang implementation built from scratch.

## Usage

launch server:

```bash
python -m mini_sglang.launch_server --model ~/huggingface/Qwen3-0.6B --gpu_memory_utilization 0.85 --log_level INFO --attention_backend fa3
```

send request:

```bash
curl -X POST "http://localhost:30000/generate" -H "Content-Type: application/json" -d '{
  "text": "The capital of France is",
  "sampling_params": {
      "temperature": 0,
      "max_new_tokens": 1024
  },
  "stream": true
}'
```

flush cache:

```bash
curl -X POST "http://localhost:30000/flush_cache"
```

## TODO

- Known Issues
  - ~~GPU memory leak~~
  - ~~GPU-CPU Synchronization issue~~

- Basic Architecture

  Tokenizer -> Managers -> Detokenizer three-stage architecture

  Managers:
  - Scheduler for scheduling
  - Model Runner used to call model forward / CUDA graph forward

  - [x] Qwen3 model structure
  - [x] Model Runner
  - [x] Scheduler
  - [x] Tokenizer-Detokenizer

- Scheduling
  - [x] FIFO
  - [x] aggressive max_new_token predict & decode retract
  - [x] Cache-Aware
  - [ ] chunked prefill

- KVcache Management
  - [x] page size == 1
  - [x] page size > 1
  - [x] Radix Attention
    - [x] prefix match
    - [x] evict strategy

- API support
  - [x] SGLang generate API
  - [ ] OpenAI Compatible API (basic)
  - [x] streaming output

- Others
  - [x] Tensor Parallelism
  - [x] CUDA graph support for decode
  - [ ] Overlap Scheduling

## Benchmark

A6000(40G), Qwen3-8B. Use sglang.bench_serving to benchmark.

```bash
python3 -m sglang.bench_serving --backend sglang --num-prompt 200 --request-rate 3
```

|                                         | mini-sglang | sglang   |
|-----------------------------------------|-------------|----------|
| Backend                                 | sglang      | sglang   |
| Traffic request rate                    | 3.0         | 3.0      |
| Max request concurrency                 | not set     | not set  |
| Successful requests                     | 200         | 200      |
| Benchmark duration (s)                  | 90.15       | 88.72    |
| Total input tokens                      | 64205       | 64205    |
| Total generated tokens                  | 42957       | 42957    |
| Total generated tokens (retokenized)    | 42954       | 42956    |
| Request throughput (req/s)              | 2.22        | 2.25     |
| Input token throughput (tok/s)          | 712.18      | 723.67   |
| Output token throughput (tok/s)         | 476.49      | 484.18   |
| Total token throughput (tok/s)          | 1188.66     | 1207.84  |
| Concurrency                             | 15.59       | 15.25    |
| **End-to-End Latency**                  |             |          |
| Mean E2E Latency (ms)                   | 7029.06     | 6763.83  |
| Median E2E Latency (ms)                 | 4582.41     | 4455.47  |
| **Time to First Token (TTFT)**          |             |          |
| Mean TTFT (ms)                          | 45.99       | 44.33    |
| Median TTFT (ms)                        | 45.97       | 43.87    |
| P99 TTFT (ms)                           | 63.18       | 60.25    |
| **Inter-Token Latency (ITL)**           |             |          |
| Mean ITL (ms)                           | 32.66       | 31.43    |
| Median ITL (ms)                         | 31.14       | 30.60    |
| P95 ITL (ms)                            | 57.04       | 54.65    |
| P99 ITL (ms)                            | 59.68       | 58.51    |
| Max ITL (ms)                            | 86.28       | 110.27   |
