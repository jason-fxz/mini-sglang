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
