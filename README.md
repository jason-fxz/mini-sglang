# mini-sglang

A lightweight SGLang implementation built from scratch.

## TODO

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
  - [ ] Cache-Aware
  - [ ] chunked prefill

- KVcache Management
  - [x] page size == 1
  - [x] page size > 1
  - [ ] Radix Attention

- API support
  - [ ] SGLang generate API
  - [ ] OpenAI Compatible API (basic)
  - [ ] streaming output

- Others
  - [x] Tensor Parallelism
  - [ ] CUDA graph support for decode
  - [ ] Overlap Scheduling
