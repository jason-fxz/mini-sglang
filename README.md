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
  - [ ] Scheduler
  - [ ] Tokenizer-Detokenizer

- Scheduling
  - [ ] FIFO
  - [ ] Cache-Aware
  - [ ] chunked prefill

- KVcache Management
  - [ ] page size == 1
  - [ ] page size > 1
  - [ ] Radix Attention

- API support
  - [ ] SGLang generate API
  - [ ] OpenAI Compatible API (basic)
  - [ ] streaming output

- Others
  - [ ] Tensor Parallelism
  - [ ] CUDA graph support for decode
  - [ ] Overlap Scheduling
