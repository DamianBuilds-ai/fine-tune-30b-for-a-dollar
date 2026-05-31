# Changelog

Notable changes to this project. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 2026-05-31

### Added
- README pipeline diagram: a Mermaid `flowchart LR` split into Training and Deployment subgraphs, tracing the full path from JSONL training data through the Vast.ai A100 QLoRA run, the 52 MB LoRA adapter, the llama.cpp merge, the 18 GB GGUF Q4_K_M model, Ollama, and ~13 tok/s CPU inference. Renders natively on GitHub.
- Static SVG hero at `docs/assets/hero.svg` leading with the cost story ($0.96, 33 min, 18 GB, ~13 tok/s). It is self-contained, with no scripts or external fonts, so it renders through GitHub's image proxy.
- README badge row: Apache-2.0, base Qwen3-30B-A3B, GGUF Q4_K_M, $0.96, ~13 tok/s.

### Changed
- Clarified that the $0.96 figure covers the full ~58 min billed session (instance startup, dependency install, and the 33m16s training run), not the 33-minute training time alone, so the cost survives an hourly-rate sanity check.
