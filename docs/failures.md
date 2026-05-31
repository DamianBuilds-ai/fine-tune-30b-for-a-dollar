# Failure Log: 9 Attempts Before It Worked

This is the credibility anchor of the whole project. Nine things broke before the first
successful training run. Read this before you start - most of these are easy to avoid.

## The Failure Table

| # | Issue | Root Cause | Fix |
|---|---|---|---|
| 1 | Model generates garbage output | Ollama Modelfile has no chat template | MUST copy the full TEMPLATE block from the base model's Modelfile |
| 2 | Missing pip dependencies at training start | Used `--no-deps` flag which skips transitive dependencies | Install WITHOUT `--no-deps` flag |
| 3 | Tool calls returned as raw `<tool_call>` text | Modelfile missing RENDERER and PARSER directives | Add `RENDERER qwen3.5` and `PARSER qwen3.5` to Modelfile |
| 4 | Disk full during HuggingFace model download | Qwen3-30B HF cache = 57 GB, requested only 50 GB disk | Always request 200 GB+ disk |
| 5 | Rate limited during HF download | No HF_TOKEN set | Always export HF_TOKEN before running training script |
| 6 | CUDA version mismatch | Host machine CUDA != Docker image CUDA | Filter `cuda_vers>=13.0` when searching Vast.ai |
| 7 | OOM on RTX 3090 (24 GB VRAM) | 24 GB is not enough for 30B MoE training even at 4-bit | Minimum 80 GB VRAM - see hardware.md |
| 8 | OOM on A6000 (48 GB VRAM) | 48 GB still not enough with optimizer state + grad checkpoints | Minimum 80 GB VRAM - A100 is the floor |
| 9 | SSH rejected immediately after instance status "running" | Host-specific issue (bad machine) | Destroy and try another instance - don't debug host issues |

## Deeper Explanations

### #1: The chat template is not optional

When you create an Ollama model from a raw GGUF file, Ollama does not know how to format
the conversation. The base model on Ollama (e.g. `qwen3:30b-a3b`) has the template in its
manifest. A raw GGUF you produce yourself does not.

Without the template, the model does not parse `<|im_start|>system`, `<|im_start|>user`,
`<|im_end|>` markers correctly. It either ignores them or treats them as literal text to
continue, generating random tokens.

Fix: Get the template from the base model and paste it into your Modelfile:
```bash
ollama show qwen3:30b-a3b --modelfile | grep -A 100 'TEMPLATE'
```

The Modelfile in this repo has the correct template for Qwen3 MoE.

### #3: RENDERER and PARSER for tool calling

This is the most subtle failure. The model generates tool calls correctly (`<tool_call>` XML
with the right JSON inside), but without RENDERER/PARSER, Ollama returns them in the `content`
field as plain text instead of the `tool_calls` array in the OpenAI API response format.

Any Agents SDK (OpenAI Agents SDK, LangChain, etc.) that expects `tool_calls` in the response
structure will see an empty `tool_calls` list and treat the entire response as a text answer.

Fix: Add to Modelfile:
```
RENDERER qwen3.5
PARSER qwen3.5
```

Use `qwen3.5` (not `qwen3` or `qwen3moe` - those are unrecognized by Ollama 0.20.x).

Verify it works:
```bash
curl -s http://localhost:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"your-model","messages":[{"role":"user","content":"log 5 minutes of meditation"}],"tools":[{"type":"function","function":{"name":"log_mindfulness","description":"Log a mindfulness session","parameters":{"type":"object","properties":{"duration_mins":{"type":"integer"}},"required":["duration_mins"]}}}]}'
```

The response MUST contain `"tool_calls"` array. If it contains `<tool_call>` in `content`, RENDERER/PARSER are missing.

### #7 and #8: OOM is not a tuning problem

Getting OOM on RTX 3090 (24 GB) or A6000 (48 GB) is not solved by adjusting batch size,
gradient accumulation, or any other training hyperparameter. The model itself at 4-bit
quantization requires roughly 15-18 GB VRAM just to load. Training overhead (optimizer state,
gradient checkpoints, activations) adds another 35-40 GB on top.

The only fix is a bigger GPU. A100 80GB is the minimum for this model size.

## What v1 Got Wrong (Design Failures, Not Environment Failures)

The 9 failures above are environment/setup issues. The v1 model also had design failures
that caused it to produce mediocre results despite running successfully:

1. **203 examples is too few** to override the base model's pretraining at r=8
2. **r=8 is too narrow** for a 30B model to learn new patterns strongly
3. **Only 2 of 7 target modules** (q_proj + v_proj) - the model cannot rewire enough
4. **All single-turn examples** - no multi-turn context, no tool call format training
5. **Wrong training objective** - trained on system facts, not on behavior/personality

v2 fixed all five. The result was an 85% improvement in eval loss (0.509 -> 0.076).
