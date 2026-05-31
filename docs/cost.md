# Cost, Time, and Throughput

## v1 vs v2 Comparison

| | v1 (failed) | v2 (deployed) |
|---|---|---|
| Training time | 7m 29s | 33m 16s |
| GPU cost (A100 ~$1/hr) | $0.29 (billed 29 min) | $0.96 (billed ~58 min) |
| HuggingFace downloads | Included (first run) | Cached (second run) |
| Total spend | $0.29 | $0.96 |
| **Total for both runs** | **$1.25** | |

Note: v1 was charged for ~29 minutes including instance setup time (HF download + training).
v2 was faster because the HF model was already cached on the second instance from the same day.
In practice, if you're starting fresh, add 10-15 minutes for the 57 GB HF download.

## Realistic Cost Estimate for First-Time Run

| Phase | Time | Notes |
|---|---|---|
| Find instance | 2-5 min | No charge |
| Instance startup | 2-5 min | Billing starts |
| SSH + install deps | 3-5 min | On the clock |
| Flash Attention 2 compile | 5-10 min | Optional but recommended |
| HuggingFace model download | 10-15 min | 57 GB at ~60-100 MB/s |
| Training (500 examples) | 33 min | Core work |
| Adapter download | 2-3 min | 52 MB, fast |
| Destroy instance | 1 min | Billing stops |
| **Total wall time** | **~60-80 min** | |
| **Total cost at $1/hr** | **~$1.00-1.33** | |

Budget for $1.50 to be safe. Load $5 and you have 3+ full training runs.

## Per-Run Cost by GPU Type (Vast.ai approximate rates)

| GPU | VRAM | Rate | 30B MoE training | Can it work? |
|---|---|---|---|---|
| RTX 3090 | 24 GB | ~$0.13/hr | N/A | No - OOM |
| A6000 | 48 GB | ~$0.40/hr | N/A | No - OOM |
| A100 SXM4 80GB | 80 GB | ~$0.75-1.15/hr | Works | Yes |
| H100 80GB | 80 GB | ~$2.00-3.50/hr | Works | Yes (overkill) |

The A100 is the sweet spot. H100 would be faster but the training is already short -
spending 3x more for 30% faster training is not worth it at this scale.

## Inference Cost (Self-Hosted CPU)

Once deployed on your own machine, inference is effectively free (hardware you already own).

- Speed: ~13 tokens/second on AMD EPYC (64 GB RAM)
- Per-query cost: electricity only (~$0.000001 per query at local electricity rates)
- The 18 GB GGUF stays in RAM between requests

Compare to API inference costs for a comparable 30B model:
- Typical API pricing for 30B-class: $0.30-0.80 per million tokens
- At 13 t/s, you'd need ~21 hours of continuous generation to hit 1M tokens
- Self-hosted breaks even after the first few thousand API calls

## R2 / Cloud Backup Cost

Storing the 18 GB merged GGUF on Cloudflare R2:
- Storage: $0.015/GB/month = ~$0.27/month for the merged model
- Egress: $0 (R2 has no egress fees)
- Optional but recommended for disaster recovery

Storing adapters (52 MB each): negligible.

## Cost Scaling for Larger Datasets

Training time scales roughly linearly with examples x epochs:

| Examples | Epochs | Estimated time | Estimated cost |
|---|---|---|---|
| 203 | 3 | 7-10 min | $0.20-0.30 |
| 500 | 5 | 30-40 min | $0.75-1.00 |
| 1000 | 5 | 60-80 min | $1.00-1.50 |
| 2000 | 5 | 2-3 hours | $2.00-4.00 |

The cost stays modest because the model itself is not being trained from scratch -
only the small LoRA adapter (~52 MB) accumulates gradients.
