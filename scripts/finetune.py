#!/usr/bin/env python3
"""QLoRA Fine-Tune on Vast.ai - v2 configuration (what worked)

This is the script that produced a working fine-tune at $0.96 / 33 minutes on an A100 80GB.

Key v2 changes vs v1:
- LoRA r=16 (was r=8), targeting all 7 attention + MLP layers (was 2)
- max_length 1024 (was 512)
- 500 examples with multi-turn (was 203 single-turn)
- 5 epochs (was 3)
- Gradient accumulation 4 (was 8) for more frequent updates
- Flash Attention 2 auto-detected (graceful fallback if not installed)

Flash Attention 2:
- Algorithm by Tri Dao -- tiles attention into SRAM blocks
- Reduces memory from O(N^2) to O(N), ~2-3x faster attention layers
- Requires: Ampere+ GPU (A100, RTX 3090/4090) + flash-attn package
- Install: pip install flash-attn --no-build-isolation (5-10 min compile)
- Enable: attn_implementation="flash_attention_2" in model loading
"""
import json, os, time, psutil, torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments)

# ============================================================
# CONFIGURATION - edit these before running
# ============================================================

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"   # HuggingFace model ID
DATA_PATH = "train.jsonl"                          # your training data file
OUTPUT_DIR = "./my-adapter"                        # where to save the LoRA adapter
LOG_PATH = "./training-progress.jsonl"             # live training log (tail -f this)

# ============================================================


class ProgressLogger(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        elapsed = time.time() - self.start_time
        sd = state.global_step
        total = state.max_steps
        eta = (elapsed / max(sd, 1) * (total - sd))
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        entry = {
            "step": sd, "total": total,
            "loss": logs.get("loss"),
            "eval_loss": logs.get("eval_loss"),
            "gpu_gb": round(gpu_mem, 2),
            "eta_min": round(eta / 60, 1),
            "time": time.strftime("%H:%M:%S"),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        pct = round(100 * sd / max(total, 1), 1)
        loss_val = entry.get("loss")
        ls = "loss={:.4f}".format(loss_val) if loss_val else ""
        eval_loss = entry.get("eval_loss")
        els = "eval_loss={:.4f}".format(eval_loss) if eval_loss else ""
        print("[{}] Step {}/{} ({:.1f}%) | {} {} | GPU: {:.1f}GB | ETA: {:.0f}min".format(
            entry["time"], sd, total, pct, ls, els, gpu_mem, entry["eta_min"]), flush=True)


def main():
    print("=" * 60)
    print("QLoRA Fine-Tune -- v2 configuration")
    print("LoRA r=16 | 7 target modules | max_length=1024")
    print("=" * 60)

    has_gpu = torch.cuda.is_available()
    if has_gpu:
        print("GPU: {}".format(torch.cuda.get_device_name(0)))
        print("VRAM: {:.1f}GB".format(torch.cuda.get_device_properties(0).total_memory / 1e9))
    else:
        print("ERROR: No GPU detected. An A100 80GB is required.")
        return

    # Check Flash Attention 2 availability
    fa2_available = False
    try:
        import flash_attn
        fa2_available = True
        print("Flash Attention 2: ENABLED (v{})".format(flash_attn.__version__))
    except ImportError:
        print("Flash Attention 2: NOT AVAILABLE (falling back to standard attention)")
        print("  Install: pip install flash-attn --no-build-isolation")

    # Load data
    ts = time.strftime("%H:%M:%S")
    print("\n[{}] Loading data from {}...".format(ts, DATA_PATH))
    with open(DATA_PATH) as f:
        raw = [json.loads(l) for l in f]
    print("  {} examples loaded".format(len(raw)))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Convert to chat format using the model's native chat template
    texts = []
    for item in raw:
        msgs = item.get("messages", [])
        if msgs:
            try:
                texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
            except Exception:
                # Fallback: manual ChatML formatting
                parts = []
                for m in msgs:
                    parts.append("<|im_start|>{}\n{}<|im_end|>\n".format(m["role"], m["content"]))
                texts.append("".join(parts))
        else:
            texts.append(item.get("text", ""))

    dataset = Dataset.from_dict({"text": texts})
    split = dataset.train_test_split(test_size=0.1, seed=42)
    print("  Train: {}, Val: {}".format(len(split["train"]), len(split["test"])))

    # Load model
    ts = time.strftime("%H:%M:%S")
    print("[{}] Loading model {} ...".format(ts, MODEL_ID))
    print("  HuggingFace will download ~57 GB on first run. Set HF_TOKEN to avoid rate limits.")

    model_kwargs = {"trust_remote_code": True}
    if fa2_available:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    loaded = False

    # Attempt 1: 4-bit NF4 quantization (preferred)
    try:
        print("  Attempting 4-bit NF4 quantization (all on GPU)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map={"": 0},
            **model_kwargs,
        )
        ts = time.strftime("%H:%M:%S")
        print("  [{}] 4-bit loaded. VRAM: {:.1f}GB".format(ts, torch.cuda.memory_allocated() / 1e9))
        loaded = True
    except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print("  4-bit failed: {}".format(e))

    # Attempt 2: 8-bit with CPU offload (fallback)
    if not loaded:
        try:
            print("  Falling back to 8-bit with CPU offload...")
            torch.cuda.empty_cache()
            import gc; gc.collect()
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory={0: "70GiB", "cpu": "120GiB"},
                **model_kwargs,
            )
            ts = time.strftime("%H:%M:%S")
            print("  [{}] 8-bit loaded. VRAM: {:.1f}GB".format(ts, torch.cuda.memory_allocated() / 1e9))
            loaded = True
        except Exception as e:
            print("  8-bit also failed: {}".format(e))
            print("  FATAL: Cannot load model. Need 80GB+ VRAM. See docs/hardware.md")
            return

    # LoRA config -- r=16, all attention + MLP modules
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",    # attention
            "gate_proj", "up_proj", "down_proj",         # MLP
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize
    ts = time.strftime("%H:%M:%S")
    print("[{}] Tokenizing (max_length=1024)...".format(ts))

    def tok_fn(ex):
        out = tokenizer(ex["text"], truncation=True, max_length=1024, padding="max_length")
        out["labels"] = [ids[:] for ids in out["input_ids"]]
        return out

    train_ds = split["train"].map(tok_fn, batched=True, remove_columns=["text"])
    val_ds = split["test"].map(tok_fn, batched=True, remove_columns=["text"])
    total_steps = (len(train_ds) * 5) // 4   # 5 epochs, grad_accum=4
    print("  Estimated steps: ~{}".format(total_steps))

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        optim="paged_adamw_8bit",
        fp16=False,
        bf16=True,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        warmup_steps=10,
        weight_decay=0.01,
        max_grad_norm=0.3,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[
            ProgressLogger(LOG_PATH),
            EarlyStoppingCallback(early_stopping_patience=5),
        ],
    )

    ts = time.strftime("%H:%M:%S")
    print("\n[{}] TRAINING STARTED".format(ts))
    print("Monitor: tail -f {}".format(LOG_PATH))
    print("=" * 60, flush=True)
    trainer.train()

    # Save adapter
    ts = time.strftime("%H:%M:%S")
    print("\n[{}] Saving adapter to {}...".format(ts, OUTPUT_DIR))
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "COMPLETE"), "w") as f:
        f.write(json.dumps({
            "model": MODEL_ID,
            "examples": len(raw),
            "completed": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": torch.cuda.get_device_name(0),
            "flash_attention_2": fa2_available,
            "lora_r": 16,
            "lora_targets": "q,k,v,o_proj + gate,up,down_proj",
            "max_length": 1024,
            "epochs": 5,
        }, indent=2))

    print("\nDONE! Adapter at {}".format(OUTPUT_DIR))
    print("Next step: download the adapter, destroy the instance, then merge.")
    print("See docs/deployment.md for the merge and Ollama setup.")


if __name__ == "__main__":
    main()
