# Building a Training Dataset

## What the Model Needs to Learn

Fine-tuning teaches the model WHAT TO DO, not what facts to know.

v1 made the mistake of teaching facts ("this domain handles these tasks").
The base model already knows facts. What it doesn't know is YOUR behavior:
how you want it to talk, when to call tools, what to refuse, what format to use.

**Train on behavior. Let the base model handle facts.**

---

## JSONL Format

Each training example is one JSON object per line:

```jsonl
{"messages": [
  {"role": "system", "content": "Your bot's system prompt here."},
  {"role": "user", "content": "User message"},
  {"role": "assistant", "content": "Bot response"}
]}
```

Multi-turn conversation:
```jsonl
{"messages": [
  {"role": "system", "content": "System prompt"},
  {"role": "user", "content": "First message"},
  {"role": "assistant", "content": "First response"},
  {"role": "user", "content": "Follow-up"},
  {"role": "assistant", "content": "Follow-up response"}
]}
```

Tool call format (Qwen3 native):
```jsonl
{"messages": [
  {"role": "system", "content": "System prompt"},
  {"role": "user", "content": "Log 5 km run"},
  {"role": "assistant", "content": "<tool_call>\n{\"name\": \"log_run\", \"arguments\": {\"distance_km\": 5}}\n</tool_call>"},
  {"role": "user", "content": "<tool_response>\n{\"status\": \"logged\", \"run_id\": 42}\n</tool_response>"},
  {"role": "assistant", "content": "Logged. 5 km run recorded."}
]}
```

The `<tool_call>` and `<tool_response>` XML wrappers match Qwen3's native tool call format.
Your training data must use these exact tags or the model will learn the wrong format.

---

## The 203 vs 500 Example Lesson

v1: 203 examples, r=8, 2 target modules -> eval loss 0.509, weak personality landing.
v2: 500 examples, r=16, 7 target modules -> eval loss 0.076, strong personality landing.

**Is 500 the magic number?** Not exactly. The interaction matters:

- With r=8 (narrow adapter), even 500 examples may not be enough
- With r=16 and all 7 modules, 500 examples is sufficient for clear personality differentiation
- The lesson: adapter capacity (r x target modules) must scale with dataset size

Rule of thumb:
- 200 examples: r=8, 2-3 target modules (minimum viable experiment)
- 500 examples: r=16, all 7 modules (personality training)
- 1000+ examples: r=32, consider multi-stage or DPO
- 2000+ examples: r=64, consider full SFT instead of LoRA

---

## What to Include in Training Data

### 1. Personality / voice examples (most important)

Show the model HOW the bot talks. Same question, distinctly different voices:

```python
# Health bot - encouraging, gym-bro
single(HEALTH_BOT_SYSTEM, "I missed my run today", "Happens. Don't compound it - just get back on track tomorrow. What's the plan for the week?")

# Finance bot - calm, precise
single(FINANCE_BOT_SYSTEM, "I missed my run today", "Noted. Not in scope for me, but let me know if you want to log it elsewhere.")
```

Each bot should have 30-50 pure voice examples before you add anything else.

### 2. Tool routing examples

When to call a tool vs when to respond directly:

```python
# Should call tool
single(BOT_SYSTEM, "Log a 5 km run", "<tool_call>\n{...}\n</tool_call>")

# Should NOT call tool
single(BOT_SYSTEM, "What's a good running pace for a beginner?", "For most beginners, 6-7 min/km is sustainable. ...")
```

### 3. Multi-turn conversations (64+ recommended)

v1 had zero multi-turn examples. This is why it struggled with context and follow-ups.
Add at least 10-15% multi-turn in your dataset.

### 4. Edge cases and refusals

What the bot should NOT do:
- Refuse out-of-scope requests
- Cross-bot routing ("that's more of a finance question")
- Graceful error handling when tools fail

### 5. Format compliance

If your deployment format is strict (e.g., Telegram HTML only, no markdown):
```python
# Wrong format - model must learn to NOT do this
single(BOT_SYSTEM, "Summarize my week", "## Week Summary\n- **Monday**: ...")

# Correct format
single(BOT_SYSTEM, "Summarize my week", "<b>Week Summary</b>\n- Monday: ...")
```

Add 20-30 format examples per bot if format compliance matters.

---

## Methods for Generating Training Data

### Method 1: Hand-write (highest quality, low volume)

Write examples yourself. Slow (2-3 minutes per example) but highest signal.
Target: 50-100 hand-written examples as the quality anchor for your dataset.

Use the `scripts/gen_dataset.py` structure as a template - define your system prompt
once, then call `single()` and `multi()` helpers to add examples.

### Method 2: Synthesize with a bigger model

Use GPT-4o or Claude to generate examples at scale:

```python
prompt = f"""
Generate 10 training examples for a bot with this system prompt:
{BOT_SYSTEM_PROMPT}

Format: JSON with messages array. One example per JSON block.
Topics: {', '.join(topic_list)}
Requirements: Use Telegram HTML format, no markdown, natural voice.
"""
```

Review and edit the synthetic output. Filter out anything that doesn't match the voice.
Use synthetic examples to fill coverage gaps, not as the primary signal.

### Method 3: Mine real transcripts

If you have real conversation logs (Telegram history, chat exports), these are the highest-
value training signal because they capture actual user behavior, not imagined user behavior.

Steps:
1. Export conversation logs as JSON
2. Filter for conversations where the bot performed well
3. Format as JSONL (map roles: "human" -> "user", "bot" -> "assistant")
4. Review each example - remove PII, off-topic messages, format violations

Real data is 5-10x more valuable per example than synthetic. Even 50 real examples
alongside 450 synthetic ones will meaningfully improve quality.

---

## Task Coverage Checklist

For each bot/persona in your training set, verify coverage across:

- [ ] Core task: happy path (straightforward request, correct tool call)
- [ ] Core task: follow-up question (multi-turn)
- [ ] Core task: tool error handling (tool returns error, bot recovers)
- [ ] Core task: multiple items in one request
- [ ] Voice: casual opener ("hey what's up")
- [ ] Voice: complex emotional context ("I've been struggling with X")
- [ ] Out-of-scope refusal (explicit redirect to correct bot/channel)
- [ ] Format compliance (your target format, not the base model's default)
- [ ] Currency/locale specifics (if applicable - base model often defaults to USD/EUR)
- [ ] Tool call: when to call
- [ ] Tool call: when NOT to call (just answer directly)

Missing any of these categories consistently produces visible failure modes in production.

---

## Data Quality Over Quantity

500 good examples beat 2000 mediocre ones.

Quality signals:
- Every example matches the target voice
- Tool calls use the exact format Qwen3 expects
- System prompt in each example matches what you'll use in production
- Multi-turn examples have natural flow (not "AI test prompt" energy)
- Edge cases reflect real user behavior, not theoretical scenarios

Red flags to filter:
- Assistant response starts with "Certainly!" or "Of course!" (base model leak)
- Response uses markdown when HTML is required
- Tool call arguments are syntactically wrong
- Response is too long for the channel (Telegram has practical message length norms)
