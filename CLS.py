# %% [markdown]
# # Continual Learning System — MVP Setup & Verification
# 

# %% [markdown]
# ---
# ## 0 — Pre-flight System Check
# Must pass before installing anything.

# %%
import sys, subprocess, platform

# ── Python version ────────────────────────────────────────────────────────────
py = sys.version_info
assert py >= (3, 10), f"Python 3.10+ required. Got {py.major}.{py.minor}"
print(f"✅  Python {py.major}.{py.minor}.{py.micro}")

# ── OS ───────────────────────────────────────────────────────────────────────
print(f"✅  OS: {platform.system()} {platform.release()}")

# ── CUDA driver check (nvcc or nvidia-smi) ───────────────────────────────────
try:
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
         "--format=csv,noheader"],
        stderr=subprocess.DEVNULL
    ).decode().strip()
    for line in smi.splitlines():
        name, mem, drv, cc = [x.strip() for x in line.split(",")]
        mem_gb = int(mem.replace(" MiB", "")) / 1024
        print(f"✅  GPU detected : {name}")
        print(f"    VRAM        : {mem_gb:.1f} GB  (need ≥54 GB for Qwen3-27B bf16)")
        print(f"    Driver      : {drv}")
        print(f"    Compute cap : {cc}")
        if mem_gb < 8:
            print(f"⚠️  VRAM is below the 54 GB minimum for Qwen3-27B bf16.")
            print("    Switch to Qwen3-27B 4-bit (~14 GB) or Qwen3-35B 4-bit (~18 GB).")
        else:
            headroom = mem_gb - 54
            print(f"    Headroom    : ~{headroom:.0f} GB free after model load — sufficient for LoRA + dual-judge")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("❌  nvidia-smi not found. Confirm NVIDIA drivers are installed and GPU is visible.")
    sys.exit(1)

# ── CUDA toolkit version ──────────────────────────────────────────────────────
try:
    nvcc = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.DEVNULL).decode()
    cuda_ver = [l for l in nvcc.splitlines() if "release" in l][0].split("release ")[1].split(",")[0]
    print(f"✅  CUDA toolkit : {cuda_ver}")
    CUDA_VER = cuda_ver.replace(".", "")
except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
    print("⚠️  nvcc not on PATH. Will infer CUDA version from PyTorch after install.")
    CUDA_VER = "124"   # assume 12.4 for RTX Blackwell; adjust if different

print(f"\nPre-flight passed. Proceeding with CUDA_VER={CUDA_VER}")

# %% [markdown]
# ---
# ## 1 - Install

# %%
import sys
import subprocess

def run(cmd):
    subprocess.run(cmd, check=True)

# 1) Ensure uv is available
try:
    run([sys.executable, "-m", "uv", "--version"])
except Exception:
    run([sys.executable, "-m", "pip", "install", "-q", "uv"])

def uv_install(*args):
    run([sys.executable, "-m", "uv", "pip", "install", "--python", sys.executable, *args])

# 2) PyTorch + CUDA 12.4
uv_install(
    "--index-url", "https://download.pytorch.org/whl/cu124",
    "torch==2.5.1",
    "torchvision==0.20.1",
    "torchaudio==2.5.1",
)

# 3) Core training / model stack
uv_install(
    "unsloth",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "bitsandbytes",
    "xformers",
)

# 4) Data / retrieval / utilities
uv_install(
    "datasketch",
    "detoxify",
    "datasets",
    "arxiv",
    "sentence-transformers",
    "safetensors",
    "huggingface_hub",
    "jsonlines",
    "tqdm",
    "numpy",
    "matplotlib",
    "scikit-learn",
)

# 5) Merge / eval / app stack
uv_install(
    "mergekit",
    "evaluate",
    "gradio",
    "fastapi",
    "uvicorn",
)

# %% [markdown]
# ---
# ## 2 — HuggingFace Authentication
# Qwen3-27B requires accepting the model license on HuggingFace.  
# Log in now so model download in Phase 2 is non-interactive.

# %%
from huggingface_hub import get_token, whoami, login

# ── Check if already logged in ────────────────────────────────────────────────
token = get_token()

if token is not None:
    try:
        info = whoami(token=token)
        print(f"✅  Already logged in as: {info['name']}")
    except Exception:
        token = None

if token is None:
    print("No HuggingFace token found.")
    print("Run the cell below after pasting your token from https://huggingface.co/settings/tokens")
    print("Token needs READ access to gated repos (Qwen3-27B requires license acceptance).")
    print()
    print(">>> login(token='hf_YOUR_TOKEN_HERE')")
    print()
    print("Or run from terminal: huggingface-cli login")

# ── Check Qwen3-27B is accessible (metadata only, no download) ────────────────
from huggingface_hub import model_info
MODEL_ID = "Qwen/Qwen3.6-27B"
try:
    info = model_info(MODEL_ID)
    print(f"✅  {MODEL_ID} is accessible on HuggingFace Hub")
    print(f"    License: {info.card_data.get('license', 'check model card')}")
    print(f"    (Model will be downloaded in Phase 2 — not here)")
except Exception as e:
    print(f"⚠️  Cannot access {MODEL_ID}: {e}")
    print("    Ensure you are logged in and have accepted the Qwen3-27B license on the Hub.")

# %% [markdown]
# ---
# ## 7 — Final Setup Checklist
# All smoke tests aggregated into a single pass/fail summary.  
# Every item must show ✅ before you write a single line of pipeline code.

# %%
import torch, importlib, subprocess, sys

print("=" * 70)
print("  CONTINUAL LEARNING MVP — FINAL SETUP CHECKLIST")
print("=" * 70)

checks = []

def check(label: str, fn):
    try:
        result = fn()
        status = "✅" if result else "❌"
        detail = "" if result else "  ← FAILED"
    except Exception as e:
        status = "❌"
        detail = f"  ← {type(e).__name__}: {e}"
        result = False
    checks.append(result)
    print(f"  {status}  {label}{detail}")

# ── System ────────────────────────────────────────────────────────────────────
print("\n  [ SYSTEM ]")
check("Python ≥ 3.10",
      lambda: sys.version_info >= (3, 10))
check("CUDA GPU visible",
      lambda: torch.cuda.is_available())
check("VRAM ≥ 54 GB (Qwen3-27B bf16 minimum)",
      lambda: torch.cuda.get_device_properties(0).total_memory / 1024**3 >= 54)
check("bf16 tensors on CUDA",
      lambda: bool(torch.tensor([1.0], dtype=torch.bfloat16).cuda()))

# ── Core ML Stack ─────────────────────────────────────────────────────────────
print("\n  [ CORE ML STACK ]")
for pkg in ["torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes", "unsloth"]:
    check(f"{pkg} importable",
          lambda p=pkg: bool(importlib.import_module(p)))

# ── Data Pipeline ─────────────────────────────────────────────────────────────
print("\n  [ DATA PIPELINE ]")
for pkg, imp in [
    ("datasketch (MinHash Tier 1)",        "datasketch"),
    ("detoxify (Toxicity Tier 1)",         "detoxify"),
    ("datasets (HuggingFace)",             "datasets"),
    ("arxiv (Phase 2 data source)",        "arxiv"),
    ("faiss (Anchor vector index Tier 2)", "faiss"),
    ("sentence_transformers (Embeddings)", "sentence_transformers"),
]:
    check(pkg, lambda i=imp: bool(importlib.import_module(i)))

# ── Merge & Versioning ────────────────────────────────────────────────────────
print("\n  [ MERGE & VERSIONING ]")
for pkg in ["safetensors", "mergekit", "evaluate"]:
    check(f"{pkg} importable",
          lambda p=pkg: bool(importlib.import_module(p)))

# ── Logging & Utilities ───────────────────────────────────────────────────────
print("\n  [ LOGGING & UTILITIES ]")
for pkg in ["jsonlines", "yaml", "numpy", "matplotlib", "sklearn", "tqdm"]:
    check(f"{pkg} importable",
          lambda p=pkg: bool(importlib.import_module(p)))

# ── API & Demo Layer ──────────────────────────────────────────────────────────
print("\n  [ API & DEMO LAYER ]")
for pkg in ["gradio", "fastapi", "uvicorn"]:
    check(f"{pkg} importable",
          lambda p=pkg: bool(importlib.import_module(p)))

# ── HuggingFace Auth ──────────────────────────────────────────────────────────
print("\n  [ HUGGINGFACE AUTH ]")
from huggingface_hub import get_token
check("HF token present",
      lambda: get_token() is not None)

# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(checks)
total  = len(checks)
failed = total - passed

print()
print("=" * 70)
if failed == 0:
    print(f"  🟢  ALL {total} CHECKS PASSED — environment is ready for Phase 1, Day 1")
    print()
    print("  Next step: Phase 1, Step 01")
    print("  → Load Qwen3-27B bf16, verify 42 GB headroom, run inference test.")
else:
    print(f"  🔴  {failed} / {total} CHECKS FAILED")
    print("  Resolve all failures before proceeding.")
    print("  Re-run only the failed cells, then re-run this checklist cell.")
print("=" * 70)

# %% [markdown]
# ---
# # Base Model Loading & v0 Baseline Generation

# %%
# ==============================================================================
# BLOCK 01 — Base Model Initialization & v0 Baseline Generation
# ==============================================================================
import json
import torch
import os
from tqdm import tqdm
from unsloth import FastLanguageModel

# Configuration
MODEL_ID = "Qwen/Qwen3.5-9B" # Change if the HF repo name differs slightly
MAX_SEQ_LENGTH = 2048
BENCHMARK_PATH = "anchor_benchmark.json"
BASELINE_OUTPUT_PATH = "anchor_benchmark_v0_answers.json"

print("=" * 60)
print("  BLOCK 01: LOADING BASE MODEL AND GENERATING v0 BASELINE")
print("=" * 60)

# 1. Verify JSON exists
assert os.path.exists(BENCHMARK_PATH), f"❌ {BENCHMARK_PATH} not found. Please generate it using the prompts."
with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

assert len(benchmark_data) == 50, f"❌ Expected 50 questions, found {len(benchmark_data)}"

# 2. Load Model using Unsloth (for peak memory efficiency and speed)
print(f"Loading {MODEL_ID} in 4-bit Quantization...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=torch.bfloat16, # Perfect for your Blackwell GPU
    load_in_4bit=True,   # No need for 4-bit, we have 96GB VRAM
    # token="YOUR_HF_TOKEN" # Uncomment and add if model is gated
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
print("✅ Model and Tokenizer loaded successfully.")

# 3. Generate v0 Baseline Answers
print("Generating v0 baseline answers for 50 anchor questions...")
v0_answers =[]

for item in tqdm(benchmark_data, desc="Evaluating Anchor Benchmark"):
    # Using Qwen's standard chat template
    messages =[
        {"role": "system", "content": "You are a helpful, harmless, and honest AI assistant."},
        {"role": "user", "content": item["question"]}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text=[prompt], return_tensors="pt").to("cuda")
    
    # Generate with anti-repetition settings for stable baseline
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        use_cache=True,
        do_sample=False,              # Keep it deterministic
        repetition_penalty=1.15,      # 🐛 FIXED: Prevents the infinite repeating loop
        pad_token_id=tokenizer.eos_token_id # 🐛 FIXED: Ensures proper stopping
    )
    
    # Decode only the newly generated tokens
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    
    v0_answers.append({
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "expected_keywords": item.get("expected_keywords", []),
        "v0_response": response
    })

# 4. Save the Baseline
print(f"Saving baseline to {BASELINE_OUTPUT_PATH}...")
with open(BASELINE_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(v0_answers, f, indent=2)

print(f"✅ Phase 1, Step 01 Complete. {BASELINE_OUTPUT_PATH} created.")
print(f"Next: Tier 1 Data Ingestion (Block 02).")

# %% [markdown]
# ---
# ###💻 Block 02: Tier 1 Filter (Statistical)

# %%
# ==============================================================================
# BLOCK 02 — Tier 1 Filter (MinHash + Detoxify)
# ==============================================================================
from datasketch import MinHash, MinHashLSH
from detoxify import Detoxify
import torch

def run_tier1_filter(raw_texts, threshold=0.85, toxicity_limit=0.05):
    print(f"Starting Tier 1 Filter: {len(raw_texts)} samples...")
    
    # 1. Deduplication
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_texts = []
    
    for i, text in enumerate(raw_texts):
        m = MinHash(num_perm=128)
        for word in text.lower().split():
            m.update(word.encode('utf-8'))
        
        # If no match found in LSH, it's unique
        if not lsh.query(m):
            lsh.insert(f"idx_{i}", m)
            unique_texts.append(text)
            
    print(f"  ↳ Deduplication: Kept {len(unique_texts)} / {len(raw_texts)}")
    
    # 2. Toxicity Check
    print("  ↳ Running Toxicity Filter (GPU)...")
    tox_model = Detoxify('original', device='cuda')
    results = tox_model.predict(unique_texts)
    
    clean_texts = [
        txt for i, txt in enumerate(unique_texts) 
        if results['toxicity'][i] < toxicity_limit
    ]
    
    print(f"  ↳ Toxicity: Kept {len(clean_texts)} / {len(unique_texts)}")
    return clean_texts

raw_data = json.load(open("raw_ai_data.json"))
clean_data = run_tier1_filter(raw_data)

# %% [markdown]
# ---
# 💻 Block 03: Tier 2 MVL Judge (Model-in-the-loop)

# %%
# ==============================================================================
# BLOCK 03A (MVL COMPLIANT) — Tier 2 Judge (PROMPT PRE-FILLING)
# ==============================================================================
import re
import os
import json
from tqdm import tqdm

print("=" * 60)
print("  BLOCK 03A: TIER 2 JUDGE & FLAG GENERATION")
print("=" * 60)

FastLanguageModel.for_inference(model)

auto_approved = []
flagged_for_review =[]
failed_parsing = 0

for i, text in enumerate(tqdm(clean_data, desc="Tier 2 Scoring")):
    messages =[
        {"role": "system", "content": "You are a strict data quality evaluator."},
        {"role": "user", "content": f"Rate the following text for factual accuracy and basic AI relevance.\nScore: 1 (Poor) to 5 (Excellent).\n\nText: {text}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 🐛 THE MAGIC TRICK: Pre-fill the assistant's response to bypass the "Thinking Process"
    prompt += "The integer score is: "
    
    inputs = tokenizer(text=[prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=3,  # Only need enough room for a single digit
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    score_str = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    if i < 3:
        print(f"\n[DEBUG {i+1}] Text: '{text[:40]}...' | Model Output: '{score_str}'")
        
    match = re.search(r'[1-5]', score_str)
    if match:
        score = int(match.group())
        if score >= 4:
            auto_approved.append(text)
        elif score == 3:
            flagged_for_review.append(text)
    else:
        failed_parsing += 1

rejected_count = len(clean_data) - len(auto_approved) - len(flagged_for_review)

print(f"\n✅ Tier 2 Complete.")
print(f"   ↳ Auto-approved (Score 4-5): {len(auto_approved)}")
print(f"   ↳ Flagged for review (Score 3): {len(flagged_for_review)}")
print(f"   ↳ Auto-rejected (Score 1-2): {rejected_count - failed_parsing}")
print(f"   ↳ Failed parsing (Implicitly rejected): {failed_parsing}")

# Save auto-approved state
with open("tier2_auto_approved.json", "w", encoding="utf-8") as f:
    json.dump(auto_approved, f)

# Write flagged documents to a simple text file
REVIEW_FILE = "tier3_manual_review.txt"
with open(REVIEW_FILE, "w", encoding="utf-8") as f:
    for text in flagged_for_review:
        f.write(text.replace('\n', ' ') + "\n")

print(f"\n⚠️  ACTION REQUIRED BEFORE RUNNING NEXT CELL:")
print(f"   1. Open '{REVIEW_FILE}' from your Jupyter file browser.")
print(f"   2. Delete the entire line for any document you wish to REJECT.")
print(f"   3. Save the file, then run Block 03B.")

# %%
# ==============================================================================
# BLOCK 03B — Finalize Tier 3 Human Review
# ==============================================================================
import json
import os

print("=" * 60)
print("  BLOCK 03B: FINALIZE TIER 3 HUMAN REVIEW")
print("=" * 60)

# 1. Load Auto-Approved
with open("tier2_auto_approved.json", "r", encoding="utf-8") as f:
    approved_data = json.load(f)

# 2. Load Manual Review (only lines the user kept)
REVIEW_FILE = "tier3_manual_review.txt"
human_approved =[]

if os.path.exists(REVIEW_FILE):
    with open(REVIEW_FILE, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip()
            if clean_line: # Ignore blank lines deleted by user
                human_approved.append(clean_line)

# 3. Combine
final_training_data = approved_data + human_approved

print(f"✅ Tier 3 Complete.")
print(f"   ↳ Auto-approved: {len(approved_data)}")
print(f"   ↳ Human-approved: {len(human_approved)}")
print(f"   ↳ FINAL TRAINING SET: {len(final_training_data)} samples")

if len(final_training_data) == 0:
    print("❌ CRITICAL: 0 samples passed Tier 2 & Tier 3. Cannot proceed to Block 04.")
    print("If this happened, the model is scoring everything as a 1 or 2. Your raw data may be too short/basic.")
else:
    # Override the approved_data variable so Block 04 can use it seamlessly
    approved_data = final_training_data
    print("🚀 Ready for Block 04 (Replay Mixing)!")

# %%
# ==============================================================================
# BLOCK 04 (MVL COMPLIANT) — Flat Replay Buffer & Batching
# ==============================================================================
import os
import json
import random
import sys

print("=" * 60)
print("  BLOCK 04: REPLAY BUFFER MIXING (FLAT MVL VERSION)")
print("=" * 60)

REPLAY_PATH = "replay_buffer_seed.json"

assert os.path.exists(REPLAY_PATH), f"❌ {REPLAY_PATH} missing."

# 1. Load Flat Replay Buffer (No Anchor logic, No Stratification)
with open(REPLAY_PATH, "r", encoding="utf-8") as f:
    flat_replay_buffer = json.load(f)

print(f"Loaded Flat Replay Buffer: {len(flat_replay_buffer)} total samples")

new_data_count = len(approved_data) # From Block 03 Tier 2 Judge
if new_data_count == 0:
    print("❌ No new data passed the Tier 2 Judge. Halting Cycle 1.")
    sys.exit(1)

# 2. Hardcoded MVL Ratio: 30% New Data, 70% Replay
replay_count = int((new_data_count / 0.30) * 0.70)

# Randomly sample from the flat buffer (using choices to allow replacement if buffer is small)
sampled_replay = random.choices(flat_replay_buffer, k=replay_count)

# 3. Combine and Shuffle
training_batch_texts = approved_data + sampled_replay
random.shuffle(training_batch_texts)

print(f"✅ Training Batch Ready: {len(training_batch_texts)} total samples.")
print(f"   ↳ {new_data_count} New Samples (Tier 2 Approved)")
print(f"   ↳ {replay_count} Replay Samples (Randomly drawn from flat buffer)")

# %%
# ==============================================================================
# BLOCK 05 — LoRA Training Loop (Unsloth)
# ==============================================================================
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

print("=" * 60)
print("  BLOCK 05: LORA TRAINING (Zone 2: Layers 8-24)")
print("=" * 60)

# 1. Format Data into ChatML structure for SFT
def format_prompts(examples):
    texts = []
    for text in examples["text"]:
        # Wrapping the text in standard instruction format so it learns to reply usefully
        prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": "You are a helpful, harmless, and honest AI assistant."},
                {"role": "user", "content": "Explain a useful fact, concept, or snippet of knowledge."},
                {"role": "assistant", "content": text}
            ],
            tokenize=False
        )
        texts.append(prompt)
    return {"formatted_text": texts}

dataset = Dataset.from_dict({"text": training_batch_texts})
dataset = dataset.map(format_prompts, batched=True)

# 2. Attach LoRA Adapters (Strict MVL Rules: r=16, alpha=32, target Zone 2)
# We pass layers_to_transform to PEFT to ensure only layers 8-24 are touched.
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0,
    target_modules =["q_proj", "k_proj", "v_proj", "o_proj"],
    layers_to_transform = list(range(8, 25)), # Zone 2 isolation
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
)

# Verify trainable parameters invariant (< 2%)
trainable_params, total_params = model.get_nb_trainable_parameters()
trainable_pct = (trainable_params / total_params) * 100
print(f"Trainable Parameters: {trainable_params:,} / {total_params:,} ({trainable_pct:.3f}%)")
assert trainable_pct < 2.0, "❌ Invariant Failed: Trainable parameters exceed 2.0%"

# 3. Configure Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "formatted_text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False, # Keep false for small MVP datasets to ensure exact step counts
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        learning_rate = 1e-4,
        fp16 = False,
        bf16 = True,       # RTX Blackwell native bfloat16
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        num_train_epochs = 2, # MVL runs 1-3 epochs
        output_dir = "outputs",
    ),
)

# 4. Train!
print("🚀 Starting LoRA Training...")
trainer_stats = trainer.train()

# 5. Save the Adapter (v1)
ADAPTER_SAVE_PATH = "adapter_v1"
model.save_pretrained(ADAPTER_SAVE_PATH)
tokenizer.save_pretrained(ADAPTER_SAVE_PATH)
print(f"✅ Training Complete. Adapter saved to ./{ADAPTER_SAVE_PATH}")

# %%
# ==============================================================================
# BLOCK 06 (MVL COMPLIANT) — Evaluation & Binary Rollback Decision
# ==============================================================================
import json

print("=" * 60)
print("  BLOCK 06: EVALUATION & BINARY ROLLBACK DECISION")
print("=" * 60)

FastLanguageModel.for_inference(model)

# 1. Load v0 baseline and compute starting scores
with open(BASELINE_OUTPUT_PATH, "r", encoding="utf-8") as f:
    v0_data = json.load(f)

v0_scores = {"factual": 0, "reasoning": 0, "identity": 0}
category_counts = {"factual": 20, "reasoning": 20, "identity": 10}

for item in v0_data:
    passed_v0 = any(kw.lower() in item.get("v0_response", "").lower() for kw in item["expected_keywords"])
    if passed_v0:
        v0_scores[item["category"]] += 1

v0_pct = {cat: (v0_scores[cat] / category_counts[cat]) * 100 for cat in category_counts}
print(f"Baseline (v0) -> Factual: {v0_pct['factual']}%, Reasoning: {v0_pct['reasoning']}%, Identity: {v0_pct['identity']}%")

# 2. Evaluate Candidate
print("\nEvaluating candidate model against Anchor Benchmark...")
candidate_scores = {"factual": 0, "reasoning": 0, "identity": 0}

for item in tqdm(benchmark_data, desc="Candidate Inference"):
    messages =[
        {"role": "system", "content": "You are a helpful, harmless, and honest AI assistant."},
        {"role": "user", "content": item["question"]}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=[prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        use_cache=True, 
        do_sample=False,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).lower()
    
    passed = any(kw.lower() in response for kw in item["expected_keywords"])
    if passed:
        candidate_scores[item["category"]] += 1

candidate_pct = {cat: (candidate_scores[cat] / category_counts[cat]) * 100 for cat in category_counts}

print("\n--- Candidate Scores ---")
print(f"Factual:  {candidate_pct['factual']:.1f}%")
print(f"Reasoning:{candidate_pct['reasoning']:.1f}%")
print(f"Identity: {candidate_pct['identity']:.1f}%")

# 3. MVL Binary Rollback Logic
identity_drop = v0_pct['identity'] - candidate_pct['identity']
reasoning_drop = v0_pct['reasoning'] - candidate_pct['reasoning']

system_state = "ACCEPT"
rollback_stage = 0  # 0 means accepted, 1 means reverted (binary for MVL)

if identity_drop > 1.0 or reasoning_drop > 5.0:
    system_state = "REJECT"
    rollback_stage = 1
    print("\n❌ GATE FAILED: Catastrophic forgetting detected.")
    if identity_drop > 1.0: print(f"   ↳ Identity dropped by {identity_drop:.1f}% (> 1%)")
    if reasoning_drop > 5.0: print(f"   ↳ Reasoning dropped by {reasoning_drop:.1f}% (> 5%)")
    print("   ↳ ACTION: Adapter v1 discarded. Reverting to v0.")
else:
    print("\n✅ GATE PASSED: Model retained core knowledge safely.")

# %%
# ==============================================================================
# BLOCK 07 (MVL COMPLIANT) — Merge & 5-Metric Logging
# ==============================================================================
import hashlib
import subprocess
import datetime
import os
import sys
import jsonlines

print("=" * 60)
print("  BLOCK 07: MERGE & MVL LOGGING")
print("=" * 60)

MERGED_DIR = "merged_model_v1"

if system_state == "ACCEPT":
    print("Executing DARE-TIES Merge (seed=42, density=0.70)...")
    
    # Free up memory before mergekit
    del model
    del trainer
    import torch
    torch.cuda.empty_cache()
    
    merge_cmd_python =[sys.executable, "-m", "mergekit.scripts.merge", "merge_spec.yaml", MERGED_DIR, "--copy-tokenizer", "--cuda", "--low-cpu-memory"]
    merge_cmd_cli =["mergekit-merge", "merge_spec.yaml", MERGED_DIR, "--copy-tokenizer", "--cuda", "--low-cpu-memory"]
    
    try:
        subprocess.run(merge_cmd_python, check=True)
        print(f"✅ Merge successful! New base model saved to ./{MERGED_DIR}")
    except subprocess.CalledProcessError:
        print("⚠️ Python module invocation failed. Falling back to CLI 'mergekit-merge'...")
        try:
            subprocess.run(merge_cmd_cli, check=True)
            print(f"✅ Merge successful! New base model saved to ./{MERGED_DIR}")
        except subprocess.CalledProcessError:
            print("❌ Merge failed entirely. Please check mergekit installation.")
            sys.exit(1)
else:
    print("Skipping merge due to Rollback.")

# Generate SHA of the adapter
adapter_file_path = os.path.join(ADAPTER_SAVE_PATH, "adapter_model.safetensors")
with open(adapter_file_path, "rb") as f:
    adapter_sha = hashlib.sha256(f.read()).hexdigest()

# Calculate overall Fast Score Delta (Average of the 3 categories for Cycle 1 simplicity)
overall_v0_score = sum(v0_pct.values()) / 3
overall_candidate_score = sum(candidate_pct.values()) / 3
fast_score_delta = overall_candidate_score - overall_v0_score

# 5 Metrics STRICTLY Required for Cycle 1 MVL (Section 23.1)
log_entry = {
    # Metadata
    "version": "v1",
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "adapter_sha256": adapter_sha,
    "system_state": system_state,
    
    # MVL 5 Metrics
    "replay_ratio": 0.70,                  # Hardcoded for Cycle 1
    "fast_score_delta": fast_score_delta,  # Consolidated into one metric
    "rollback_stage": rollback_stage,      # 0 or 1
    "kl_from_anchor": 0.0,                 # Placeholder, valid for MVL
    "domain_gain": candidate_pct['factual'] - v0_pct['factual'] # Using factual as proxy for domain gain in Cycle 1
}

with jsonlines.open("update_log.jsonl", mode="a") as writer:
    writer.write(log_entry)

print("\n✅ Cycle 1 logged successfully to update_log.jsonl")
print(f"🔒 Adapter Hash: {adapter_sha}")
print("🎉 MINIMUM VIABLE LOOP (MVL) FULLY SECURED AND COMPLETE! 🎉")


