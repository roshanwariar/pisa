!pip install vllm transformers datasets pandas tqdm matplotlib
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import torch
import random
import math
import re
import matplotlib.pyplot as plt

# ==========================================
# SETUP
# ==========================================
model_name = "Qwen/Qwen3-14B"  # or Qwen/Qwen2.5-14B-Instruct

print(f"Loading {model_name}...")
llm = LLM(
    model=model_name,
    tensor_parallel_size=1,
    dtype="bfloat16",
    enable_prefix_caching=True, # Critical for Power Sampling speed
    gpu_memory_utilization=0.90,
    enforce_eager=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading Dataset...")
dataset = load_dataset("yentinglin/aime_2025", "default")
df = pd.DataFrame(dataset["train"])
problems = df["problem"].tolist()
answers = df["answer"].tolist()

SYSTEM_PROMPT = "You are in a math competition only use reasoning, no Python scripts. Show step-by-step, and put your final answer in \\boxed{}. "

def smc_logprob_beam(llm, tokenizer, prompt,
                     num_particles=16,
                     alpha=2.0,
                     temperature=0.6,
                     max_tokens=2048,
                     block_size=256):

    prompt_ids = tokenizer.encode(prompt)
    particles = [{"text": prompt, "ids": prompt_ids, "finished": False, "cum_score": 0.0} for _ in range(num_particles)]

    step = 0

    while True:
        # 1. Identify active particles
        active_indices = [i for i, p in enumerate(particles) if not p["finished"]]
        if not active_indices:
            break

        # --- BUG FIX: Check length of LONGEST active particle, not particles[0] ---
        # If we check particles[0], and particles[0] is finished, we loop forever.
        max_active_len = max(len(particles[i]["ids"]) for i in active_indices)
        if max_active_len >= max_tokens + len(prompt_ids):
            break

        # 2. Parallel Generation
        prompts_to_run = [particles[i]["text"] for i in active_indices]

        params = SamplingParams(
            temperature=temperature,
            max_tokens=block_size,
            top_p=0.95, top_k=-1, logprobs=1,
            stop_token_ids=[tokenizer.eos_token_id]
        )

        outputs = llm.generate(prompts_to_run, sampling_params=params, use_tqdm=False)

        # 3. Score
        step_scores = []
        new_extensions = []

        for i, output in enumerate(outputs):
            out_obj = output.outputs[0]
            new_ids = list(out_obj.token_ids)
            new_text = out_obj.text

            # Avg LogProb
            block_logprobs = []
            for t_idx, token_id in enumerate(new_ids):
                if t_idx < len(out_obj.logprobs):
                    lp = out_obj.logprobs[t_idx][token_id].logprob
                    block_logprobs.append(lp)
                else:
                    block_logprobs.append(-100.0)

            if block_logprobs:
                avg_score = sum(block_logprobs) / len(block_logprobs)
            else:
                avg_score = -100.0

            step_scores.append(avg_score)

            # Check finish status
            is_finished = (out_obj.finish_reason == "stop") or ("\\boxed" in new_text and "}" in new_text)

            new_extensions.append({
                "ids": new_ids,
                "text": new_text,
                "score": avg_score,
                "finished": is_finished
            })

        # 4. Resample
        max_s = max(step_scores)
        exp_scores = np.exp([alpha * (s - max_s) for s in step_scores])
        total_w = sum(exp_scores)

        if total_w == 0:
            probs = [1.0/len(step_scores)] * len(step_scores)
        else:
            probs = [w/total_w for w in exp_scores]

        num_to_select = len(active_indices)
        selected_indices_local = np.random.choice(len(active_indices), size=num_to_select, p=probs)

        # 5. Update Population
        next_gen_particles = []

        # Keep finished particles
        for i, p in enumerate(particles):
            if i not in active_indices:
                next_gen_particles.append(p)

        # Evolve active particles
        for local_idx in selected_indices_local:
            extension = new_extensions[local_idx]
            parent_global_idx = active_indices[local_idx]
            parent = particles[parent_global_idx]

            child = {
                "text": parent["text"] + extension["text"],
                "ids": parent["ids"] + extension["ids"],
                "finished": extension["finished"] or parent["finished"],
                "cum_score": parent.get("cum_score", 0) + extension["score"]
            }
            next_gen_particles.append(child)

        particles = next_gen_particles
        step += 1

    # Return particle with best score (approximate) or just the first valid one
    return particles[0]["text"]

# ==========================================
# RUN
# ==========================================

def extract_answer(text):
    matches = re.findall(r'\\boxed\s*\{(.*?)\}', text)
    if matches:
        return matches[-1].strip()
    return None

correct = 0
print("\n--- Starting Exp 9: Wide Stochastic Beam (16 Particles) ---")

for i, problem in tqdm(enumerate(problems), total=len(problems)):
    prompt = SYSTEM_PROMPT + problem

    # Num particles = 16 (High Width)
    solution = smc_logprob_beam(
        llm, tokenizer, prompt,
        num_particles=128,   # <--- The Muscle
        alpha=4.0,          # <--- Lower alpha to keep diversity in the larger pool
        temperature=0.6,
        block_size=256,
        max_tokens=4096
    )

    pred = extract_answer(solution)
    if pred == answers[i]:
        correct += 1

print(f"\nFinal Accuracy: {(correct/len(problems))*100:.2f}%")
