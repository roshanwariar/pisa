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

def extract_answer(text):
    """Extracts the answer from \boxed{...}"""
    matches = re.findall(r'\\boxed\s*\{(.*?)\}', text)
    if matches:
        return matches[-1].strip()
    return None

def power_sampling_global_algorithm_1(llm, tokenizer, prompt, temperature, alpha,
                                      block_size=256, num_blocks=16, mcmc_steps=10):

    # 1. Setup & Encoding
    # We strip the system prompt from the 'active' generation buffer
    # to prevent the model from optimizing the prompt itself.
    prompt_ids = tokenizer.encode(prompt)
    prompt_len = len(prompt_ids)

    # 'curr_ids' will track the FULL sequence (Prompt + Gen)
    curr_ids = list(prompt_ids)

    # We only track logprobs for the GENERATED part
    curr_logprobs = []

    # Force the model to think before boxing
    # If the output is too short, we reject it.
    MIN_REASONING_TOKENS = 100

    for k in range(num_blocks):

        # --- A. INITIALIZATION (Draft Phase) ---
        prefix_str = tokenizer.decode(curr_ids)

        # Calculate how much "new" text we have generated so far
        gen_len_so_far = len(curr_ids) - prompt_len

        # Adaptive Stopping: If we already have a box, stop adding blocks.
        if "\\boxed" in prefix_str[len(prompt):]:
            break

        params = SamplingParams(
            temperature=temperature,
            min_tokens=block_size,
            max_tokens=block_size,
            top_p=0.95, logprobs=1,
            ignore_eos=True,
            # PENALTY: Prevent repeating the system prompt
            repetition_penalty=1.05
        )

        out = llm.generate(prefix_str, params, use_tqdm=False)
        new_ids = list(out[0].outputs[0].token_ids)

        # Extract Logprobs
        new_logprobs = []
        for i, token_id in enumerate(new_ids):
            if i < len(out[0].outputs[0].logprobs):
                new_logprobs.append(out[0].outputs[0].logprobs[i][token_id].logprob)
            else:
                new_logprobs.append(-100.0)

        curr_ids.extend(new_ids)
        curr_logprobs.extend(new_logprobs)

        # Total generated length (excluding prompt)
        total_gen_len = len(curr_logprobs)

        # --- B. MCMC REFINEMENT (Global) ---
        for _ in range(mcmc_steps):
            if total_gen_len < 5: break # Too short to mutate

            # 1. Global Slice (Randomly pick start point in generated text)
            m = random.randint(0, total_gen_len - 1)
            regen_len = total_gen_len - m

            # 2. Construct Proposal
            # Keep prompt + kept_gen
            kept_ids = curr_ids[:(prompt_len + m)]
            kept_str = tokenizer.decode(kept_ids)

            prop_params = SamplingParams(
                temperature=temperature,
                min_tokens=regen_len,
                max_tokens=regen_len, # Exact length match
                top_p=0.95, logprobs=1,
                ignore_eos=True
            )

            prop_out = llm.generate(kept_str, prop_params, use_tqdm=False)
            prop_ids = list(prop_out[0].outputs[0].token_ids)

            # 3. Safety Check: Did we just generate empty text or garbage?
            if len(prop_ids) != regen_len:
                continue # Skip this mutation if length mismatch

            prop_scores = []
            for i, token_id in enumerate(prop_ids):
                if i < len(prop_out[0].outputs[0].logprobs):
                    prop_scores.append(prop_out[0].outputs[0].logprobs[i][token_id].logprob)
                else:
                    prop_scores.append(-100.0)

            # 4. Acceptance Ratio
            new_sum = sum(prop_scores)
            old_sum = sum(curr_logprobs[m:])

            log_ratio = alpha * (new_sum - old_sum)

            if log_ratio > 0 or random.random() < math.exp(log_ratio):
                # ACCEPT
                curr_ids = curr_ids[:(prompt_len + m)] + prop_ids
                curr_logprobs = curr_logprobs[:m] + prop_scores

        # --- C. EXIT CHECKS ---
        current_text = tokenizer.decode(curr_ids)

        # 1. Check if the prompt leaked into the answer (naive check)
        # If the generated text starts with "System:" or similar, we might want to kill it.
        # But usually, just checking for the BOX is enough.

        # 2. Valid Exit: Box found AND we reasoned enough
        generated_text_only = current_text[len(prompt):]
        if "\\boxed" in generated_text_only and "}" in generated_text_only:
             # Ensure we didn't just output "\boxed{0}" instantly
             if len(curr_logprobs) > MIN_REASONING_TOKENS:
                 break

    return tokenizer.decode(curr_ids)

# ==========================================
# EXECUTION HARNESS (Progress & Storage)
# ==========================================

def run_experiment(llm, tokenizer, problems, answers):

    # Configuration (Matches Paper/Screenshot)
    config = {
        "temperature": 0.25,
        "alpha": 4.0,
        "block_size": 256,
        "num_blocks": 16,
        "mcmc_steps": 10
    }

    print(f"\nStarting Global Algorithm 1 on {len(problems)} problems...")
    print(f"Config: {config}\n")

    # Storage Array
    detailed_results = []
    correct_count = 0

    # Progress Bar with TQDM
    pbar = tqdm(enumerate(problems), total=len(problems), desc="Processing")

    for i, problem in pbar:
        prompt = SYSTEM_PROMPT + problem

        try:
            # Run the heavy Algorithm
            final_text = power_sampling_global_algorithm_1(
                llm, tokenizer, prompt,
                temperature=config["temperature"],
                alpha=config["alpha"],
                block_size=config["block_size"],
                num_blocks=config["num_blocks"],
                mcmc_steps=config["mcmc_steps"]
            )

            # Extract & Compare
            predicted_ans = extract_answer(final_text)
            correct_ans = answers[i]
            is_correct = (predicted_ans == correct_ans)

            if is_correct:
                correct_count += 1

            # Live Update in Console
            # We use tqdm.write so it doesn't break the progress bar layout
            status_icon = "✅" if is_correct else "❌"
            tqdm.write(f"[Prob {i}] {status_icon} | Pred: {predicted_ans} | True: {correct_ans}")

            # Save Data
            detailed_results.append({
                "problem_id": i,
                "prompt": problem,
                "full_output": final_text,
                "predicted": predicted_ans,
                "truth": correct_ans,
                "is_correct": is_correct
            })

            # Update Progress Bar Description with current Accuracy
            current_acc = (correct_count / (i + 1)) * 100
            pbar.set_description(f"Acc: {current_acc:.1f}%")

        except Exception as e:
            tqdm.write(f"[Prob {i}] CRASHED: {e}")
            detailed_results.append({
                "problem_id": i,
                "error": str(e),
                "is_correct": False
            })

    # Final Summary
    final_score = (correct_count / len(problems)) * 100
    print(f"\n>>> FINAL SCORE: {final_score:.2f}% ({correct_count}/{len(problems)})")

    return detailed_results

# ==========================================
# RUN AND QUERY
# ==========================================

# Run the experiment
results_array = run_experiment(llm, tokenizer, problems, answers)

# Example of how to query the array later:
# print(results_array[0]['full_output'])  # See reasoning for Problem 0
# df = pd.DataFrame(results_array)        # Convert to DataFrame for analysis
