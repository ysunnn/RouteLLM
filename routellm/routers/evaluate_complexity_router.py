#!/usr/bin/env python3
"""
evaluate_complexity_router.py

Evaluates the `complexity_AZ_router` HTTP endpoint on the GSM8K benchmark.
Sends each question via a `router-complexity_AZ_router-{threshold}` model name,
records accuracy and percent strong-model calls, and writes results to CSV.
"""
import re
import json
import csv
import argparse
import requests
from tqdm import tqdm
from datasets import load_dataset


def extract_answer(text: str) -> float:
    """
    Extract the first integer or floating-point number from the model's response text.
    Returns float or raises ValueError if none found.
    """
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        raise ValueError(f"No numeric answer found in response: {text}")
    return float(match.group(0))


def evaluate_router(
    threshold: float,
    strong_model: str,
    weak_model: str,
    endpoint_url: str,
    subset: int = None
) -> (float, float):
    """
    Runs the GSM8K evaluate over the specified threshold.
    Returns (accuracy, strong_call_percentage).
    """
    # Load the train split of GSM8K
    ds = load_dataset("gsm8k", "main", split="train")
    if subset:
        ds = ds.select(range(subset))

    total = len(ds)
    correct = 0
    strong_calls = 0

    for ex in tqdm(ds, desc=f"Threshold {threshold:.2f}"):
        question = ex["question"].strip()
        try:
            expected = float(ex["answer"].strip())
        except Exception:
            # skip if answer is not parseable
            total -= 1
            continue

        payload = {
            "model": f"router-complexity_AZ_router-{threshold}",
            "messages": [{"role": "user", "content": question}]
        }
        headers = {"Content-Type": "application/json"}

        resp = requests.post(endpoint_url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        # determine which model was actually used
        model_used = data.get("model") or data.get("choices", [{}])[0].get("finish_reason")
        if model_used == strong_model:
            strong_calls += 1

        # extract content
        content = data["choices"][0]["message"]["content"]
        try:
            answer = extract_answer(content)
            if answer == expected:
                correct += 1
        except ValueError:
            # treat missing parse as incorrect
            pass

    accuracy = correct / total if total else 0.0
    strong_pct = strong_calls / total if total else 0.0
    return accuracy, strong_pct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate complexity_AZ_router on GSM8K via HTTP endpoint"
    )
    parser.add_argument(
        "--endpoint", type=str,
        default="http://localhost:6060/v1/chat/completions",
        help="URL of the chat completions endpoint"
    )
    parser.add_argument(
        "--strong-model", type=str,
        default="gpt-4o20240806",
        help="Name of the strong model"  
    )
    parser.add_argument(
        "--weak-model", type=str,
        default="gpt-3.5-turbo",
        help="Name of the weak model"
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[i/10 for i in range(1, 10)],
        help="Thresholds to evaluate (e.g. 0.1 0.2 ... 0.9)"
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Optional: only use first N examples"
    )
    parser.add_argument(
        "--output", type=str, default="gsm8k_complexity_eval.csv",
        help="CSV file for saving results"
    )
    args = parser.parse_args()

    print("Starting evaluation of complexity_AZ_router on GSM8K")
    print("Endpoint:", args.endpoint)
    print("Strong model:", args.strong_model)
    print("Weak model:", args.weak_model)
    print("Thresholds:", args.thresholds)
    if args.subset:
        print(f"Using first {args.subset} examples")

    results = []
    for t in args.thresholds:
        acc, pct = evaluate_router(
            t, args.strong_model, args.weak_model,
            args.endpoint, args.subset
        )
        print(f"Threshold {t:.2f} -> Accuracy: {acc:.3f}, Strong calls: {pct:.1%}")
        results.append((t, acc, pct))

    # write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "accuracy", "strong_call_pct"]);
        for row in results:
            writer.writerow(row)
    print("Results saved to", args.output)
