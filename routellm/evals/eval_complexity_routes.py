#!/usr/bin/env python3
"""
evaluate_server_routers.py

Evaluates a single complexity router via HTTP on the first N examples of GSM8K,
collecting separate prompt and completion token counts for strong/weak models,
estimating costs (with separate input/output million‐token rates), and plotting results.
Outputs a CSV and a bar-plot.
"""
import argparse
import requests
import csv
import os
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd


def evaluate_router(
    endpoint: str,
    router_name: str,
    threshold: float,
    strong_model: str,
    weak_model: str,
    subset: int,
) -> dict:
    """
    Sends the first `subset` GSM8K questions through
    `router-<router_name>-<threshold>` and aggregates stats.
    Returns a dict of metrics.
    """
    ds = load_dataset("gsm8k", "main", split="train")
    if subset:
        ds = ds.select(range(subset))

    total = len(ds)
    strong_calls = weak_calls = 0
    strong_prompt = strong_completion = 0
    weak_prompt = weak_completion = 0

    for ex in ds:
        question = ex["question"].strip()
        payload = {
            "model": f"router-{router_name}-{threshold}",
            "messages": [{"role": "user", "content": question}]
        }
        headers = {"Content-Type": "application/json"}

        resp = requests.post(endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)

        served_model = data.get("model", "")
        if strong_model in served_model:
            strong_calls += 1
            strong_prompt += pt
            strong_completion += ct
        else:
            weak_calls += 1
            weak_prompt += pt
            weak_completion += ct

    return {
        "router": router_name,
        "threshold": threshold,
        "total_calls": total,
        "strong_calls": strong_calls,
        "weak_calls": weak_calls,
        "pct_strong": strong_calls / total if total else 0,
        "strong_prompt_tokens": strong_prompt,
        "strong_completion_tokens": strong_completion,
        "weak_prompt_tokens": weak_prompt,
        "weak_completion_tokens": weak_completion,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a single complexity router on GSM8K subset"
    )
    parser.add_argument(
        "--endpoint", required=True,
        help="Endpoint URL for chat completions"
    )
    parser.add_argument(
        "--router", required=True,
        choices=[
            "complexity_AZ_router",
            "complexity_self_answer_router",
            "complexity_evaluation_router",
        ],
        help="Name of the router to evaluate"
    )
    parser.add_argument(
        "--threshold", type=float, required=True,
        help="Routing threshold to use (0.0–1.0)"
    )
    parser.add_argument(
        "--strong-model", required=True,
        help="Substring of the strong model name (e.g. 'gpt-4.1-nano')"
    )
    parser.add_argument(
        "--weak-model", required=True,
        help="Substring of the weak model name (e.g. 'gpt-3.5-turbo')"
    )
    parser.add_argument(
        "--subset", type=int, default=30,
        help="Number of GSM8K examples to test"
    )
    parser.add_argument(
        "--cost-strong-input-per-million", type=float, default=30.0,
        help="USD per 1M prompt tokens (strong model)"
    )
    parser.add_argument(
        "--cost-strong-output-per-million", type=float, default=30.0,
        help="USD per 1M completion tokens (strong model)"
    )
    parser.add_argument(
        "--cost-weak-input-per-million", type=float, default=2.0,
        help="USD per 1M prompt tokens (weak model)"
    )
    parser.add_argument(
        "--cost-weak-output-per-million", type=float, default=2.0,
        help="USD per 1M completion tokens (weak model)"
    )
    parser.add_argument(
        "--output-csv", default="router_eval.csv",
        help="CSV file to write results"
    )
    parser.add_argument(
        "--output-plot", default="router_eval_plot.png",
        help="Filename for saved bar-plot"
    )
    args = parser.parse_args()

    # Evaluate
    stats = evaluate_router(
        args.endpoint,
        args.router,
        args.threshold,
        args.strong_model,
        args.weak_model,
        args.subset
    )

    # Compute cost breakdown (rates per million tokens)
    cs_in = stats["strong_prompt_tokens"] / 1_000_000 * args.cost_strong_input_per_million
    cs_out = stats["strong_completion_tokens"] / 1_000_000 * args.cost_strong_output_per_million
    cw_in = stats["weak_prompt_tokens"] / 1_000_000 * args.cost_weak_input_per_million
    cw_out= stats["weak_completion_tokens"] / 1_000_000 * args.cost_weak_output_per_million
    stats.update({
        "cost_strong_prompt_usd": cs_in,
        "cost_strong_completion_usd": cs_out,
        "cost_weak_prompt_usd": cw_in,
        "cost_weak_completion_usd": cw_out,
        "cost_total_usd": cs_in + cs_out + cw_in + cw_out
    })

    # Write CSV
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
        writer.writeheader()
        writer.writerow(stats)
    print(f"Results saved to {args.output_csv}")

    # Plot
    df = pd.DataFrame([stats]).set_index('router')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    df[[
        'strong_prompt_tokens', 'strong_completion_tokens',
        'weak_prompt_tokens', 'weak_completion_tokens'
    ]].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Token Usage (per router)')
    axes[0].set_ylabel('Tokens')

    df[[
        'cost_strong_prompt_usd', 'cost_strong_completion_usd',
        'cost_weak_prompt_usd', 'cost_weak_completion_usd'
    ]].plot(kind='bar', ax=axes[1])
    axes[1].set_title('Cost Breakdown (USD per million tokens)')
    axes[1].set_ylabel('USD')

    plt.tight_layout()
    plt.savefig(args.output_plot, bbox_inches='tight')
    print(f"Plot saved to {args.output_plot}")
