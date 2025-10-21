"""Build a reference embedding set from non-hallucinated samples."""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch
from tqdm import tqdm

from VL_Uncertainty import (
    BENCHMARK_MAP,
    BENCHMARK_TYPE,
    LLM_MAP,
    LVLM_MAP,
    obtain_benchmark,
    obtain_llm,
    obtain_lvlm,
    fix_seed,
)
from util.reference_set import save_reference_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reference embeddings for geometric detection.")
    parser.add_argument("--lvlm", type=str, default="Qwen2-VL-2B-Instruct")
    parser.add_argument("--benchmark", type=str, default="MMVet")
    parser.add_argument("--llm", type=str, default="Qwen2.5-3B-Instruct")
    parser.add_argument("--output", type=str, required=True, help="Path to save the reference set (e.g., reference/qwen2.pt)")
    parser.add_argument("--max_samples", type=int, default=200, help="Maximum number of dataset examples to inspect.")
    parser.add_argument("--max_reference", type=int, default=128, help="Maximum number of correct samples to store.")
    parser.add_argument("--inference_temp", type=float, default=0.1)
    parser.add_argument("--progress_bar", action="store_true", help="Display a tqdm progress bar.")
    return parser.parse_args()


def verify_answer(sample: Dict, ans: str, benchmark_name: str, llm) -> bool:
    if BENCHMARK_TYPE[benchmark_name] == "MULTI_CHOICE":
        return str(sample["gt_ans"]) in ans
    question = (
        f"Ground truth: {sample['gt_ans']}. Model answer: {ans}. "
        "Please verify if the model ans matches the ground truth. Respond with either 'Correct' or 'Wrong' only."
    )
    llm_response = llm.generate(question, 0.1)
    return any(token in llm_response for token in ["Correct", "correct", "C", "c"])


def main() -> None:
    args = parse_args()
    if args.lvlm not in LVLM_MAP:
        raise ValueError(f"Unsupported LVLM: {args.lvlm}")
    if args.benchmark not in BENCHMARK_MAP:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")
    if args.llm not in LLM_MAP:
        raise ValueError(f"Unsupported LLM: {args.llm}")

    fix_seed(0)
    lvlm = obtain_lvlm(args)
    benchmark = obtain_benchmark(args)
    llm = obtain_llm(args)

    reference_embeddings: List[torch.Tensor] = []
    reference_metadata: List[Dict] = []

    iterable = range(min(args.max_samples, benchmark.obtain_size()))
    iterator = tqdm(iterable, desc="Harvesting") if args.progress_bar else iterable

    for idx in iterator:
        sample = benchmark.retrieve(idx)
        if sample is None or sample.get("img") is None or sample.get("question") is None:
            continue
        try:
            embedding = lvlm.encode_prompt(sample["img"], sample["question"])  # type: ignore[attr-defined]
        except NotImplementedError:
            raise RuntimeError(
                f"LVLM {args.lvlm} does not expose prompt embeddings. Cannot build reference set."
            )
        answer = lvlm.generate(sample["img"], sample["question"], args.inference_temp)
        if not verify_answer(sample, answer, args.benchmark, llm):
            continue
        reference_embeddings.append(embedding.squeeze(0))
        reference_metadata.append(
            {
                "idx": idx,
                "question": sample["question"],
                "gt_ans": sample["gt_ans"],
                "answer": answer,
            }
        )
        if len(reference_embeddings) >= args.max_reference:
            break

    if not reference_embeddings:
        raise RuntimeError("No valid reference samples collected. Check LVLM/benchmark configuration.")

    stacked = torch.stack(reference_embeddings, dim=0)
    config = {
        "lvlm": args.lvlm,
        "benchmark": args.benchmark,
        "llm": args.llm,
        "inference_temp": args.inference_temp,
        "max_reference": args.max_reference,
    }
    save_reference_set(args.output, stacked, reference_metadata, config)
    summary = {
        "saved": len(reference_embeddings),
        "output": args.output,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
