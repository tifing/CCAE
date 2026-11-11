import argparse
import json
import os
import re
import time

import numpy as np
from tqdm import tqdm

CHOICES = list("ABCDEFGHIJKLMNOP")

MMLU_PRO_CATEGORIES = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "Humanities": ["history", "philosophy", "law"],
    "Social Sciences": ["economics", "psychology"],
    "Other": ["other", "business", "health"],
}

PROMPT_TEMPLATES = {
    "qwen3-train": {
        "system": (
            "You are a helpful AI Assistant, designed to provided well-reasoned "
            "and detailed responses. You FIRST think about the reasoning "
            "process step by step and then provide the user with the answer. Please "
            "enclose your final answer in the box: \\boxed{Your Answer}. Please "
            "stop generation immediately after outputing the box."
        ),
        "template": (
            "<|im_start|>system\n{system}<|im_end|>\n"
            "<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "stop_token_ids": [151645, 151643],
    },
    "qwen-base": {
        "system": "Please reason step by step, and put your final answer within \\boxed{}.",
        "template": (
            "<|im_start|>system\n{system}<|im_end|>\n"
            "<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "stop_token_ids": [151645, 151643],
    },
}


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_mmlu_pro(data_dir, split="test"):
    local_path = os.path.join(data_dir, "mmlu_pro", f"{split}.jsonl") if data_dir else None
    if local_path and os.path.exists(local_path):
        print(f"Loading MMLU-Pro from local: {local_path}")
        return _read_jsonl(local_path)

    print("Loading MMLU-Pro from HuggingFace (TIGER-Lab/MMLU-Pro)...")
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    examples = []
    for i, item in enumerate(ds):
        examples.append(
            {
                "idx": i,
                "question": item["question"],
                "options": item["options"],
                "answer": item["answer"],
                "answer_index": item["answer_index"],
                "category": item.get("category", ""),
                "type": item.get("category", "mmlu_pro"),
            }
        )

    if local_path:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Cached MMLU-Pro to {local_path}")

    return examples


def load_gpqa_diamond(data_dir):
    local_path = os.path.join(data_dir, "gpqa_diamond", "test.jsonl") if data_dir else None
    if not local_path or not os.path.exists(local_path):
        raise FileNotFoundError(
            "GPQA-Diamond data not found. Set --data_dir so that "
            "`gpqa_diamond/test.jsonl` exists."
        )

    print(f"Loading GPQA-Diamond from: {local_path}")
    examples = _read_jsonl(local_path)
    for i, ex in enumerate(examples):
        if "idx" not in ex:
            ex["idx"] = i
        if "type" not in ex:
            ex["type"] = "gpqa_diamond"
    return examples


def _parse_gpqa_main_options(problem_text):
    suffix = r"Please write your final answer in the form of \boxed{A}, \boxed{B}, \boxed{C}, or \boxed{D}"
    text = problem_text.rstrip()
    if text.endswith(suffix):
        text = text[: -len(suffix)].rstrip()

    pattern = re.compile(r"(?:^|\n)\(([A-D])\)\s", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) < 4:
        matches = list(re.finditer(r"(?<!\w)\(([A-D])\)\s", text))
    if len(matches) < 4:
        return None, None

    abcd_matches = {}
    for match in matches:
        letter = match.group(1)
        if letter in ("A", "B", "C", "D") and letter not in abcd_matches:
            abcd_matches[letter] = match
    if len(abcd_matches) != 4:
        return None, None

    sorted_matches = [abcd_matches[c] for c in "ABCD"]
    question = text[: sorted_matches[0].start()].strip()
    options = []
    for i, match in enumerate(sorted_matches):
        start = match.end()
        end = sorted_matches[i + 1].start() if i < 3 else len(text)
        options.append(text[start:end].strip())
    return question, options


def _load_gpqa_main_from_arrow(arrow_root):
    import pyarrow.ipc as ipc

    for root, _, files in os.walk(arrow_root):
        for file_name in files:
            if file_name.endswith(".arrow"):
                arrow_path = os.path.join(root, file_name)
                print(f"Loading GPQA-Main arrow: {arrow_path}")
                reader = ipc.open_stream(arrow_path)
                table = reader.read_all()
                return (
                    table.column("problem").to_pylist(),
                    table.column("solution").to_pylist(),
                    table.column("domain").to_pylist(),
                )
    return None


def load_gpqa_main(data_dir):
    cache_path = os.path.join(data_dir, "gpqa_main", "test.jsonl") if data_dir else None
    if cache_path and os.path.exists(cache_path):
        print(f"Loading GPQA-Main from local cache: {cache_path}")
        examples = _read_jsonl(cache_path)
        print(f"  Loaded {len(examples)} examples")
        return examples

    problems = solutions = domains = None
    arrow_root = os.path.join(data_dir, "gpqa_main", "hendrydong___gpqa_main_mc") if data_dir else None
    if arrow_root and os.path.isdir(arrow_root):
        loaded = _load_gpqa_main_from_arrow(arrow_root)
        if loaded is not None:
            problems, solutions, domains = loaded

    if problems is None:
        print("Arrow not found, trying HuggingFace hendrydong/gpqa_main_mc...")
        from datasets import load_dataset

        ds = load_dataset("hendrydong/gpqa_main_mc", split="test")
        problems = ds["problem"]
        solutions = ds["solution"]
        domains = ds["domain"]

    print(f"  Raw GPQA-Main: {len(problems)} questions")
    examples = []
    skipped = 0
    for i in range(len(problems)):
        answer_match = re.search(r"\\boxed\{([A-D])\}", solutions[i])
        if not answer_match:
            skipped += 1
            continue

        question, options = _parse_gpqa_main_options(problems[i])
        if question is None:
            skipped += 1
            continue

        gt_idx = ord(answer_match.group(1)) - ord("A")
        examples.append(
            {
                "idx": len(examples),
                "question": question,
                "choices": options,
                "answer": gt_idx,
                "category": domains[i],
                "type": "gpqa_main",
            }
        )

    print(f"  Parsed: {len(examples)} examples (skipped {skipped} due to parse failure)")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Cached to {cache_path}")

    return examples


def format_question_mmlu_pro(example):
    """Each option on its own line: 'A. opt'"""
    options = example.get("options", example.get("choices", []))
    lines = [f"{CHOICES[i]}. {str(opt).strip()}" for i, opt in enumerate(options) if i < len(CHOICES)]
    return example["question"].strip() + "\n" + "\n".join(lines)


def format_question_gpqa(example):
    """Each option on its own line: 'A. opt'"""
    lines = [f"{CHOICES[i]}. {str(opt).strip()}" for i, opt in enumerate(example["choices"])]
    return example["question"].strip() + "\n" + "\n".join(lines)


def build_prompt(question_text, prompt_cfg, data_name):
    hint = "(the correct letter choice from A to P)" if data_name == "mmlu_pro" else "(A, B, C, or D)"
    user_content = (
        f"Question: {question_text}\n"
        f"Reason step by step and put the answer {hint} in \\boxed{{}}."
    )
    return prompt_cfg["template"].format(system=prompt_cfg["system"], question=user_content)


def normalize_final_answer(final_answer):
    final_answer = str(final_answer)
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", r"$\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", r"\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", final_answer)
    return final_answer


def extract_prediction_boxed(output_text):
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse

    parsed = parse(
        output_text,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=False,
                    equations=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )
    if len(parsed) == 0:
        return ""
    return normalize_final_answer(str(parsed[-1]))


def check_correct_mmlu_pro(prediction, example):
    gt_letter = example["answer"]
    if prediction == gt_letter:
        return True
    gt_idx = ord(gt_letter) - ord("A")
    if 0 <= gt_idx < len(example["options"]):
        content = example["options"][gt_idx]
        return prediction == f"{gt_letter}. {content}"
    return False


def check_correct_gpqa(prediction, example):
    gt_idx = example["answer"]
    gt_letter = CHOICES[gt_idx]
    if prediction == gt_letter:
        return True
    if 0 <= gt_idx < len(example["choices"]):
        content = example["choices"][gt_idx]
        return prediction == f"{gt_letter}. {content}"
    return False


def get_gt_letter(example, data_name):
    return example["answer"] if data_name == "mmlu_pro" else CHOICES[example["answer"]]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MMLU-Pro / GPQA with boxed-answer extraction")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True, choices=["mmlu_pro", "gpqa_diamond", "gpqa_main"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--prompt_type", type=str, default="qwen3-train", choices=list(PROMPT_TEMPLATES.keys()))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_test_sample", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--n_sampling", type=int, default=1)
    parser.add_argument("--max_tokens_per_call", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.temperature == 0:
        args.top_p = 1
    return args


def load_examples(args):
    if args.data_name == "mmlu_pro":
        examples = load_mmlu_pro(args.data_dir, args.split)
        format_fn = format_question_mmlu_pro
        check_fn = check_correct_mmlu_pro
    elif args.data_name == "gpqa_main":
        examples = load_gpqa_main(args.data_dir)
        format_fn = format_question_gpqa
        check_fn = check_correct_gpqa
    else:
        examples = load_gpqa_diamond(args.data_dir)
        format_fn = format_question_gpqa
        check_fn = check_correct_gpqa
    return examples, format_fn, check_fn


def build_output_paths(args):
    output_dir = args.output_dir
    if not os.path.isabs(output_dir) and not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    temp_str = f"{args.temperature:g}"
    out_prefix = (
        f"test_{args.prompt_type}"
        f"_{args.num_test_sample}_seed{args.seed}"
        f"_t{temp_str}_s{args.start}_e{args.end}"
    )
    out_dir = os.path.join(output_dir, args.data_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{out_prefix}.jsonl")
    metrics_file = os.path.join(out_dir, f"{out_prefix}_metrics.json")
    return out_file, metrics_file


def load_completed(out_file, overwrite):
    completed = {}
    if overwrite and os.path.exists(out_file):
        os.remove(out_file)
        return completed
    if not os.path.exists(out_file):
        return completed

    with open(out_file) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "score" in record and record["score"]:
                completed[record["idx"]] = record
    if completed:
        print(f"Resume: found {len(completed)} completed samples in {out_file}")
    return completed


def prepare_samples(args, examples, format_fn, completed):
    samples = []
    prompts = []
    pending_indices = []
    for example in examples:
        question_text = format_fn(example)
        gt_letter = get_gt_letter(example, args.data_name)

        if example["idx"] in completed:
            samples.append(completed[example["idx"]])
            prompts.append(None)
            continue

        sample = {
            "idx": example["idx"],
            "question": question_text,
            "gt_cot": "None",
            "gt": gt_letter,
            "type": example.get("type", args.data_name),
        }
        choices = example.get("options", example.get("choices"))
        if choices is not None:
            sample["choices"] = choices
        if "answer" in example:
            sample["answer"] = example["answer"]
        if "category" in example:
            sample["category"] = example["category"]

        samples.append(sample)
        prompts.append(build_prompt(question_text, PROMPT_TEMPLATES[args.prompt_type], args.data_name))
        pending_indices.append(len(samples) - 1)

    return samples, prompts, pending_indices


def run_generation(args, prompt_cfg, pending_prompts):
    from vllm import LLM, SamplingParams

    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    tp = len(available_gpus) // args.pipeline_parallel_size
    llm_kwargs = {
        "model": args.model_name_or_path,
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    print("Loading model...")
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens_per_call,
        n=args.n_sampling,
        stop_token_ids=prompt_cfg.get("stop_token_ids"),
    )

    print(
        f"Generating {len(pending_prompts)} prompts × {args.n_sampling} samples = "
        f"{len(pending_prompts) * args.n_sampling} completions..."
    )
    start_time = time.time()
    generations = llm.generate(pending_prompts, sampling_params)
    generations = sorted(generations, key=lambda x: int(x.request_id))
    return [[o.text for o in g.outputs] for g in generations], time.time() - start_time


def compute_summary(args, samples, gen_time):
    score_mat = [s["score"] for s in samples]
    pass1 = np.mean([np.mean(s) for s in score_mat]) * 100
    pass_at_k = np.mean([1.0 if any(s) else 0.0 for s in score_mat]) * 100

    subject_scores = {}
    subject_scores_all = {}
    for sample in samples:
        key = sample.get("category", sample.get("type", "unknown"))
        subject_scores.setdefault(key, []).append(float(np.mean(sample["score"])))
        subject_scores_all.setdefault(key, []).append(sample["score"])

    subject_acc = {k: round(np.mean(v) * 100, 2) for k, v in sorted(subject_scores.items())}
    subject_pass_k = {
        k: round(np.mean([1.0 if any(s) else 0.0 for s in v]) * 100, 2)
        for k, v in sorted(subject_scores_all.items())
    }

    cat_acc = {}
    cat_pass_k = {}
    if args.data_name == "mmlu_pro":
        subj_to_cat = {subj: cat for cat, subjs in MMLU_PRO_CATEGORIES.items() for subj in subjs}
        cat_scores = {}
        cat_scores_all = {}
        for subj, scores in subject_scores.items():
            cat = subj_to_cat.get(subj, "Other")
            cat_scores.setdefault(cat, []).extend(scores)
            cat_scores_all.setdefault(cat, []).extend(subject_scores_all[subj])
        cat_acc = {k: round(np.mean(v) * 100, 2) for k, v in sorted(cat_scores.items())}
        cat_pass_k = {
            k: round(np.mean([1.0 if any(s) else 0.0 for s in v]) * 100, 2)
            for k, v in sorted(cat_scores_all.items())
        }

    result_json = {
        "num_samples": len(samples),
        "acc": pass1,
        f"pass@{args.n_sampling}": pass_at_k,
        "empty_samples": len([s for s in samples if not s["pred"][-1]]),
        "subject_acc": subject_acc,
        "subject_pass_k": subject_pass_k,
        "cat_acc": cat_acc,
        "cat_pass_k": cat_pass_k,
        "time_use_in_second": gen_time,
        "args": {
            "model": args.model_name_or_path,
            "data_name": args.data_name,
            "prompt_type": args.prompt_type,
            "temperature": args.temperature,
            "n_sampling": args.n_sampling,
            "max_tokens": args.max_tokens_per_call,
        },
    }
    return result_json


def print_summary(args, samples, result_json):
    print(f"\n{'=' * 60}")
    print(f"Dataset: {args.data_name}")
    print(f"Model: {args.model_name_or_path}")
    print(f"Prompt: {args.prompt_type}")
    print(f"Pass@1: {result_json['acc']:.2f}%")
    if args.n_sampling > 1:
        print(f"Pass@{args.n_sampling}: {result_json[f'pass@{args.n_sampling}']:.2f}%")
    print(f"{'=' * 60}")

    print("\n--- Per-Subject ---")
    for subj in sorted(result_json["subject_acc"].keys()):
        n = sum(1 for sample in samples if sample.get("category", sample.get("type", "unknown")) == subj)
        p1 = result_json["subject_acc"][subj]
        pk = result_json["subject_pass_k"][subj]
        correct = int(p1 * n / 100)
        if args.n_sampling > 1:
            print(f"  {subj}: Pass@1={p1:.2f}% Pass@{args.n_sampling}={pk:.2f}% ({correct}/{n})")
        else:
            print(f"  {subj}: {p1:.2f}% ({correct}/{n})")

    if args.data_name == "mmlu_pro":
        print("\n--- Per-Category ---")
        for cat in ["STEM", "Humanities", "Social Sciences", "Other"]:
            if cat not in result_json["cat_acc"]:
                continue
            p1 = result_json["cat_acc"][cat]
            pk = result_json["cat_pass_k"][cat]
            if args.n_sampling > 1:
                print(f"  {cat}: Pass@1={p1:.2f}% Pass@{args.n_sampling}={pk:.2f}%")
            else:
                print(f"  {cat}: {p1:.2f}%")

    print("\n--- Overall ---")
    print(f"  Pass@1: {result_json['acc']:.2f}% (n={len(samples)})")
    if args.n_sampling > 1:
        print(f"  Pass@{args.n_sampling}: {result_json[f'pass@{args.n_sampling}']:.2f}%")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    prompt_cfg = PROMPT_TEMPLATES[args.prompt_type]

    examples, format_fn, check_fn = load_examples(args)
    if args.subjects:
        norm_subjects = [s.replace("_", " ") for s in args.subjects]
        before = len(examples)
        examples = [e for e in examples if e.get("category", e.get("type", "")) in norm_subjects]
        print(f"Filtered by subjects {norm_subjects}: {before} -> {len(examples)}")

    if args.num_test_sample > 0:
        examples = examples[: args.num_test_sample]
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]
    print(f"Dataset: {args.data_name}, samples: {len(examples)}")

    out_file, metrics_file = build_output_paths(args)
    completed = load_completed(out_file, args.overwrite)
    samples, prompts, pending_indices = prepare_samples(args, examples, format_fn, completed)
    pending_prompts = [prompts[i] for i in pending_indices]

    if not pending_prompts:
        print(f"All {len(samples)} samples already completed, skipping generation.")
        gen_time = 0.0
    else:
        print(f"Total: {len(samples)}, completed: {len(completed)}, pending: {len(pending_prompts)}")
        print("=" * 60)
        print("Example prompt (first pending sample):")
        print(pending_prompts[0][:1500])
        print("..." if len(pending_prompts[0]) > 1500 else "")
        print("=" * 60)

        grouped_outputs, gen_time = run_generation(args, prompt_cfg, pending_prompts)
        print(f"Generation took {gen_time:.1f}s")
        print("Extracting predictions...")
        for gen_i, sample_i in enumerate(tqdm(pending_indices, desc="Scoring")):
            outputs = grouped_outputs[gen_i]
            preds = []
            scores = []
            for output in outputs:
                pred = extract_prediction_boxed(output)
                preds.append(pred)
                scores.append(1.0 if check_fn(pred, examples[sample_i]) else 0.0)

            samples[sample_i]["code"] = outputs
            samples[sample_i]["pred"] = preds
            samples[sample_i]["score"] = scores

            if args.save_outputs:
                with open(out_file, "a") as f:
                    f.write(json.dumps(samples[sample_i], ensure_ascii=False) + "\n")

    result_json = compute_summary(args, samples, gen_time)
    print_summary(args, samples, result_json)

    if args.save_outputs:
        with open(out_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"\nResults saved to {out_file}")

    with open(metrics_file, "w") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
