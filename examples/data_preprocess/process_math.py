import argparse
import os
from functools import partial

from datasets import load_dataset

from verl.utils.hdfs_io import copy, makedirs


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. "
    "You FIRST think about the reasoning process step by step and then provide the user with the answer. "
    "Please enclose your final answer in the box: \\boxed{Your Answer}. "
    "Please stop generation immediately after outputing the box."
)


def extract_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


def example_map_fn(example, idx, data_source, split, system_prompt=None):
    ground_truth = example["solution"]

    boxed_content = extract_boxed(ground_truth)
    if boxed_content is not None:
        ground_truth = boxed_content

    prompt = []
    if system_prompt:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": example["problem"]})

    data = {
        "prompt": prompt,
        "data_source": data_source,
        "reward_model": {"ground_truth": ground_truth},
    }
    return data


def build_math_lighteval_dataset(data_dir, split="train", system_prompt=None):
    parquet_file = os.path.join(data_dir, f"{split}-00000-of-00001.parquet")
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

    data_source = "lighteval/MATH"
    print(f"Loading {data_source} dataset from {parquet_file}...", flush=True)

    dataset = load_dataset("parquet", data_files=parquet_file)["train"]

    map_fn = partial(example_map_fn, data_source=data_source, split=split, system_prompt=system_prompt)
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MATH-lighteval dataset for verl")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/MATH/raw",
        help="Input directory containing train-00000-of-00001.parquet and test-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/MATH",
        help="Output directory for processed parquet files",
    )
    parser.add_argument("--hdfs_dir", type=str, default=None, help="HDFS directory (optional)")
    parser.add_argument(
        "--splits",
        type=str,
        default="train,test",
        help="Comma-separated list of splits to process (default: train,test)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="System prompt to embed in the data (optional). If not provided, uses default.",
    )

    args = parser.parse_args()

    if args.system_prompt is None:
        args.system_prompt = DEFAULT_SYSTEM_PROMPT
        print("Using default system prompt for MATH dataset")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    os.makedirs(args.output_dir, exist_ok=True)

    for split in splits:
        print(f"\nProcessing {split} split...")

        dataset = build_math_lighteval_dataset(args.input_dir, split=split, system_prompt=args.system_prompt)
        print(f"Processed {len(dataset)} examples")

        output_file = os.path.join(args.output_dir, f"{split}.parquet")
        dataset.to_parquet(output_file)
        print(f"Saved to {output_file}")

    print(f"\nDone! Output directory: {args.output_dir}")

    if args.hdfs_dir is not None:
        print(f"Copying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=args.output_dir, dst=args.hdfs_dir)
        print("HDFS copy completed!")
