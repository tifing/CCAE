# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modified by the CCAE authors (2026); see NOTICE.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter, defaultdict
import os
import traceback
import threading
from typing import List, Tuple, Any
import multiprocessing as mp

import torch

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from verl import DataProto
from verl.utils.reward_score import default_compute_score, get_reward_logger
from verl.workers.reward_manager import register


_REWARD_POOL = None
_REWARD_POOL_SIZE = 32
_REWARD_POOL_LOCK = threading.Lock()
_REWARD_POOL_MAXTASKS = int(os.environ.get("REWARD_POOL_MAXTASKS", "500"))


def _ensure_valid_cwd():
    try:
        os.getcwd()
    except (FileNotFoundError, OSError):
        fallback_dir = os.environ.get("REWARD_POOL_CWD", "/tmp")
        print(f"[RewardPool] Warning: cwd was deleted, changing to {fallback_dir}")
        os.chdir(fallback_dir)


def _get_reward_pool(num_workers: int = 32):
    global _REWARD_POOL, _REWARD_POOL_SIZE

    _ensure_valid_cwd()

    with _REWARD_POOL_LOCK:
        if _REWARD_POOL is None or _REWARD_POOL_SIZE != num_workers:
            if _REWARD_POOL is not None:
                try:
                    _REWARD_POOL.close()
                    _REWARD_POOL.join()
                except Exception:
                    pass
            _ensure_valid_cwd()
            ctx = mp.get_context('spawn')
            _REWARD_POOL = ctx.Pool(
                processes=num_workers,
                maxtasksperchild=_REWARD_POOL_MAXTASKS,
            )
            _REWARD_POOL_SIZE = num_workers
            print(f"[RewardPool] Created pool with {num_workers} workers, maxtasksperchild={_REWARD_POOL_MAXTASKS}")
    return _REWARD_POOL


def _maybe_restart_reward_pool(task_count: int = 1, num_workers: int = 32):
    pass


def reset_reward_pool(timeout: float = 5.0):
    global _REWARD_POOL
    import signal

    with _REWARD_POOL_LOCK:
        if _REWARD_POOL is None:
            return
        pool = _REWARD_POOL
        _REWARD_POOL = None

    killed = 0
    try:
        if hasattr(pool, "_pool") and pool._pool:
            for proc in list(pool._pool):
                try:
                    if proc.is_alive():
                        os.kill(proc.pid, signal.SIGKILL)
                        killed += 1
                except (ProcessLookupError, OSError, AttributeError):
                    pass
    except Exception:
        pass
    if killed:
        print(f"[RewardPool] reset: SIGKILL'd {killed} worker(s)")

    def _terminate_and_join():
        try:
            pool.terminate()
        except Exception:
            pass
        try:
            pool.join()
        except Exception:
            pass

    t = threading.Thread(target=_terminate_and_join, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        print(f"[RewardPool] reset: terminate/join still running after {timeout}s, "
              f"abandoning pool object (workers already killed)")


def _dummy_task():
    return 0


def warmup_reward_pool(num_workers: int = 32, timeout: float = 90.0):
    import time
    print(f"[RewardPool] Warming up process pool with {num_workers} workers...")
    start_time = time.time()

    pool = _get_reward_pool(num_workers)
    async_results = [pool.apply_async(_dummy_task) for _ in range(num_workers)]
    for ar in async_results:
        try:
            ar.get(timeout=timeout)
        except Exception as e:
            print(f"[RewardPool] Warning: warmup task failed: {e}")

    elapsed = time.time() - start_time
    print(f"[RewardPool] Pool warmup completed in {elapsed:.2f}s")


def _compute_score_wrapper(args: Tuple) -> Any:
    data_source, solution_str, ground_truth, extra_info = args
    try:
        result = default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        return result
    except Exception:
        return 0.0


def _cleanup_pool_processes(pool, timeout: float = 5.0):
    if not _HAS_PSUTIL:
        return

    terminated_count = 0
    try:
        if hasattr(pool, '_pool'):
            for proc in pool._pool:
                try:
                    if proc.is_alive():
                        p = psutil.Process(proc.pid)
                        p.terminate()
                        try:
                            p.wait(timeout=timeout)
                        except psutil.TimeoutExpired:
                            p.kill()
                        terminated_count += 1
                except (psutil.NoSuchProcess, AttributeError):
                    pass
                except Exception as e:
                    print(f"[RewardPool] Error cleaning up process: {e}")
    except Exception:
        pass

    if terminated_count > 0:
        print(f"[RewardPool] Cleaned up {terminated_count} subprocess(es)")


def run_parallel_reward_scoring(
    score_args_list: List[Tuple],
    num_processes: int = 32,
    timeout: float = 90.0,
) -> List[Any]:
    if not score_args_list:
        return []

    num_tasks = len(score_args_list)
    results = [0.0] * num_tasks
    timed_out_count = 0
    pool = None
    need_reset = False

    try:
        pool = _get_reward_pool(num_processes)
        async_results = [pool.apply_async(_compute_score_wrapper, (args,)) for args in score_args_list]

        for i, ar in enumerate(async_results):
            try:
                result = ar.get(timeout=timeout)
                results[i] = result if result is not None else 0.0
            except mp.TimeoutError:
                results[i] = 0.0
                timed_out_count += 1
            except Exception:
                results[i] = 0.0
        
        if timed_out_count > 0:
            reset_threshold = max(16, (num_tasks + 9) // 10)
            print(f"[RewardPool] {timed_out_count}/{num_tasks} tasks timed out after {timeout}s each "
                  f"(reset threshold: {reset_threshold})")
            if timed_out_count >= reset_threshold:
                need_reset = True
            else:
                print(f"[RewardPool] Keeping pool alive — isolated timeouts below threshold.")

    except Exception as e:
        print(f"[RewardPool] Exception in parallel reward scoring: {e}")
        traceback.print_exc()
        need_reset = True
    finally:
        if need_reset and pool is not None:
            reset_reward_pool()

    return results


def _compute_voting_info_for_naive(response_texts, uids):
    """Compute consensus targets (majority answers) and per-sample agreement by uid grouping."""
    from collections import Counter
    from verl.trainer.ppo.consensus import (
        batch_simplify_expression_strings,
        extract_answer_string,
    )

    batch_size = len(response_texts)
    is_positive = [False] * batch_size
    is_non_extractable = [False] * batch_size
    majority_gt = ["None"] * batch_size
    majority_ratio = [0.0] * batch_size

    sample_extracted = [None] * batch_size
    for i, comp in enumerate(response_texts):
        ans = extract_answer_string(comp)
        if ans is None:
            is_non_extractable[i] = True
        else:
            sample_extracted[i] = ans

    all_extracted_idx = [i for i, ans in enumerate(sample_extracted) if ans is not None]
    if all_extracted_idx:
        unique_raw = list({sample_extracted[i] for i in all_extracted_idx})
        normalized = batch_simplify_expression_strings(
            unique_raw, timeout_seconds=1.0, chunk_timeout_seconds=50.0
        )
        raw_to_norm = dict(zip(unique_raw, normalized))
    else:
        raw_to_norm = {}

    id2indices = defaultdict(list)
    for i, uid in enumerate(uids):
        id2indices[uid].append(i)

    for uid, indices in id2indices.items():
        group_size = len(indices)
        extracted_in_group = [
            (global_idx, raw_to_norm[sample_extracted[global_idx]])
            for global_idx in indices
            if sample_extracted[global_idx] is not None
        ]

        if not extracted_in_group:
            for global_idx in indices:
                majority_gt[global_idx] = "None"
                majority_ratio[global_idx] = 0.0
            continue

        counter = Counter(norm for _, norm in extracted_in_group)
        winner_answer, winner_count = counter.most_common(1)[0]
        ratio = winner_count / group_size if group_size > 0 else 0.0

        for global_idx in indices:
            majority_gt[global_idx] = winner_answer
            majority_ratio[global_idx] = ratio

        for global_idx, norm in extracted_in_group:
            if norm == winner_answer:
                is_positive[global_idx] = True

    is_training_sample = [True] * batch_size

    return {
        "is_positive": is_positive,
        "is_non_extractable": is_non_extractable,
        "is_training_sample": is_training_sample,
        "majority_gt": majority_gt,
        "majority_ratio": majority_ratio,
    }


@register("naive")
class NaiveRewardManager:
    """Decodes responses for CCAE voting; reference-answer scores, if any, are for logging only."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        num_reward_processes: int = 32,
        reward_timeout: float = 90.0,
        emit_responses: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.num_reward_processes = num_reward_processes
        self.reward_timeout = reward_timeout
        self.emit_responses = emit_responses

        warmup_reward_pool(num_workers=num_reward_processes, timeout=90.0)

    def __call__(self, data: DataProto, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        reward_logger = get_reward_logger()
        reward_logger.new_batch()

        batch_size = len(data)
        score_args_list = []
        valid_response_lengths = []
        prompt_strs = []
        response_strs = []
        ground_truths = []
        data_sources = []
        
        for i in range(batch_size):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score_args_list.append((data_source, response_str, ground_truth, extra_info))
            valid_response_lengths.append(valid_response_length)
            prompt_strs.append(prompt_str)
            response_strs.append(response_str)
            ground_truths.append(ground_truth)
            data_sources.append(data_source)

        scores = run_parallel_reward_scoring(
            score_args_list,
            num_processes=self.num_reward_processes,
            timeout=self.reward_timeout,
        )

        if self.emit_responses:
            reward_extra_info["responses"].extend(response_strs)
            if "uid" in data.non_tensor_batch:
                voting_info = _compute_voting_info_for_naive(
                    response_strs, list(data.non_tensor_batch["uid"])
                )
                reward_extra_info["is_positive"].extend(voting_info["is_positive"])
                reward_extra_info["is_non_extractable"].extend(voting_info["is_non_extractable"])
                reward_extra_info["is_training_sample"].extend(voting_info["is_training_sample"])
                reward_extra_info["majority_gt"].extend(voting_info["majority_gt"])
                reward_extra_info["majority_ratio"].extend(voting_info["majority_ratio"])

        for i in range(batch_size):
            score = scores[i]
            valid_response_length = valid_response_lengths[i]
            prompt_str = prompt_strs[i]
            response_str = response_strs[i]
            ground_truth = ground_truths[i]
            data_source = data_sources[i]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    if key == "responses":
                        continue
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if i == 0:
                data_item = data[i]
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                response_ids = data_item.batch["responses"]
                valid_response_ids = response_ids[:valid_response_length]

                prompt_str_raw = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
                response_str_raw = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                reward_logger.log_batch_sample(
                    prompt=prompt_str,
                    response=response_str,
                    gt=ground_truth,
                    score=reward,
                    prompt_raw=prompt_str_raw,
                    response_raw=response_str_raw
                )

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


