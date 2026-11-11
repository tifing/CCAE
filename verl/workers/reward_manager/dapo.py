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

from collections import defaultdict
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
    """Fall back to /tmp if the cwd was deleted (multiprocessing.spawn calls os.getcwd())."""
    try:
        os.getcwd()
    except (FileNotFoundError, OSError):
        fallback_dir = os.environ.get("REWARD_POOL_CWD", "/tmp")
        print(f"[RewardPool] Warning: cwd was deleted, changing to {fallback_dir}")
        os.chdir(fallback_dir)


def _get_reward_pool(num_workers: int = 32):
    """Global scoring pool; workers are recycled via maxtasksperchild."""
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
            ctx = mp.get_context('spawn')  # safe under CUDA/Ray
            _REWARD_POOL = ctx.Pool(
                processes=num_workers,
                maxtasksperchild=_REWARD_POOL_MAXTASKS,
            )
            _REWARD_POOL_SIZE = num_workers
            print(f"[RewardPool] Created pool with {num_workers} workers, maxtasksperchild={_REWARD_POOL_MAXTASKS}")
    return _REWARD_POOL

def _maybe_restart_reward_pool(task_count: int = 1, num_workers: int = 32):
    """No-op; worker recycling is handled by maxtasksperchild."""
    pass


def reset_reward_pool(timeout: float = 5.0):
    """Terminate the pool; must follow a timeout or exception so the pool does not respawn workers."""
    global _REWARD_POOL
    import signal
    import time
    
    with _REWARD_POOL_LOCK:
        if _REWARD_POOL is not None:
            pool = _REWARD_POOL
            _REWARD_POOL = None
            
            try:
                try:
                    pool.terminate()
                except Exception:
                    pass
                
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if hasattr(pool, '_pool') and pool._pool:
                        alive_workers = [p for p in pool._pool if p.is_alive()]
                        if not alive_workers:
                            break
                    else:
                        break
                    time.sleep(0.1)
                
                if hasattr(pool, '_pool') and pool._pool:
                    for proc in pool._pool:
                        if proc.is_alive():
                            try:
                                os.kill(proc.pid, signal.SIGKILL)
                            except (ProcessLookupError, OSError):
                                pass
                
                time.sleep(0.2)
                
                # join in a thread so a stuck join() cannot block forever
                def _join_pool():
                    try:
                        pool.join()
                    except Exception:
                        pass
                
                join_thread = threading.Thread(target=_join_pool, daemon=True)
                join_thread.start()
                join_thread.join(timeout=2.0)
                    
            except Exception as e:
                print(f"[RewardPool] Warning during pool reset: {e}")


def _compute_score_wrapper(args: Tuple) -> Any:
    """Module-level so it is picklable for the pool."""
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
    """Clean up pool processes using psutil for graceful termination."""
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


def _dummy_task():
    """Dummy task for pool warmup."""
    return 0


def warmup_reward_pool(num_workers: int = 32, timeout: float = 90.0):
    """Start all workers up front so the first scoring call does not hit the spawn latency."""
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


def run_parallel_reward_scoring(
    score_args_list: List[Tuple],
    num_processes: int = 32,
    timeout: float = 90.0,
) -> List[Any]:
    """Score (data_source, solution_str, ground_truth, extra_info) tuples in parallel; timed-out tasks score 0."""
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
            print(f"[RewardPool] {timed_out_count}/{num_tasks} tasks timed out after {timeout}s each")
            need_reset = True
            
    except Exception as e:
        print(f"[RewardPool] Exception in parallel reward scoring: {e}")
        traceback.print_exc()
        need_reset = True
    finally:
        if need_reset and pool is not None:
            _cleanup_pool_processes(pool)
            reset_reward_pool()
    
    return results


@register("dapo")
class DAPORewardManager:
    """DAPO reward manager with an async parallel process pool for answer scoring."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        num_reward_processes: int = 32,
        reward_timeout: float = 90.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.num_reward_processes = num_reward_processes
        self.reward_timeout = reward_timeout

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

        warmup_reward_pool(num_workers=num_reward_processes, timeout=90.0)

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Compute rewards via an async parallel process pool."""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
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
        extra_infos = []
        
        eos_token = self.tokenizer.eos_token
        
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
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            score_args_list.append((data_source, response_str, ground_truth, extra_info))
            valid_response_lengths.append(valid_response_length)
            prompt_strs.append(prompt_str)
            response_strs.append(response_str)
            ground_truths.append(ground_truth)
            data_sources.append(data_source)
            extra_infos.append(extra_info)

        results = run_parallel_reward_scoring(
            score_args_list,
            num_processes=self.num_reward_processes,
            timeout=self.reward_timeout,
        )

        for i in range(batch_size):
            result = results[i]
            valid_response_length = valid_response_lengths[i]
            prompt_str = prompt_strs[i]
            response_str = response_strs[i]
            ground_truth = ground_truths[i]
            data_source = data_sources[i]

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result

            reward = score

            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

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
                if isinstance(result, dict):
                    for key, value in result.items():
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
