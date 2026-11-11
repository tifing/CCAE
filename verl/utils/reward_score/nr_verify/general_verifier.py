"""Model-based answer equivalence for CCAE-NR (e.g. TIGER-Lab/general-verifier).

GeneralVerifier runs in-process on a GPU; SubprocessVerifier proxies it from processes
without GPU access (e.g. the Ray TaskRunner). Clustering follows EMPO (arXiv:2504.05812).
"""

import os
import traceback
import multiprocessing as mp

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from rouge import Rouge
    _rouge = Rouge()
except ImportError:
    _rouge = None


def _format_verification_prompt(answer1, answer2, question):
    """Build the prompt used by GeneralVerifier for pairwise answer comparison."""
    return (
        f"User: ### Question: {question}\n\n"
        f"### Ground Truth Answer: {answer1}\n\n"
        f"### Student Answer: {answer2}\n\n"
        "For the above question, please verify if the student's answer "
        "is equivalent to the ground truth answer.\n"
        "Do not solve the question by yourself; just check if the student's "
        "answer is equivalent to the ground truth answer.\n"
        'If correct, output "Final Decision: Yes". '
        'If incorrect, output "Final Decision: No".\n'
        "Assistant: Final Decision: "
    )


class GeneralVerifier:
    """Pairwise answer equivalence from the verifier's Yes/No next-token probabilities."""

    def __init__(self, model_path=None, device=None):
        if model_path is None:
            model_path = os.environ.get(
                'NR_VERIFIER_MODEL_PATH', 'TIGER-Lab/general-verifier'
            )

        if device is None:
            gpu_count = torch.cuda.device_count()
            device = f"cuda:{gpu_count - 1}" if gpu_count > 0 else "cpu"

        self.device = device
        print(f"[GeneralVerifier] Loading model from {model_path} to {device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

        decision_tokens = self.tokenizer(" Yes", " No", add_special_tokens=False)
        self.yes_id = decision_tokens.input_ids[0]
        self.no_id = decision_tokens.input_ids[1]

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[GeneralVerifier] Model loaded successfully on {device}")

    def check_implication(self, answer1, answer2, question):
        """Return (is_equivalent, yes_prob - no_prob); equivalent iff yes_prob > no_prob, as in EMPO."""
        prompt = _format_verification_prompt(answer1, answer2, question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model.forward(inputs.input_ids)

        next_token_logits = outputs.logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=0)
        yes_prob = probs[self.yes_id].item()
        no_prob = probs[self.no_id].item()

        confidence = yes_prob - no_prob
        is_equivalent = yes_prob > no_prob
        return is_equivalent, confidence

    def check_implication_batch(self, answer1, answer2_list, question,
                                max_batch_size=None):
        """check_implication over answer2_list, one left-padded forward pass per max_batch_size chunk."""
        if max_batch_size is None:
            try:
                max_batch_size = int(os.environ.get('NR_VERIFIER_BATCH_SIZE', '16'))
            except (ValueError, TypeError):
                max_batch_size = 16
            max_batch_size = max(max_batch_size, 1)

        if not answer2_list:
            return []

        results = [(False, -1.0)] * len(answer2_list)

        valid = []
        for idx, answer2 in enumerate(answer2_list):
            if answer1 == '' or answer2 == '':
                continue
            valid.append((idx, _format_verification_prompt(answer1, answer2, question)))

        if not valid:
            return results

        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'

        try:
            for chunk_start in range(0, len(valid), max_batch_size):
                chunk = valid[chunk_start:chunk_start + max_batch_size]
                prompts = [p for _, p in chunk]

                inputs = self.tokenizer(
                    prompts, return_tensors="pt", padding=True
                ).to(self.device)

                with torch.inference_mode():
                    outputs = self.model.forward(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                    )

                for batch_idx, (orig_idx, _) in enumerate(chunk):
                    next_token_logits = outputs.logits[batch_idx, -1, :]
                    probs = torch.softmax(next_token_logits, dim=0)
                    yes_prob = probs[self.yes_id].item()
                    no_prob = probs[self.no_id].item()

                    confidence = yes_prob - no_prob
                    is_equiv = yes_prob > no_prob
                    results[orig_idx] = (is_equiv, confidence)
        finally:
            self.tokenizer.padding_side = orig_padding_side

        return results

    def compute_group_rewards(self, predictions, question):
        return cluster_nr_predictions(predictions, question, self)


# --- Subprocess verifier (for processes without GPU access, e.g. Ray TaskRunner) ---

def _default_subprocess_group_timeout():
    """Hard cap per group in subprocess (SIGALRM; see module docstring)."""
    return int(os.environ.get("NR_SUBPROCESS_GROUP_TIMEOUT", "300"))


def _default_subprocess_request_timeout():
    """Parent waits this long for one group result; must be > group timeout."""
    return int(os.environ.get("NR_SUBPROCESS_REQUEST_TIMEOUT", "360"))


def _default_model_clustering_timeout():
    """Python-level cap inside get_semantic_ids (sequential pairwise forwards)."""
    return int(os.environ.get("NR_MODEL_CLUSTERING_TIMEOUT", "240"))


def _subprocess_worker(model_path, gpu_id, req_q, resp_q, group_timeout_s):
    """Worker process: sets CUDA_VISIBLE_DEVICES, loads model, handles requests."""
    import signal

    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        verifier = GeneralVerifier(model_path=model_path, device='cuda:0')
        resp_q.put('ready')
    except Exception as e:
        traceback.print_exc()
        resp_q.put(f'error: {e}')
        return

    class _GroupTimeout(Exception):
        pass

    def _group_timeout_handler(signum, frame):
        raise _GroupTimeout()

    while True:
        try:
            msg = req_q.get()
            if msg is None:
                break
            cmd = msg[0]
            if cmd == 'check':
                _, a1, a2, q = msg
                resp_q.put(verifier.check_implication(a1, a2, q))
            elif cmd == 'group':
                _, predictions, question = msg
                old_handler = signal.signal(signal.SIGALRM, _group_timeout_handler)
                signal.alarm(group_timeout_s)
                try:
                    result = cluster_nr_predictions(predictions, question, verifier)
                    resp_q.put(result)
                except _GroupTimeout:
                    print(f"[SubprocessVerifier worker] Group timed out after "
                          f"{group_timeout_s}s, returning zeros")
                    n = len(predictions)
                    resp_q.put(([-1] * n, [False] * n, 0))
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
        except Exception as e:
            print(f"[SubprocessVerifier worker] Error: {e}")
            traceback.print_exc()
            resp_q.put(f'error: {e}')


class SubprocessVerifier:
    """Runs GeneralVerifier in a child process with GPU access; a lock serializes requests."""

    def __init__(self, model_path, gpu_id='7'):
        self._model_path = model_path
        self._gpu_id = str(gpu_id)
        self._group_timeout_s = _default_subprocess_group_timeout()
        self._request_timeout_s = _default_subprocess_request_timeout()
        if self._request_timeout_s <= self._group_timeout_s:
            self._request_timeout_s = self._group_timeout_s + 60
        self._lock = mp.Lock()
        self._start_subprocess()

    def _start_subprocess(self):
        ctx = mp.get_context('spawn')
        self._req_q = ctx.Queue()
        self._resp_q = ctx.Queue()
        self._process = ctx.Process(
            target=_subprocess_worker,
            args=(self._model_path, self._gpu_id,
                  self._req_q, self._resp_q, self._group_timeout_s),
            daemon=True,
        )
        self._process.start()
        print(f"[SubprocessVerifier] Waiting for model to load on GPU {self._gpu_id}...")
        status = self._resp_q.get(timeout=600)
        if status != 'ready':
            raise RuntimeError(f"Verifier subprocess failed: {status}")
        print(f"[SubprocessVerifier] Ready (GPU {self._gpu_id})")

    def _restart_subprocess(self):
        """Kill the hung subprocess and start a fresh one."""
        print("[SubprocessVerifier] Restarting subprocess due to timeout...")
        try:
            self._process.kill()
            self._process.join(timeout=10)
        except Exception:
            pass
        self._start_subprocess()

    def check_implication(self, answer1, answer2, question):
        import queue
        with self._lock:
            self._req_q.put(('check', answer1, answer2, question))
            try:
                result = self._resp_q.get(timeout=self._request_timeout_s)
                if isinstance(result, str) and result.startswith('error:'):
                    return False, -1.0
                return result
            except queue.Empty:
                print("[SubprocessVerifier] Timeout on check_implication, restarting")
                self._restart_subprocess()
                return False, -1.0

    def compute_group_rewards(self, predictions, question):
        import queue
        with self._lock:
            self._req_q.put(('group', list(predictions), question))
            try:
                result = self._resp_q.get(timeout=self._request_timeout_s)
                if isinstance(result, str) and result.startswith('error:'):
                    print(f"[SubprocessVerifier] Worker error: {result}")
                    n = len(predictions)
                    return [-1] * n, [False] * n, 0
                return result
            except queue.Empty:
                print(f"[SubprocessVerifier] Timeout ({self._request_timeout_s}s) "
                      f"waiting for group rewards, restarting subprocess")
                self._restart_subprocess()
                n = len(predictions)
                return [-1] * n, [False] * n, 0

    def __del__(self):
        try:
            self._req_q.put(None)
            self._process.join(timeout=5)
        except Exception:
            pass


# --- Clustering and reward functions ---

def get_semantic_ids(strings_list, question, model):
    import time as _t
    clustering_budget = _default_model_clustering_timeout()
    deadline = _t.monotonic() + clustering_budget

    n = len(strings_list)
    semantic_set_ids = [-1] * n
    next_id = 0
    timed_out = False

    for i in range(n):
        if timed_out:
            break
        if semantic_set_ids[i] != -1:
            continue

        # Empty strings seed their own cluster and skip pairwise comparison.
        if strings_list[i] == '':
            semantic_set_ids[i] = next_id
            next_id += 1
            continue

        semantic_set_ids[i] = next_id

        # Compare only against non-empty j > i; empty ones seed their own cluster.
        candidates_j = [j for j in range(i + 1, n) if strings_list[j] != '']
        if not candidates_j:
            next_id += 1
            continue

        if _t.monotonic() > deadline:
            print(f"[get_semantic_ids] Timeout ({clustering_budget}s) "
                  f"at seed {i}/{n}, assigning remaining individually")
            timed_out = True
            next_id += 1
            break

        seed_answer = strings_list[i]

        for j in candidates_j:
            if _t.monotonic() > deadline:
                timed_out = True
                break
            if strings_list[j] == seed_answer:
                semantic_set_ids[j] = next_id
                continue
            is_equiv, _conf = model.check_implication(
                seed_answer, strings_list[j], question
            )
            if is_equiv:
                semantic_set_ids[j] = next_id

        next_id += 1

    for i in range(n):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id
            next_id += 1

    return semantic_set_ids


_clustering_stats = {'model': 0, 'model_samples': 0, '_last_logged': 0}


def cluster_nr_predictions(predictions, question, verifier):
    global _clustering_stats
    n_generations = len(predictions)
    n_extractable = sum(1 for p in predictions if p != '')

    _clustering_stats['model'] += 1
    _clustering_stats['model_samples'] += n_extractable
    try:
        semantic_ids = get_semantic_ids(predictions, question, verifier)
    except Exception as e:
        print(f'[GeneralVerifier] Failed to cluster: {e}')
        return [-1] * n_generations, [False] * n_generations, 0

    total = _clustering_stats['model']
    if total > 0 and total % 128 == 0 and total != _clustering_stats['_last_logged']:
        _clustering_stats['_last_logged'] = total
        s = _clustering_stats
        print(
            f"[Clustering Stats] groups={total} | "
            f"model={s['model']} | samples={s['model_samples']}"
        )

    counts = np.bincount(semantic_ids)
    probabilities = counts / n_generations
    assert np.isclose(probabilities.sum(), 1)

    max_prob = np.max(probabilities)

    extractable_cluster_ids = set()
    for index in range(n_generations):
        if predictions[index] != '':
            extractable_cluster_ids.add(semantic_ids[index])
    num_extractable_clusters = len(extractable_cluster_ids)

    global_sparse = max_prob < 0.2 and len(probabilities) > 1
    valid_mask = []
    for index in range(n_generations):
        if predictions[index] == '':
            valid_mask.append(False)
            continue
        if global_sparse:
            valid_mask.append(False)
            continue
        try:
            if _rouge is not None and predictions[index] != '':
                rouge_score = _rouge.get_scores(
                    predictions[index].lower(), question.lower()
                )
                rep = rouge_score[0]["rouge-l"]["p"]
            else:
                rep = 0.0
        except Exception:
            rep = 1.0

        if rep > 0.8 or predictions[index].lower() in question.lower():
            valid_mask.append(False)
        else:
            valid_mask.append(True)

    return semantic_ids, valid_mask, num_extractable_clusters


def compute_model_based_frequency(predictions, question, verifier):
    n_generations = len(predictions)
    if n_generations == 0:
        return [], 0, []
    semantic_ids, valid_mask, num_extractable_clusters = cluster_nr_predictions(
        predictions, question, verifier
    )
    if all(s == -1 for s in semantic_ids):
        return [0.0] * n_generations, 0, semantic_ids

    counts = np.bincount(semantic_ids)
    probabilities = counts / n_generations
    rewards = []
    for index in range(n_generations):
        if not valid_mask[index]:
            rewards.append(0.0)
        else:
            rewards.append(float(probabilities[semantic_ids[index]]))

    return rewards, num_extractable_clusters, semantic_ids
