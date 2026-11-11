"""Natural Reasoning grading: last \\boxed{} answer graded by nr_answer_verify."""

from .math import last_boxed_only_string, remove_boxed
from .nr_answer_verify import grade_single_answer


def compute_score(solution_str: str, ground_truth: str, fast: bool = False) -> float:
    """1.0 if the boxed answer matches ground_truth, else 0.0. `fast` is unused."""
    try:
        boxed_str = last_boxed_only_string(solution_str)
        if boxed_str is None:
            return 0.0
        answer = remove_boxed(boxed_str)
        if not answer or not ground_truth:
            return 0.0
        correct = grade_single_answer(answer, ground_truth)
        return 1.0 if correct else 0.0
    except Exception:
        return 0.0
