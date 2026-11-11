"""GPQA grading, following EMPO's evaluation (math_verify extraction + MMLU-style normalization)."""

import re

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse


GPQA_EXTRACTION_CONFIG = [
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
]


def mmlu_normalize_final_answer(final_answer: str) -> str:
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    return final_answer


def compute_score(solution_str: str, ground_truth: str) -> float:
    """1.0 if the extracted letter matches ground_truth (A-D), also accepting the "D. option" form."""
    try:
        parsed = parse(
            solution_str,
            extraction_config=GPQA_EXTRACTION_CONFIG,
            extraction_mode="first_match",
        )
        if len(parsed) == 0:
            prediction = ""
        elif len(parsed) == 1:
            prediction = str(parsed[0])
        else:
            prediction = str(parsed[-1])
        prediction = mmlu_normalize_final_answer(prediction)

        gt = ground_truth.strip()
        if prediction == gt:
            return 1.0
        if len(gt) == 1 and gt in "ABCD" and len(prediction) > 1:
            if prediction[0] == gt and prediction[1] in ".) ":
                return 1.0
    except Exception:
        pass
    return 0.0
