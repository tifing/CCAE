# Copyright 2024 PRIME team and/or its affiliates
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
"""
Answer checker API that uses sympy to simplify expressions and check for equality.
Enhanced with 3-layer fallback (mathd + sympy + math_verify) for maximum recall.

Call grade_answer(given_answer: str, ground_truth: str).

FROM: https://github.com/openai/prm800k/blob/main/prm800k/grading/grader.py
"""

import contextlib
import math
import re

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
from sympy.parsing.latex import parse_latex

from verl.utils.py_functional import timeout_limit

from . import math_normalize
from .grader import math_equal

try:
    from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse as math_verify_parse, verify as math_verify_verify
    _HAS_MATH_VERIFY = True
except ImportError:
    _HAS_MATH_VERIFY = False

# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
        "liter",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub("\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        with contextlib.suppress(Exception):
            expr = _parse_latex(expr)

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    return all(re.search(bad_regex, expr) is None for bad_regex in BAD_REGEXES)


@timeout_limit(seconds=10)
def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    if given_answer is None:
        return False

    ground_truth_normalized_mathd = math_normalize.normalize_answer(ground_truth)
    given_answer_normalized_mathd = math_normalize.normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if (
        len(ground_truth_elems) > 1
        and (ground_truth_normalized[0] != given_normalized[0] or ground_truth_normalized[-1] != given_normalized[-1])
        or len(ground_truth_elems) != len(given_elems)
    ):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match
                # (no sympy.simplify)
                is_correct = False
            else:
                try:
                    is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
                except Exception as e:
                    # if there's an error, we'll just say it's not correct
                    is_correct = False
                    print(f"Error: {e} from are_equal_under_sympy, {ground_truth_elem}, {given_elem}")
            if not is_correct:
                break

    return is_correct


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1 : right_brace_idx].strip()


def match_answer(response):
    is_matched = False
    for ans_marker in ["answer:", "answer is", "answers are"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker) :].strip()
            if response.endswith("\n"):
                response = response[:-2]

    for ans_marker in ["is answer", "is the answer", "are answers", "are the answers"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[:ans_idx].strip()
            if response.endswith("\n"):
                response = response[:-2]

    # Find boxed
    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed

    if ". " in response:
        dot_idx = response.lower().rfind(". ")
        if dot_idx != -1:
            response = response[:dot_idx].strip()

    for ans_marker in ["be ", "is ", "are ", "=", ": ", "get ", "be\n", "is\n", "are\n", ":\n", "get\n"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker) :].strip()
            if response.endswith("\n"):
                response = response[:-2]

    is_matched = is_matched if any([c.isdigit() for c in response]) else False  # answer must have a digit
    return is_matched, response


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]


REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer for fallback comparison."""
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def _fallback_latex_equal(given_answer: str, ground_truth: str) -> bool:
    """Fallback: normalize and compare with sympy/latex parsing."""
    try:
        norm1, norm2 = normalize_final_answer(given_answer), normalize_final_answer(ground_truth)
        sym1 = parse_latex(norm1)
        sym2 = parse_latex(norm2)
        val1 = sym1.evalf()
        val2 = sym2.evalf()
        if sym1 == sym2 or val1 == val2:
            return True
        return norm1 == norm2
    except Exception:
        try:
            norm1, norm2 = normalize_final_answer(given_answer), normalize_final_answer(ground_truth)
            return norm1 == norm2
        except Exception:
            return False


def is_latex_equal(given_answer: str, ground_truth: str, timeout_seconds: int = 5) -> bool:
    """Layer 3: math_verify equivalence, with a sympy/latex fallback."""
    if not _HAS_MATH_VERIFY:
        return _fallback_latex_equal(given_answer, ground_truth)
    
    try:
        if "$" not in given_answer:
            given_answer = f"${given_answer}$"
        if "$" not in ground_truth:
            ground_truth = f"${ground_truth}$"
        
        result = math_verify_verify(
            math_verify_parse(
                ground_truth,
                extraction_config=(
                    LatexExtractionConfig(boxed_match_priority=0),
                    ExprExtractionConfig(),
                ),
                fallback_mode="no_fallback",
                extraction_mode=["first_match"],
                parsing_timeout=1,
            ),
            math_verify_parse(
                given_answer,
                extraction_config=(
                    LatexExtractionConfig(boxed_match_priority=0),
                    ExprExtractionConfig(),
                ),
                fallback_mode="no_fallback",
                extraction_mode=["first_match"],
                parsing_timeout=1,
            ),
            timeout_seconds=timeout_seconds,
        )
        if result:
            return True
        return _fallback_latex_equal(given_answer, ground_truth)
    except Exception:
        return _fallback_latex_equal(given_answer, ground_truth)


def compute_score(model_output: str, ground_truth: str, fast: bool = False) -> bool:
    """Return (is_correct, format_correct, extracted_answer); fast=True skips math_verify."""
    model_output = str(model_output)
    ground_truth = str(ground_truth)

    is_matched, extracted_model_output = match_answer(model_output)
    format_correctness = "Step 2:" in model_output and "\\box" in model_output

    if grade_answer(extracted_model_output, ground_truth):
        return True, True, extracted_model_output

    is_correct = False
    try:
        if "\pi" in extracted_model_output or "\pi" in ground_truth:
            equivs = []
            for pi in [math.pi, 3.14]:
                equivs.append(math_equal(extracted_model_output, ground_truth, timeout=True, pi=pi))
            is_correct = any(equivs)
        else:
            is_correct = math_equal(extracted_model_output, ground_truth, timeout=True)
    except Exception:
        is_correct = False

    if not is_correct and not fast and _HAS_MATH_VERIFY:
        try:
            is_correct = is_latex_equal(extracted_model_output, ground_truth)
        except Exception:
            pass

    return is_correct, format_correctness, extracted_model_output
