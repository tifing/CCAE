# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright 2024 PRIME team and/or its affiliates
# Modified by the CCAE authors (2026); see NOTICE.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

"""Math answer grading in three layers: mathd string match, sympy equality, math_verify (optional)."""

import os
import re
import signal

try:
    import sympy
    from sympy.parsing.latex import parse_latex  # noqa: F401
    from sympy.parsing import sympy_parser
    _HAS_SYMPY = True
except ImportError:
    _HAS_SYMPY = False

try:
    from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse as math_verify_parse, verify as math_verify_verify
    _HAS_MATH_VERIFY = True
except ImportError:
    _HAS_MATH_VERIFY = False

try:
    from pylatexenc import latex2text
    _HAS_PYLATEXENC = True
except ImportError:
    _HAS_PYLATEXENC = False


def timeout_ours(timeout_seconds: int = 8):
    """Timeout decorator for POSIX systems."""
    if os.name == "posix":
        def decorator(func):
            def handler(signum, frame):
                raise TimeoutError("Operation timed out!")

            def wrapper(*args, **kwargs):
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)

                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

            return wrapper

        return decorator
    else:
        def decorator(func):
            return func
        return decorator


def compute_score(solution_str, ground_truth, fast=False) -> float:
    """1.0 if the last boxed answer matches ground_truth; fast=True skips math_verify."""
    retval = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if grade(answer, ground_truth, fast=fast):
                retval = 1.0
    except Exception:
        pass

    return retval


def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    if solution is None:
        return None
    solution = remove_boxed(solution)
    return solution


# units mainly from MathQA
unit_texts = [
    "east", "degree", "mph", "kmph", "ft", "m sqaure", " m east", "sq m", "deg",
    "mile", "q .", "monkey", "prime", "ratio", "profit of rs", "rd", "o", "gm",
    "p . m", "lb", "tile", "per", "dm", "lt", "gain", "ab", "way", "west",
    "a .", "b .", "c .", "d .", "e .", "f .", "g .", "h .", "t", "a", "h",
    "no change", "men", "soldier", "pie", "bc", "excess", "st", "inches", "noon",
    "percent", "by", "gal", "kmh", "c", "acre", "rise", "a . m", "th", "π r 2",
    "sq", "mark", "l", "toy", "coin", "sq . m", "gallon", "° f", "profit", "minw",
    "yr", "women", "feet", "am", "pm", "hr", "cu cm", "square", "v â € ™", "are",
    "rupee", "rounds", "cubic", "cc", "mtr", "s", "ohm", "number", "kmph", "day",
    "hour", "minute", "min", "second", "man", "woman", "sec", "cube", "mt",
    "sq inch", "mp", "∏ cm ³", "hectare", "more", "sec", "unit", "cu . m", "cm 2",
    "rs .", "rs", "kg", "g", "month", "km", "m", "cm", "mm", "apple", "liter",
    "loss", "yard", "pure", "year", "increase", "decrease", "d", "less", "Surface",
    "litre", "pi sq m", "s .", "metre", "meter", "inch",
]
unit_texts.extend([t + "s" for t in unit_texts])


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) == 0 or substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    """Strip and normalize a string (mathd style)."""
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # inequality symbols
    string = string.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    # Remove unit: texts
    for _ in range(2):
        for unit_text in unit_texts:
            _string = re.sub(r"(^|\W)" + re.escape(unit_text) + r"($|\W)", r"\1\2", string)
            if _string != "":
                string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def mathd_normalize_answer(answer):
    """Normalize answer using mathd style (Layer 1)."""
    if answer is None:
        return None
    answer = str(answer).strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return strip_string(answer)
    except Exception:
        return answer


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    """Grade answer using mathd normalization (Layer 1)."""
    ground_truth_normalized = mathd_normalize_answer(ground_truth)
    given_answer_normalized = mathd_normalize_answer(given_answer)
    
    if ground_truth_normalized == given_answer_normalized:
        return True
    return False


TUPLE_CHARS = "()[]"


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


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _strip_properly_formatted_commas(expr: str) -> str:
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _inject_implicit_mixed_number(step: str) -> str:
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)
    return step


def _parse_latex_to_text(expr: str) -> str:
    if not _HAS_PYLATEXENC:
        return expr
    
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")
    
    return expr.strip()


def _normalize_for_sympy(expr: str):
    if expr is None:
        return None
    
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
        "degree", "cm", "centimeter", "meter", "mile", "second", "minute",
        "hour", "day", "week", "month", "year", "foot", "feet", "inch", "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\\^[0-9]+)?", "", expr)
    expr = re.sub("\\^ *\\\\circ", "", expr)
    
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]
    
    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    
    if "\\" in expr:
        try:
            expr = _parse_latex_to_text(expr)
        except Exception:
            pass
    
    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")
    expr = expr.lower()
    
    if _str_is_int(expr):
        expr = str(_str_to_int(expr))
    
    return expr


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _split_tuple(expr: str):
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


def _sympy_parse(expr: str):
    if not _HAS_SYMPY:
        return expr
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + 
                        (sympy_parser.implicit_multiplication_application,)),
    )


BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]


def _should_allow_eval(expr: str) -> bool:
    expr_for_count = expr.replace("sqrt", "").replace("frac", "")
    letters_count = len(set([x for x in expr_for_count if x.isalpha()]))
    if letters_count > 2:
        return False
    
    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False
    
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False
    
    return True


@timeout_ours(timeout_seconds=1)
def _are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str) -> bool:
    if not _HAS_SYMPY:
        return False
    
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if _should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    """Layer 2: symbolic equality after sympy normalization."""
    MAX_EXPR_LENGTH = 500
    if len(given_answer) > MAX_EXPR_LENGTH or len(ground_truth) > MAX_EXPR_LENGTH:
        return False
    
    ground_truth_normalized = _normalize_for_sympy(ground_truth)
    given_normalized = _normalize_for_sympy(given_answer)
    
    if ground_truth_normalized is None:
        return False
    
    if ground_truth_normalized == given_normalized:
        return True
    
    if not given_normalized or len(given_normalized) == 0:
        return False
    
    ground_truth_elems = _split_tuple(ground_truth_normalized)
    given_elems = _split_tuple(given_normalized)
    
    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0] or 
        ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        is_correct = True
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                is_correct = False
            else:
                try:
                    is_correct = _are_equal_under_sympy(ground_truth_elem, given_elem)
                except TimeoutError:
                    is_correct = False
            if not is_correct:
                break
    
    return is_correct


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
        if _HAS_SYMPY:
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


import threading as _mv_threading
import os as _mv_os

_MATH_VERIFY_POOL = None
_MATH_VERIFY_POOL_LOCK = _mv_threading.Lock()
_MATH_VERIFY_POOL_MAXTASKS = int(_mv_os.environ.get("MATH_VERIFY_POOL_MAXTASKS", "100"))
_MATH_VERIFY_POOL_INITIALIZED = False


def _init_math_verify_pool():
    """Lazily create the math_verify process pool."""
    global _MATH_VERIFY_POOL, _MATH_VERIFY_POOL_INITIALIZED
    
    with _MATH_VERIFY_POOL_LOCK:
        if _MATH_VERIFY_POOL is None:
            import multiprocessing as mp
            ctx = mp.get_context('spawn')  # safe under CUDA/Ray
            _MATH_VERIFY_POOL = ctx.Pool(processes=2, maxtasksperchild=_MATH_VERIFY_POOL_MAXTASKS)
            if not _MATH_VERIFY_POOL_INITIALIZED:
                print(f"[MathVerifyPool] Created pool with 2 workers, maxtasksperchild={_MATH_VERIFY_POOL_MAXTASKS}")
                _MATH_VERIFY_POOL_INITIALIZED = True
    return _MATH_VERIFY_POOL


def _maybe_restart_math_verify_pool(task_count: int = 1):
    """No-op; worker recycling is handled by maxtasksperchild."""
    pass


def _math_verify_task(args):
    """Worker task for math_verify in process pool."""
    given_answer, ground_truth = args
    try:
        from math_verify import (
            LatexExtractionConfig, 
            ExprExtractionConfig, 
            parse as mv_parse, 
            verify as mv_verify
        )
        
        if "$" not in given_answer:
            given_answer = f"${given_answer}$"
        if "$" not in ground_truth:
            ground_truth = f"${ground_truth}$"
        
        return mv_verify(
            mv_parse(
                ground_truth,
                extraction_config=(
                    LatexExtractionConfig(boxed_match_priority=0),
                    ExprExtractionConfig(),
                ),
                fallback_mode="no_fallback",
                extraction_mode=["first_match"],
                parsing_timeout=1,
            ),
            mv_parse(
                given_answer,
                extraction_config=(
                    LatexExtractionConfig(boxed_match_priority=0),
                    ExprExtractionConfig(),
                ),
                fallback_mode="no_fallback",
                extraction_mode=["first_match"],
                parsing_timeout=1,
            ),
            timeout_seconds=1,
        )
    except Exception:
        return False


def is_latex_equal(given_answer: str, ground_truth: str, timeout_seconds: int = 2) -> bool:
    """Layer 3: math_verify equivalence in a pooled subprocess with timeout; falls back to sympy/latex."""
    if not _HAS_MATH_VERIFY:
        return _fallback_latex_equal(given_answer, ground_truth)
    
    MAX_EXPR_LENGTH = 500
    if len(given_answer) > MAX_EXPR_LENGTH or len(ground_truth) > MAX_EXPR_LENGTH:
        return _fallback_latex_equal(given_answer, ground_truth)
    
    try:
        pool = _init_math_verify_pool()
        async_result = pool.apply_async(_math_verify_task, ((given_answer, ground_truth),))
        result = async_result.get(timeout=timeout_seconds)
        _maybe_restart_math_verify_pool(1)
        if result:
            return True
        return _fallback_latex_equal(given_answer, ground_truth)
    except TimeoutError:
        _maybe_restart_math_verify_pool(1)
        return _fallback_latex_equal(given_answer, ground_truth)
    except Exception:
        _maybe_restart_math_verify_pool(1)
        return _fallback_latex_equal(given_answer, ground_truth)


def grade(model_answer: str, gt_answer: str, fast: bool = True) -> bool:
    """mathd or sympy match; with fast=False, also math_verify."""
    if gt_answer and "\\boxed" in gt_answer:
        gt_answer = extract_boxed_answer(gt_answer) or gt_answer
    
    correct = grade_answer_mathd(model_answer, gt_answer) or grade_answer_sympy(model_answer, gt_answer)
    
    if not fast and not correct:
        correct = is_latex_equal(model_answer, gt_answer)
    
    return correct


def is_equiv(str1, str2, verbose=False):
    """Kept for backward compatibility."""
    return grade(str1, str2, fast=True)
