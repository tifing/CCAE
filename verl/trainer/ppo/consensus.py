import os
import time
import re
import signal
import torch
from typing import Optional, Dict, List, Tuple
from collections import defaultdict, Counter
import threading
import multiprocessing as mp

unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])

_SYMPY_POOL = None
_SYMPY_POOL_LOCK = threading.Lock()
_SYMPY_POOL_INITIALIZED = False

def _get_sympy_pool():
    """Global process pool for sympy calls; processes (unlike threads) can be killed on timeout."""
    global _SYMPY_POOL, _SYMPY_POOL_INITIALIZED
    
    with _SYMPY_POOL_LOCK:
        if _SYMPY_POOL is None:
            # 'spawn' is safe under CUDA/Ray
            start_method = os.getenv("SYMPY_POOL_START", "spawn")
            try:
                ctx = mp.get_context(start_method)
            except ValueError:
                ctx = mp.get_context("spawn")
            cpu_count = os.cpu_count() or 4
            default_procs = max(4, min(64, cpu_count // 2))
            pool_size = int(os.getenv("SYMPY_POOL_SIZE", default_procs))
            maxtasks = int(os.getenv("SYMPY_POOL_MAXTASKS", "1000"))
            _SYMPY_POOL = ctx.Pool(processes=pool_size, maxtasksperchild=maxtasks)
            if not _SYMPY_POOL_INITIALIZED:
                print(f"[SympyPool] Created pool with {pool_size} workers, maxtasksperchild={maxtasks}")
                _SYMPY_POOL_INITIALIZED = True
    return _SYMPY_POOL


def _maybe_restart_sympy_pool(task_count: int = 1):
    """No-op; worker recycling is handled by maxtasksperchild."""
    pass


def reset_sympy_pool(timeout: float = 5.0):
    """Terminate the sympy pool; called at the end of each training step."""
    global _SYMPY_POOL
    
    with _SYMPY_POOL_LOCK:
        if _SYMPY_POOL is not None:
            pool = _SYMPY_POOL
            _SYMPY_POOL = None
            
            try:
                pool.terminate()

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

                time.sleep(0.5)

                try:
                    pool.join()
                except Exception:
                    pass
                    
            except Exception as e:
                print(f"[SympyPool] Warning during pool reset: {e}")


try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr
    from sympy.parsing.latex import parse_latex
    from sympy import simplify
    _HAS_SYMPY = True
except ImportError:
    _HAS_SYMPY = False
    print("Warning: sympy not available, voting will use simple string matching")

try:
    from latex2sympy2_extended import latex2sympy
    _HAS_LATEX2SYMPY = True
except ImportError:
    try:
        from latex2sympy2 import latex2sympy
        _HAS_LATEX2SYMPY = True
    except ImportError:
        _HAS_LATEX2SYMPY = False

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



def last_boxed_only_string(string: str) -> Optional[str]:
    """Return the last \\boxed{...} (or \\fbox{...}) substring, or None."""
    idx = string.rfind("\\boxed")
    # "\boxed x" form without braces
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
    
    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]
    
    return retval


def remove_boxed(s: str) -> Optional[str]:
    """Strip the \\boxed{...} wrapper; None if malformed."""
    # "\boxed x" form without braces
    if "\\boxed " in s:
        left = "\\boxed "
        try:
            assert s[: len(left)] == left
            return s[len(left):]
        except Exception:
            return None
    
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def extract_boxed_answer(solution: str) -> Optional[str]:
    """Content of the last \\boxed{...} in `solution`, or None."""
    solution = last_boxed_only_string(solution)
    if solution is None:
        return None
    solution = remove_boxed(solution)
    return solution


def extract_answer_string(completion_text: str) -> Optional[str]:
    """Extract the \\boxed{} answer from a completion, or None."""
    if "\\boxed" not in completion_text:
        return None
    return extract_boxed_answer(completion_text)


def simplify_expression_string(expression_string: str, timeout_seconds: int = 1) -> str:
    """Sympy-normalize an expression (with timeout) so equivalent answers such as 1/2 and 0.5 merge."""
    if not _HAS_SYMPY:
        return expression_string

    try:
        pool = _get_sympy_pool()
        async_result = pool.apply_async(_simplify_expression_string_impl, (expression_string,))
        result = async_result.get(timeout=timeout_seconds)
        _maybe_restart_sympy_pool(1)
        return result
    except mp.TimeoutError:
        _maybe_restart_sympy_pool(1)
        return expression_string
    except Exception:
        _maybe_restart_sympy_pool(1)
        return expression_string


def batch_simplify_expression_strings(
    expression_strings: List[str], 
    timeout_seconds: float = 1.0,
    chunk_timeout_seconds: float = 50.0
) -> List[str]:
    """Simplify a list of expression strings in parallel; returns results in the same order."""
    if not _HAS_SYMPY or not expression_strings:
        return list(expression_strings)
    
    unique_strings = list(set(expression_strings))
    num_tasks = len(unique_strings)
    
    try:
        pool = _get_sympy_pool()
        async_results = {s: pool.apply_async(_simplify_expression_string_impl, (s,)) for s in unique_strings}
        result_map = {}
        start_time = time.time()
        timed_out = False
        for s in unique_strings:
            remaining = chunk_timeout_seconds - (time.time() - start_time)
            if remaining <= 0:
                timed_out = True
                result_map[s] = s
                continue
            try:
                result_map[s] = async_results[s].get(timeout=min(timeout_seconds, remaining))
            except mp.TimeoutError:
                result_map[s] = s
            except Exception:
                result_map[s] = s
        
        if timed_out:
            print(f"[Warning] batch_simplify_expression_strings timed out after {chunk_timeout_seconds}s")

        _maybe_restart_sympy_pool(num_tasks)

        return [result_map[s] for s in expression_strings]
    except mp.TimeoutError:
        print(f"[Warning] batch_simplify_expression_strings timed out after {chunk_timeout_seconds}s")
        _maybe_restart_sympy_pool(num_tasks)
        return list(expression_strings)
    except Exception as e:
        print(f"[Warning] batch_simplify_expression_strings failed: {e}")
        _maybe_restart_sympy_pool(num_tasks)
        return list(expression_strings)


def _simplify_expression_string_impl(expression_string: str) -> str:
    """Core simplification logic without timeout."""
    try:
        sympy_expr = parse_expr(
            expression_string, 
            transformations="all",
            evaluate=False
        )
        simplified_expr = simplify(sympy_expr)
        return str(simplified_expr)
    except Exception:
        if _HAS_LATEX2SYMPY:
            try:
                sympy_expr = latex2sympy(expression_string)
                simplified_expr = simplify(sympy_expr)
                return str(simplified_expr)
            except Exception:
                pass
        return expression_string


def _fix_fracs(string: str) -> str:
    """Fix fraction formatting."""
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
                except Exception:
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


def _fix_a_slash_b(string: str) -> str:
    """Convert a/b to \\frac{a}{b}."""
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
    except Exception:
        return string


def _remove_right_units(string: str) -> str:
    """Remove units on the right."""
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    """Fix sqrt formatting."""
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


def _strip_string(string: str) -> str:
    """Strip and normalize a string (mathd style)."""
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\neq", "\\ne").replace("\\leq", "\\le").replace("\\geq", "\\ge")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string
    for _ in range(2):
        for unit_text in unit_texts:
            _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
            if _string != "":
                string = _string
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    """Layer 1: Hendrycks MATH-style normalization."""
    if answer is None:
        return None
    answer = answer.strip()
    try:
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    """Layer 1: string equality after mathd normalization."""
    ground_truth_normalized = mathd_normalize_answer(ground_truth)
    given_answer_normalized = mathd_normalize_answer(given_answer)
    
    if ground_truth_normalized == given_answer_normalized:
        return True
    return False


def _is_float(num: str) -> bool:
    """Check if string is a float."""
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    """Check if float is an integer."""
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_is_int(x: str) -> bool:
    """Check if string represents an integer."""
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    """Convert string to int."""
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _strip_properly_formatted_commas(expr: str) -> str:
    """Strip commas from numbers like 1,000,000."""
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _inject_implicit_mixed_number(step: str) -> str:
    """Convert mixed numbers like '7 3/4' to '7+3/4'."""
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)
    return step


def _parse_latex_to_text(expr: str) -> str:
    """Convert LaTeX to plain text."""
    if not _HAS_PYLATEXENC:
        return expr
    
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")
    
    return expr.strip()


def _normalize_for_sympy(expr: str) -> Optional[str]:
    """Normalize expression for sympy comparison."""
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
    
    # edge case with mixed numbers and negative signs
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
    """Check if expression is a fraction."""
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


TUPLE_CHARS = "()[]"


def _split_tuple(expr: str) -> List[str]:
    """Split elements in a tuple/interval."""
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
    """Parse expression with sympy."""
    if not _HAS_SYMPY:
        return expr
    py_expr = expr.replace("^", "**")
    return sympy.parsing.sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy.parsing.sympy_parser.standard_transformations + 
                        (sympy.parsing.sympy_parser.implicit_multiplication_application,)),
    )


BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]


def _should_allow_eval(expr: str) -> bool:
    """Check if expression is safe to evaluate."""
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


def _are_equal_under_sympy_impl(ground_truth_normalized: str, given_normalized: str) -> bool:
    """Check equality using sympy (without timeout wrapper)."""
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


def _are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str, timeout_seconds: int = 1) -> bool:
    """Check equality using sympy with timeout (process-based, can actually kill stuck workers)."""
    try:
        pool = _get_sympy_pool()
        async_result = pool.apply_async(_are_equal_under_sympy_impl, (ground_truth_normalized, given_normalized))
        result = async_result.get(timeout=timeout_seconds)
        _maybe_restart_sympy_pool(1)
        return result
    except mp.TimeoutError:
        _maybe_restart_sympy_pool(1)
        return False
    except Exception:
        _maybe_restart_sympy_pool(1)
        return False


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    """Layer 2: symbolic equality after sympy normalization."""
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
                # if fractions aren't reduced, shouldn't be marked correct
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if ground truth is integer, require strict match
                is_correct = False
            else:
                is_correct = _are_equal_under_sympy(ground_truth_elem, given_elem)
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
    """Normalize a final answer for the fallback comparison."""
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


def is_latex_equal(given_answer: str, ground_truth: str, timeout_seconds: int = 1) -> bool:
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
                parsing_timeout=timeout_seconds,
            ),
            math_verify_parse(
                given_answer,
                extraction_config=(
                    LatexExtractionConfig(boxed_match_priority=0),
                    ExprExtractionConfig(),
                ),
                fallback_mode="no_fallback",
                extraction_mode=["first_match"],
                parsing_timeout=timeout_seconds,
            ),
            timeout_seconds=timeout_seconds,
        )
        if result:
            return True
        return _fallback_latex_equal(given_answer, ground_truth)
    except Exception:
        return _fallback_latex_equal(given_answer, ground_truth)


def is_equiv(str1: str, str2: str, verbose: bool = False) -> bool:
    """Check whether two answer strings are equivalent after normalization."""
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False
    
    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def grade(model_answer: str, gt_answer: str, fast: bool = True) -> bool:
    """Compare two answers via mathd/sympy fallbacks; if fast=False, additionally try math_verify."""
    if "\\boxed" in gt_answer:
        gt_answer = extract_boxed_answer(gt_answer) or gt_answer

    correct = grade_answer_mathd(model_answer, gt_answer) or grade_answer_sympy(model_answer, gt_answer)

    if not fast and not correct:
        correct = is_latex_equal(model_answer, gt_answer)
    
    return correct


def determine_positive_negative_by_voting(
    completions_text: List[str],
    num_generations: int
) -> Tuple[torch.Tensor, int, List[bool], List[List[List[int]]], List[int], torch.Tensor, List[float], List[str]]:
    """Compute per-group consensus targets (majority answers) and per-sample agreement
    statistics from grouped rollouts via answer voting."""
    num_total = len(completions_text)
    num_groups = num_total // num_generations
    is_positive = torch.zeros(num_total, dtype=torch.bool)
    is_non_extractable = torch.zeros(num_total, dtype=torch.bool)

    all_extracted: List[Tuple[int, str]] = []
    
    for i, comp in enumerate(completions_text):
        answer = extract_answer_string(comp)
        if answer is None:
            is_non_extractable[i] = True
        else:
            all_extracted.append((i, answer))

    # Normalize in one batch to avoid per-call IPC overhead.
    if all_extracted:
        raw_answers = [ans for _, ans in all_extracted]
        unique_raw_answers = list(set(raw_answers))

        normalized_results = batch_simplify_expression_strings(
            unique_raw_answers,
            timeout_seconds=1.0,
            chunk_timeout_seconds=50.0,
        )

        raw_to_normalized = dict(zip(unique_raw_answers, normalized_results))

        all_extracted_normalized = [
            (global_idx, raw_to_normalized[raw_ans])
            for global_idx, raw_ans in all_extracted
        ]
    else:
        all_extracted_normalized = []

    num_successful_votes = 0
    group_voting_success = []
    group_components = []
    group_num_extractable = []
    group_majority_ratios = []
    majority_gt_list = []

    idx_to_normalized: Dict[int, str] = {
        global_idx: norm_ans for global_idx, norm_ans in all_extracted_normalized
    }
    
    for group_idx in range(num_groups):
        start_idx = group_idx * num_generations

        extracted_answers = []  # (local_idx, normalized_answer)
        for local_idx in range(num_generations):
            global_idx = start_idx + local_idx
            if global_idx in idx_to_normalized:
                extracted_answers.append((local_idx, idx_to_normalized[global_idx]))
        
        num_extractable = len(extracted_answers)
        group_num_extractable.append(num_extractable)
        
        if num_extractable == 0:
            group_voting_success.append(False)
            group_components.append([])
            group_majority_ratios.append(0.0)
            majority_gt_list.append("None")
            continue

        answer_to_indices = defaultdict(list)
        for local_idx, norm_answer in extracted_answers:
            answer_to_indices[norm_answer].append(local_idx)

        all_components = [indices for indices in answer_to_indices.values()]
        group_components.append(all_components)

        counter = Counter([norm_answer for _, norm_answer in extracted_answers])
        winner_answer, majority_count = counter.most_common(1)[0]

        majority_ratio = majority_count / num_generations
        group_majority_ratios.append(majority_ratio)

        num_successful_votes += 1
        group_voting_success.append(True)

        for local_idx in answer_to_indices[winner_answer]:
            is_positive[start_idx + local_idx] = True

        majority_gt_list.append(winner_answer)
    
    return (is_positive, num_successful_votes, group_voting_success, group_components, 
            group_num_extractable, is_non_extractable, group_majority_ratios, majority_gt_list)

