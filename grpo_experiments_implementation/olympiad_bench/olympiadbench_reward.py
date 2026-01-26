import re

_SOLUTION_CLIP_CHARS = 300

# #### <number> (e.g., "#### 1,234.5")
_RE_HASHES_NUM = re.compile(r"####\s*([\-0-9\.,]+)")

# \boxed{ANSWER} (optionally wrapped in $ or $$)
_RE_BOXED = re.compile(
    r"(?:\${1,2}\s*)?oxed\s*\{\s*([^}]*)\s*\}(?:\s*\${1,2})?",
    flags=re.IGNORECASE,
)

def _normalize_num(s: str) -> str:
    """Light cleanup: remove commas and dollar signs; strip whitespace."""
    return s.strip().replace(",", "").replace("$", "")

def extract_after_think_hash_then_boxed(text: str):
    """
    If </think> is present:
      1) try '#### <number>' after it
      2) else try '\boxed{...}' after it
    Return the extracted string (normalized) or None.
    """

    tail = text if len(text) <= _SOLUTION_CLIP_CHARS else text[-_SOLUTION_CLIP_CHARS:]

    # 1) #### <number>
    hits = _RE_HASHES_NUM.findall(tail)
    if hits:
        return _normalize_num(hits[-1])

    # 2) \boxed{...}
    boxed_hits = _RE_BOXED.findall(tail)
    if boxed_hits:
        return _normalize_num(boxed_hits[-1])

    return None


def my_reward_fn(data_source=None, solution_str=None, ground_truth=None, extra_info=None):

    ans = extract_after_think_hash_then_boxed(solution_str or "")
    if ans is None:
        return 0.0
    
    final_reward = 1.0 if ground_truth is not None and str(ans) == str(ground_truth) else 0.0
    return final_reward


if __name__ == '__main__':

    sol = '''</think> Natalia sold 48/2 = <<48/2=24 ###4.95 $$
\boxed{64}
$$
'''
    print(extract_after_think_hash_then_boxed(sol))