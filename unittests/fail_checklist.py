# unittests/fail_checklist.py
import re
from copy import deepcopy

from sudregex import checklist_abc as base

# Start from the real checklist so we keep exactly the same format/keys.
checklist = deepcopy(base)

# --- Intentional changes to cause mismatches, while preserving format ---

# 20: SO_concern – make it broader (will create false positives)
# Original matches family member + concern near analgesics; we broaden to any "concern"
# so rows without the opioid context may now match.
checklist["20"] = {
    **checklist["20"],
    "pat": re.compile(r"(?i)\bconcern(ed)?\b"),
}

# 11b: strong_preference_IV – make it too strict (will create false negatives)
# Require the exact phrase "prefers to take IV" instead of your broader pattern.
checklist["11b"] = {
    **checklist["11b"],
    "pat": re.compile(r"(?i)\bprefers to take\s+iv\b"),
}

# 1b: problem_drinking – remove common false-positive filters (will create false positives)
if "common_fp" in checklist["1b"]:
    checklist["1b"] = {**checklist["1b"], "common_fp": []}

# 6: prn – flip negation handling (may hide some positives that used to pass)
# If the base had negation=False, set to True so negated contexts zero it out.
checklist["6"] = {**checklist["6"], "negation": True}

# 8: bought_on_street – slightly broaden (add "street" anywhere)
checklist["8"] = {
    **checklist["8"],
    "pat": re.compile(r"(?i)\bstreet\b"),
}

# 18a: minimal_relief_x – tweak so it fires more (allow "not much relief" loosely)
checklist["18a"] = {
    **checklist["18a"],
    "pat": re.compile(r"(?i)(not( much)?|minimal|limited).{0,30}relief"),
}
