# """
# Rubric/prompt construction and robust JSON parsing of VLM responses.

# The prompt text below deliberately mirrors your exact human-evaluator
# instructions (primary basis = textual description, reference image is
# contextual only, focus on structural/conceptual correctness not style),
# combined with the detailed sub-criteria you gave for each of the three
# dimensions.
# """

# import json
# import re

# SYSTEM_PROMPT = (
#     "You are an expert biology educator evaluating AI-generated educational "
#     "biology images. You are acting as a substitute for a human expert "
#     "evaluator, following the exact protocol below. Be strict, consistent, "
#     "and base your judgment primarily on scientific/pedagogical correctness, "
#     "not aesthetic preference."
# )

# RUBRIC_TEMPLATE = """\
# The objective of this evaluation is to assess an AI-generated biological \
# image intended for educational purposes. You are given a textual description \
# of a biological concept or structure, a reference image, and a generated image. \
# Assess how effectively the generated image represents the given description.

# Treat the textual description as the PRIMARY basis for your evaluation. The \
# reference image is included only for contextual comparison and represents \
# just one of many possible ways to visualize the concept — do not penalize the \
# generated image merely for looking different from the reference.

# Focus your assessment on the structural and conceptual representation of the \
# biological concept, not stylistic preferences (color palette, art style, etc.).

# Concept description:
# \"\"\"{description}\"\"\"

# Score the generated image from 1 (very poor) to 10 (excellent) on each of the \
# following three dimensions:

# 1. Scientific Accuracy
#    - Correctness of biological structures
#    - Correctness of relationships between structures
#    - Presence of important components
#    - Absence of biological hallucinations/errors

# 2. Visual Clarity
#    - Diagrammatic coherence and structural integrity
#    - Distinguishability of structures
#    - Readability and educational usefulness
#    - Absence of visual artifacts

# 3. Complexity Alignment
#    - Whether the level of visual DETAIL is appropriate for the concept described
#    - Whether the image is overly simplified or unnecessarily complex
#    - Whether it balances completeness and clarity
#    - This dimension concerns ONLY the appropriateness of visual detail — do \
# NOT factor scientific correctness or clarity into this score.

# Evaluate this image independently of any other image you may have seen.

# Respond with ONLY a single valid JSON object and nothing else — no markdown \
# fences, no preamble, no explanation outside the JSON:

# {{
#   "scientific_accuracy": <integer 1-10>,
#   "visual_clarity": <integer 1-10>,
#   "complexity_alignment": <integer 1-10>,
#   "reasoning": "<2-4 sentences justifying all three scores>"
# }}
# """


# def build_user_prompt(description: str) -> str:
#     return RUBRIC_TEMPLATE.format(description=description.strip())


# _JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
# _REQUIRED_KEYS = ("scientific_accuracy", "visual_clarity", "complexity_alignment")


# class ResponseParseError(Exception):
#     pass


# def parse_vlm_json(raw_text: str) -> dict:
#     """
#     Robustly extract the scoring JSON from a raw VLM text response.
#     Raises ResponseParseError if it cannot be recovered.
#     """
#     text = raw_text.strip()

#     # Strip markdown code fences if present
#     text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
#     text = re.sub(r"```$", "", text.strip()).strip()

#     candidates = [text]
#     match = _JSON_BLOCK_RE.search(text)
#     if match:
#         candidates.append(match.group(0))

#     last_err = None
#     for cand in candidates:
#         try:
#             obj = json.loads(cand)
#         except json.JSONDecodeError as e:
#             last_err = e
#             continue

#         missing = [k for k in _REQUIRED_KEYS if k not in obj]
#         if missing:
#             last_err = ValueError(f"missing keys: {missing}")
#             continue

#         try:
#             for k in _REQUIRED_KEYS:
#                 v = float(obj[k])
#                 if not (1 <= v <= 10):
#                     raise ValueError(f"{k}={v} out of range 1-10")
#                 obj[k] = v
#         except (TypeError, ValueError) as e:
#             last_err = e
#             continue

#         obj["reasoning"] = str(obj.get("reasoning", "")).strip()
#         obj["overall_score"] = round(
#             (obj["scientific_accuracy"] + obj["visual_clarity"] + obj["complexity_alignment"]) / 3,
#             3,
#         )
#         return obj

#     raise ResponseParseError(f"Could not parse JSON from response: {last_err}\nRaw: {raw_text[:500]}")

"""
Rubric/prompt construction and robust JSON parsing of VLM responses.

The prompt text below deliberately mirrors your exact human-evaluator
instructions (primary basis = textual description, reference image is
contextual only, focus on structural/conceptual correctness not style),
combined with the detailed sub-criteria you gave for each of the three
dimensions.
"""

import json
import re

SYSTEM_PROMPT = (
    "You are an expert biology educator evaluating AI-generated educational "
    "biology images. You are acting as a substitute for a human expert "
    "evaluator, following the exact protocol below. Be strict, consistent, "
    "and base your judgment primarily on scientific/pedagogical correctness, "
    "not aesthetic preference."
)

RUBRIC_TEMPLATE = """\
The objective of this evaluation is to assess an AI-generated biological \
image intended for educational purposes. You are given a textual description \
of a biological concept or structure, a reference image, and a generated image. \
Assess how effectively the generated image represents the given description.

Treat the textual description as the PRIMARY basis for your evaluation. The \
reference image is included only for contextual comparison and represents \
just one of many possible ways to visualize the concept — do not penalize the \
generated image merely for looking different from the reference.

Focus your assessment on the structural and conceptual representation of the \
biological concept, not stylistic preferences (color palette, art style, etc.).

Concept description:
\"\"\"{description}\"\"\"

Score the generated image from 1 (very poor) to 10 (excellent) on each of the \
following three dimensions:

1. Scientific Accuracy
   - Correctness of biological structures
   - Correctness of relationships between structures
   - Presence of important components
   - Absence of biological hallucinations/errors

2. Visual Clarity
   - Diagrammatic coherence and structural integrity
   - Distinguishability of structures
   - Readability and educational usefulness
   - Absence of visual artifacts

3. Complexity Alignment
   - Whether the level of visual DETAIL is appropriate for the concept described
   - Whether the image is overly simplified or unnecessarily complex
   - Whether it balances completeness and clarity
   - This dimension concerns ONLY the appropriateness of visual detail — do \
NOT factor scientific correctness or clarity into this score.

Evaluate this image independently of any other image you may have seen.

Respond with ONLY a single valid JSON object and nothing else — no markdown \
fences, no preamble, no explanation outside the JSON:

{{
  "scientific_accuracy": <integer 1-10>,
  "visual_clarity": <integer 1-10>,
  "complexity_alignment": <integer 1-10>,
  "reasoning": "<2-4 sentences justifying all three scores>"
}}
"""


def build_user_prompt(description: str) -> str:
    return RUBRIC_TEMPLATE.format(description=description.strip())


_REQUIRED_KEYS = ("scientific_accuracy", "visual_clarity", "complexity_alignment")


class ResponseParseError(Exception):
    pass


def _extract_first_json_object(text: str):
    r"""
    Find and parse the FIRST complete, balanced JSON object in text, ignoring
    anything that follows it (e.g. a model that keeps generating after the
    JSON and trails off into unrelated content). Returns the parsed dict, or
    None if no valid JSON object could be found.

    This deliberately does NOT use a greedy regex like r"\{.*\}" — that
    matches from the first '{' to the LAST '}' in the whole text, which
    swallows any trailing garbage (including any stray braces in it) as
    part of the "JSON" and then fails to parse.
    """
    decoder = json.JSONDecoder()
    start = 0
    while True:
        idx = text.find("{", start)
        if idx == -1:
            return None
        try:
            obj, _end = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = idx + 1


def parse_vlm_json(raw_text: str) -> dict:
    """
    Robustly extract the scoring JSON from a raw VLM text response.
    Raises ResponseParseError if it cannot be recovered.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    candidates = []

    # Fast path: the whole response is exactly one JSON object, nothing else.
    try:
        candidates.append(json.loads(text))
    except json.JSONDecodeError:
        pass

    # Robust path: find the first complete balanced JSON object anywhere in
    # the text, ignoring anything before/after it (handles models that add
    # preamble, or keep generating past the JSON into unrelated content).
    extracted = _extract_first_json_object(text)
    if extracted is not None:
        candidates.append(extracted)

    last_err = None
    for obj in candidates:
        missing = [k for k in _REQUIRED_KEYS if k not in obj]
        if missing:
            last_err = ValueError(f"missing keys: {missing}")
            continue

        try:
            for k in _REQUIRED_KEYS:
                v = float(obj[k])
                if not (1 <= v <= 10):
                    raise ValueError(f"{k}={v} out of range 1-10")
                obj[k] = v
        except (TypeError, ValueError) as e:
            last_err = e
            continue

        obj["reasoning"] = str(obj.get("reasoning", "")).strip()
        obj["overall_score"] = round(
            (obj["scientific_accuracy"] + obj["visual_clarity"] + obj["complexity_alignment"]) / 3,
            3,
        )
        return obj

    raise ResponseParseError(f"Could not parse JSON from response: {last_err}\nRaw: {raw_text[:500]}")