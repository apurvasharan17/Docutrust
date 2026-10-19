"""Self-consistency signal: does the model give the same answer when asked again?

For each document we have one greedy answer (temperature 0, the one we score)
and n sampled answers (temperature 0.7, fixed seeds). For each field:

  self_consistency = share of samples whose value agrees with the greedy value

  1.0 -> every sample gave the same answer       (model is stable)
  0.4 -> only 2 of 5 samples agreed               (model is guessing)

It is defined for null answers too: if the greedy answer is null but some
samples return a real value, the score drops, flagging a "confident null".
"""
from docutrust.signals.logprob import FIELDS, parse_values
from docutrust.signals.ocr import same, value_words


def canon(value):
    """Normalised form used for comparison; None stays None."""
    if value is None:
        return None
    return " ".join(value_words(value))


def agree(a, b):
    """True if two extracted values mean the same thing."""
    ca, cb = canon(a), canon(b)
    if ca is None or cb is None:
        return ca is None and cb is None      # both null -> agree
    return same(ca, cb)                       # exact, or equal as numbers


def consistency_signals(greedy_values, sample_responses):
    """greedy_values: {field: str or None}; sample_responses: list of raw JSON strings.

    Returns {field: score in [0, 1]}, or {field: None} if there are no samples.
    """
    samples = [parse_values(r) for r in sample_responses]
    if not samples:
        return {f: None for f in FIELDS}
    out = {}
    for field in FIELDS:
        g = greedy_values.get(field)
        hits = sum(agree(g, s.get(field)) for s in samples)
        out[field] = hits / len(samples)
    return out