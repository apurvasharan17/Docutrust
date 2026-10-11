"""Logprob signals: turn one LLM response into per-field confidence scores.

For each field we find the tokens that produced its VALUE (not the key,
not the quote marks) and aggregate their probabilities 4 ways.
All confidences are stored as probabilities in [0, 1]:

  logprob_geo   = exp(mean of logprobs)        geometric mean prob
  logprob_mean  = mean of exp(logprob)         arithmetic mean prob
  logprob_min   = exp(min logprob)             weakest token
  logprob_first = exp(first logprob)           first token only
  margin_mean / margin_min = top1 prob - top2 prob, per token
"""
import hashlib
import json
import math
import re

FIELDS = ["merchant_name", "invoice_date", "invoice_number",
          "total_amount", "cgst_amount", "sgst_amount"]

# value can be: null | "string" | number
VALUE_RE = r'\s*:\s*(?:(null)|"((?:[^"\\]|\\.)*)"|(-?\d+(?:\.\d+)?))'


def token_spans(logprobs):
    """Byte span (start, end) of every output token."""
    spans, pos = [], 0
    for t in logprobs:
        n = len(bytes(t["bytes"]))
        spans.append((pos, pos + n))
        pos += n
    return spans


def value_char_span(response, field):
    """Character span of the field's value. Strings: inside the quotes only."""
    m = re.search(rf'"{re.escape(field)}"' + VALUE_RE, response)
    if not m:
        return None
    for g in (2, 3, 1):  # string content, number, null
        if m.group(g) is not None:
            s, e = m.span(g)
            if s == e:           # empty string "" -> fall back to the quotes
                s, e = s - 1, e + 1
            return s, e
    return None


def margin(token):
    """top-1 probability minus top-2 probability for one token."""
    alts = sorted((a["logprob"] for a in token.get("top_logprobs", [])), reverse=True)
    if len(alts) < 2:
        return None
    return math.exp(alts[0]) - math.exp(alts[1])


def field_signals(response, logprobs, field):
    span = value_char_span(response, field)
    if span is None:
        return None
    # chars -> bytes (matters for ₹ and other non-ASCII)
    bs = len(response[:span[0]].encode())
    be = len(response[:span[1]].encode())

    idx = [i for i, (a, b) in enumerate(token_spans(logprobs)) if a < be and b > bs]
    if not idx:
        return None
    lps = [logprobs[i]["logprob"] for i in idx]
    margins = [m for m in (margin(logprobs[i]) for i in idx) if m is not None]

    return {
        "token_indices": idx,
        "logprob_geo": math.exp(sum(lps) / len(lps)),
        "logprob_mean": sum(math.exp(lp) for lp in lps) / len(lps),
        "logprob_min": math.exp(min(lps)),
        "logprob_first": math.exp(lps[0]),
        "margin_mean": sum(margins) / len(margins) if margins else None,
        "margin_min": min(margins) if margins else None,
    }


def parse_values(response):
    """Actual field values from the JSON. Numbers become strings."""
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return {}
    out = {}
    for f in FIELDS:
        v = data.get(f)
        out[f] = None if v is None else str(v)
    return out


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()