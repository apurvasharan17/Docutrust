"""OCR signals: how sure was Tesseract about the words the LLM copied?

ocr_conf_min : lowest Tesseract confidence (0-1) among the OCR words that
               make up the extracted value. None if value is null or not found.
grounded     : True if every word of the value appears in the OCR words,
               False if any word is missing (LLM changed or invented it).
               None if value is null.
"""
STRIP = ".,:;()[]{}'\"*|"


def norm(text):
    """Lowercase, drop rupee sign and commas, trim punctuation at the ends."""
    t = text.lower().replace("₹", "").replace(",", "")
    return t.strip(STRIP)


def same(a, b):
    """Exact match, or equal as numbers ("1250.0" == "1250.00")."""
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def match_conf(vword, words):
    """Best Tesseract confidence (0-1) for one value word, or None if not found."""
    best = None
    for w in words:
        ow = norm(w["word"])
        if ow and same(vword, ow):
            c = w["conf"] / 100
            best = c if best is None else max(best, c)
    return best


def ocr_signals(value, words):
    if value is None:
        return {"ocr_conf_min": None, "grounded": None}
    vwords = [v for v in (norm(x) for x in value.split()) if v]
    if not vwords:
        return {"ocr_conf_min": None, "grounded": None}

    confs = [match_conf(v, words) for v in vwords]
    found = [c for c in confs if c is not None]
    return {
        "ocr_conf_min": min(found) if found else None,
        "grounded": len(found) == len(confs),
    }