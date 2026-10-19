"""OCR signals: how sure was the OCR about the words the LLM copied?

ocr_conf_min     : lowest Tesseract confidence (0-1) among the OCR words that
                   make up the extracted value. None if value is null or not found.
grounded         : True if every word of the value appears in the Tesseract words,
                   False if any word is missing (LLM changed or invented it).
                   None if value is null.
doc_ocr_conf     : mean Tesseract confidence (0-1) over the WHOLE document.
                   Same for every field of a doc; defined even for null values.
engine_agreement : fraction (0-1) of the value's words that a SECOND engine
                   (EasyOCR) also read. Catches cases where Tesseract misread
                   the image confidently and the LLM copied the mistake.
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


def value_words(value):
    """Split an extracted value into normalised words (empty ones dropped)."""
    return [v for v in (norm(x) for x in value.split()) if v]


def match_conf(vword, words):
    """Best OCR confidence (0-1) for one value word, or None if not found."""
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
    vwords = value_words(value)
    if not vwords:
        return {"ocr_conf_min": None, "grounded": None}

    confs = [match_conf(v, words) for v in vwords]
    found = [c for c in confs if c is not None]
    return {
        "ocr_conf_min": min(found) if found else None,
        "grounded": len(found) == len(confs),
    }


def doc_ocr_conf(words):
    """Mean Tesseract confidence (0-1) over the WHOLE document.

    Same value for every field of a doc. Unlike ocr_conf_min it is defined
    even when the LLM returned null, so it can flag 'confident nulls' on
    documents the OCR could barely read.
    """
    if not words:
        return None
    return sum(w["conf"] for w in words) / len(words) / 100


def engine_agreement(value, easy_words):
    """Fraction of the value's words that EasyOCR also read. None if value is null."""
    if value is None:
        return None
    vwords = value_words(value)
    if not vwords:
        return None
    easy = [norm(w["word"]) for w in easy_words]
    hits = sum(any(e and same(v, e) for e in easy) for v in vwords)
    return hits / len(vwords)