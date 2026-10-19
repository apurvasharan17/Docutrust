"""Validation signal: does the extracted value pass simple business rules?

validation_passed, per field:
  True  -> value looks valid, or the field is null AND its label is not on
           the document either (genuinely absent)
  False -> a rule is broken: bad date, CGST != SGST, total too small, ...
           or the value is null while the field's label IS visible in the
           OCR text (a "confident null": the LLM gave up on a field that exists)
  None  -> no rule applies

Rules are kept general (not tied to the synthetic template) so they also
work on real receipts: "Grand Total", "C.G.S.T", "Invoice Number", etc.
"""
import re
from datetime import datetime

from docutrust.signals.logprob import FIELDS

# label patterns that mean "this field is printed on the document"
LABELS = {
    "invoice_date":   r"\bdate\b",
    "invoice_number": r"\binv(oice)?\.?\s*(no|number|#)",
    "total_amount":   r"\btotal\b",
    "cgst_amount":    r"\bc\.?\s*g\.?\s*s\.?\s*t\b",
    "sgst_amount":    r"\bs\.?\s*g\.?\s*s\.?\s*t\b",
}

DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d/%m/%y"]


def to_amount(v):
    """'₹1,250.00' -> 1250.0 ; anything unparseable -> None."""
    if v is None:
        return None
    try:
        return float(v.replace("₹", "").replace(",", "").strip())
    except ValueError:
        return None


def valid_date(v):
    for fmt in DATE_FORMATS:
        try:
            d = datetime.strptime(v.strip(), fmt)
            return 2000 <= d.year <= 2035
        except ValueError:
            pass
    return False


def valid_invoice_no(v):
    """One token (letters, digits, / or -) with at least one digit."""
    v = v.strip()
    return bool(re.fullmatch(r"[A-Za-z0-9/\-]+", v)) and any(ch.isdigit() for ch in v)


def valid_merchant(v):
    """At least 3 characters, mostly letters."""
    chars = v.replace(" ", "")
    letters = sum(ch.isalpha() for ch in chars)
    return len(chars) >= 3 and letters >= 0.6 * len(chars)


def check_value(field, v, amounts):
    """Rule check for a field that HAS a value."""
    if field == "merchant_name":
        return valid_merchant(v)
    if field == "invoice_date":
        return valid_date(v)
    if field == "invoice_number":
        return valid_invoice_no(v)

    amt = to_amount(v)
    if amt is None or amt < 0:
        return False
    cgst, sgst = amounts["cgst_amount"], amounts["sgst_amount"]

    if field == "total_amount":
        if cgst is None or sgst is None:
            return True                      # nothing to compare against
        return amt > cgst + sgst

    # cgst / sgst: equal for intra-state GST
    other = sgst if field == "cgst_amount" else cgst
    return other is None or abs(amt - other) < 0.01


def check_null(field, text):
    """Rule check for a field the LLM returned as null."""
    if field == "merchant_name":
        return False                         # every bill names the shop
    return re.search(LABELS[field], text) is None   # label visible -> suspicious


def validation_signals(values, ocr_text):
    """values: {field: str or None}  ->  {field: True / False / None}"""
    text = ocr_text.lower()
    amounts = {f: to_amount(values.get(f))
               for f in ("total_amount", "cgst_amount", "sgst_amount")}
    out = {}
    for field in FIELDS:
        v = values.get(field)
        if v is None:
            out[field] = check_null(field, text)
        else:
            out[field] = check_value(field, v, amounts)
    return out