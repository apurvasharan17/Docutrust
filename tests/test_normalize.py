"""Tests for the CORRECT / WRONG / MISS labelling policy (normalize.py)."""
import pytest

from docutrust.normalize import label, normalize_amount, normalize_text


# ---------- labelling policy: null handling ----------

def test_both_null_is_correct():
    # field absent in gold and model says null -> correct
    assert label(None, None, "invoice_number") == "CORRECT"


def test_gold_present_predicted_null_is_miss():
    assert label(None, "2627.00", "total_amount") == "MISS"


def test_gold_null_predicted_value_is_wrong():
    # model invented a value that is not on the document
    assert label("123.45", None, "cgst_amount") == "WRONG"


# ---------- amounts: compared as numbers ----------

@pytest.mark.parametrize("predicted, gold", [
    ("₹2,627.00", "2627.00"),   # rupee sign and comma stripped
    ("2627", "2627.00"),        # 2627 == 2627.00 numerically
    ("Rs. 123.98", "123.98"),   # "Rs." prefix
    ("INR 1,23,456", "123456"), # Indian digit grouping
])
def test_amount_formats_match(predicted, gold):
    assert label(predicted, gold, "total_amount") == "CORRECT"


def test_wrong_amount_is_wrong():
    # the real DMart receipt case: model read 34.87, true CGST is 123.98
    assert label("34.87", "123.98", "cgst_amount") == "WRONG"


def test_unparseable_amount_is_wrong_not_crash():
    assert normalize_amount("abc") is None
    assert label("abc", "100.00", "total_amount") == "WRONG"


# ---------- text: case- and whitespace-insensitive ----------

def test_text_case_insensitive():
    assert label("dmart banashankari", "DMART BANASHANKARI", "merchant_name") == "CORRECT"


def test_text_extra_spaces_ignored():
    assert normalize_text("  DMART   Banashankari ") == "dmart banashankari"


def test_different_text_is_wrong():
    assert label("Chtynappa Naidu", "DMART BANASHANKARI", "merchant_name") == "WRONG"
