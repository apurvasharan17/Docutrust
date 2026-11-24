"""Tests for FieldRecord JSONL round-trip (schema.py)."""
import json

from docutrust.schema import FieldRecord


def make_record():
    return FieldRecord(
        doc_id="doc_001", field_name="total_amount", ocr_text_sha256="abc123",
        condition="blur_r2.0", raw_value="2627.00", token_indices=[4, 5, 6],
        logprob_geo=0.999, ocr_conf_min=14.0, grounded=True,
        gold_value="2627.00", label="CORRECT", is_correct=True,
        model="qwen2.5:7b-instruct", ollama_version="0.33.2",
    )


def test_jsonl_round_trip():
    rec = make_record()
    line = json.dumps(rec.to_dict(), ensure_ascii=False) 
    back = FieldRecord.from_dict(json.loads(line))
    assert back == rec


def test_missing_signal_stays_none_not_zero():
    rec = make_record()
    back = FieldRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
    assert back.self_consistency is None
    assert back.engine_agreement is None


def test_rupee_text_survives_round_trip():
    rec = make_record()
    rec.raw_value = "₹2,627.00"
    back = FieldRecord.from_dict(json.loads(json.dumps(rec.to_dict(), ensure_ascii=False)))
    assert back.raw_value == "₹2,627.00"
