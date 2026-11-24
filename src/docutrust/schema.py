from dataclasses import dataclass,field,asdict
from typing import List, Optional

@dataclass
class TokenRecord:
    text: str
    logprob: float
    byte_start: int
    byte_end: int
    top_alternatives: list  # [(text, logprob), ...]


@dataclass
class FieldRecord:
    # identity
    doc_id: str
    field_name: str
    ocr_text_sha256: str
    condition: Optional[str] = None  # clean / blur_r1.0 / ...

    # output
    raw_value: Optional[str] = None
    token_indices: list = field(default_factory=list)

    # --- logprob signals (aggregation is an experimental variable) ---
    logprob_mean: Optional[float] = None
    logprob_min: Optional[float] = None
    logprob_geo: Optional[float] = None
    logprob_first: Optional[float] = None

    # --- margin signals (top-1 minus top-2) ---
    margin_mean: Optional[float] = None
    margin_min: Optional[float] = None

    # --- other signals (Month 2) ---
    ocr_conf_min: Optional[float] = None
    doc_ocr_conf: Optional[float] = None
    engine_agreement: Optional[float] = None
    grounded: Optional[bool] = None
    validation_passed: Optional[bool] = None
    self_consistency: Optional[float] = None
    cross_chunk_agreement: Optional[float] = None
    # --- labels ---
    gold_value: Optional[str] = None
    label: Optional[str] = None  # CORRECT / WRONG / MISS
    is_correct: Optional[bool] = None

    # --- provenance ---
    model: Optional[str] = None
    ollama_version: Optional[str] = None
    kv_cache_type: Optional[str] = None
    hardware: Optional[str] = None
    run_timestamp: Optional[str] = None
    def to_dict(self):
        ...
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)