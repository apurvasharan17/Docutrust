"""LLM stage: send cached OCR text to Qwen (via Ollama) and save raw output.

Runs on Kaggle (T4).

1) Main run (greedy, temperature 0, with logprobs)  -> the answer we score
     run_batch("<cond>_ocr.jsonl", "<cond>_llm.jsonl")
2) Self-consistency run (n samples at temperature > 0, fixed seeds)
     run_samples("<cond>_ocr.jsonl", "<cond>_samples.jsonl")
   Later, on the Mac, each field's self_consistency = share of samples that
   agree with the greedy answer.

Both are resumable: docs already in the output file are skipped, so a Kaggle
disconnect never costs you finished work.
"""
import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:7b-instruct"

FIELDS = ["merchant_name", "invoice_date", "invoice_number",
          "total_amount", "cgst_amount", "sgst_amount"]

PROMPT_TEMPLATE = """Extract fields from this OCR text of an Indian retail receipt.
The OCR is noisy and may contain errors. Do not correct or guess.
If a field is not present in the text, return null for it.

Return ONLY a JSON object with exactly these keys:
{fields}

OCR TEXT:
{ocr_text}
"""

REQUEST_OPTIONS = {
    "stream": False,
    "format": "json",      # fix: no ```json fences
    "logprobs": True,
    "top_logprobs": 5,
    "options": {"temperature": 0},
}

# self-consistency: same prompt, sampling on, no logprobs needed
SAMPLE_TEMPERATURE = 0.7
SAMPLE_N = 5


def start_server(timeout=30):
    """Start `ollama serve` in the background and wait until it answers."""
    subprocess.Popen(["ollama", "serve"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(timeout):
        try:
            return get_version()
        except requests.RequestException:
            time.sleep(1)
    raise RuntimeError("ollama server did not start")


def get_version():
    return requests.get(f"{OLLAMA_URL}/api/version", timeout=2).json()["version"]


def build_prompt(ocr_text):
    return PROMPT_TEMPLATE.format(fields=json.dumps(FIELDS), ocr_text=ocr_text)


def extract(ocr_text):
    """One greedy LLM call. Returns response text + token logprobs + timing."""
    body = {"model": MODEL, "prompt": build_prompt(ocr_text), **REQUEST_OPTIONS}
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=300)
    r.raise_for_status()
    d = r.json()

    # integrity check: token bytes must rebuild the exact response
    rebuilt = b"".join(bytes(t["bytes"]) for t in d["logprobs"]).decode()
    assert rebuilt == d["response"], "byte reconstruction FAILED"

    return {
        "response": d["response"],
        "logprobs": d["logprobs"],
        "eval_count": d["eval_count"],
        "eval_duration": d["eval_duration"],
    }


def sample(ocr_text, seed, temperature=SAMPLE_TEMPERATURE):
    """One sampled LLM call. A fixed seed makes every sample reproducible."""
    body = {
        "model": MODEL,
        "prompt": build_prompt(ocr_text),
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "seed": seed},
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=300)
    r.raise_for_status()
    return r.json()["response"]


def already_done(out_path):
    if not os.path.exists(out_path):
        return set()
    with open(out_path) as f:
        return {json.loads(line)["doc_id"] for line in f if line.strip()}


def load_todo(in_path, out_path):
    done = already_done(out_path)
    with open(in_path) as f:
        docs = [json.loads(line) for line in f if line.strip()]
    todo = [d for d in docs if d["doc_id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go")
    return todo


def run_batch(in_path, out_path, gpu="Tesla T4", kv_cache="f16"):
    """Greedy run: one answer per doc, with logprobs."""
    version = get_version()
    todo = load_todo(in_path, out_path)

    with open(out_path, "a") as out:
        for i, doc in enumerate(todo, 1):
            t0 = time.time()
            result = extract(doc["ocr_text"])
            record = {
                "doc_id": doc["doc_id"],
                "model": MODEL,
                "ollama_version": version,
                "gpu": gpu,
                "kv_cache": kv_cache,
                "prompt": build_prompt(doc["ocr_text"]),
                "request_options": REQUEST_OPTIONS,
                "ocr_text": doc["ocr_text"],
                **result,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()  # saved immediately, survives a crash
            print(f"[{i}/{len(todo)}] {doc['doc_id']} {time.time() - t0:.1f}s")


def run_samples(in_path, out_path, n=SAMPLE_N, temperature=SAMPLE_TEMPERATURE,
                workers=SAMPLE_N, gpu="Tesla T4", kv_cache="f16"):
    """Self-consistency run: n sampled answers per doc, sent in parallel.

    Needs OLLAMA_NUM_PARALLEL >= workers set BEFORE start_server(),
    otherwise Ollama queues the requests and there is no speed-up.
    """
    version = get_version()
    todo = load_todo(in_path, out_path)
    seeds = list(range(1, n + 1))

    with open(out_path, "a") as out, ThreadPoolExecutor(workers) as pool:
        for i, doc in enumerate(todo, 1):
            t0 = time.time()
            text = doc["ocr_text"]
            responses = list(pool.map(lambda s: sample(text, s, temperature), seeds))
            record = {
                "doc_id": doc["doc_id"],
                "model": MODEL,
                "ollama_version": version,
                "gpu": gpu,
                "kv_cache": kv_cache,
                "temperature": temperature,
                "seeds": seeds,
                "responses": responses,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"[{i}/{len(todo)}] {doc['doc_id']} {time.time() - t0:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--samples", action="store_true",
                   help="self-consistency run instead of the greedy run")
    args = p.parse_args()
    if args.samples:
        run_samples(args.in_path, args.out_path)
    else:
        run_batch(args.in_path, args.out_path)