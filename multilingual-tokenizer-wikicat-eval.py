#!/usr/bin/env python3

import argparse
import io
import json
import os
import sys
import warnings
import logging
from typing import Dict, Set
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# Suppress tokenizer warnings for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

tokenizers = [
    {"name": "GPT-2", "id": "gpt2"},
    {"name": "Gemma-3", "id": "google/gemma-3-4b-it"},    
    {"name": "SmolLM3", "id": "HuggingFaceTB/SmolLM3-3B"},
    {"name": "GPT-OSS", "id": "openai/gpt-oss-20b"},
    {"name": "Kimi-K2", "id": "moonshotai/Kimi-K2-Instruct"},
    {"name": "Aya-Expanse", "id": "CohereLabs/aya-expanse-8b"},
    {"name": "Qwen-3-VL", "id": "Qwen/Qwen3-VL-2B-Instruct"},
    {"name": "DeepSeek-3.2", "id": "deepseek-ai/DeepSeek-V3.2-Exp"},
    {"name": "Nemotron-v2", "id": "nvidia/NVIDIA-Nemotron-Nano-12B-v2"},
    {"name": "Llama-4", "id": "meta-llama/Llama-4-Scout-17B-16E-Instruct"},
    {"name": "MiniMax-M2", "id": "MiniMaxAI/MiniMax-M2"},
    {"name": "Granite-4", "id": "ibm-granite/granite-4.0-h-1b"},
]


def download_dataset(eval_dir: str) -> str:
    """Download the dataset from HuggingFace if not already present."""
    local_path = os.path.join(eval_dir, "wikipedia_cat-272_languages-utf8-with-text.jsonl")
    
    if os.path.exists(local_path):
        print(f"Dataset already exists at: {local_path}")
        return local_path
    
    print("Downloading dataset from HuggingFace...")
    downloaded_path = hf_hub_download(
        repo_id="Norod78/WikiCat-Multilingual",
        filename="wikipedia_cat-272_languages-utf8-with-text.jsonl",
        repo_type="dataset",
        local_dir=eval_dir,
        local_dir_use_symlinks=False
    )
    print(f"Dataset downloaded to: {downloaded_path}")
    return downloaded_path


def load_tokenizers():
    """Load all tokenizers."""
    print("Loading tokenizers...")
    loaded_tokenizers = []
    for tokenizer_info in tokenizers:
        print(f"  Loading {tokenizer_info['name']}...")
        # Suppress stdout during tokenizer loading
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_info["id"], trust_remote_code=True, verbosity=0
            )
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        loaded_tokenizers.append({"name": tokenizer_info["name"], "tokenizer": tokenizer})
    print("All tokenizers loaded.")
    return loaded_tokenizers


def tokenize_text(tokenizer, text: str) -> int:
    """Tokenize text and return token count."""
    # Suppress stdout/stderr during tokenization to avoid verbose logging
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    try:
        encoded = tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt"
        )
        result = encoded.size()[-1]
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return result


def load_processed_titles(output_path: str) -> Set[str]:
    """Load already processed titles for resumability."""
    processed: Set[str] = set()
    if not os.path.exists(output_path):
        return processed
    
    with io.open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and 'title' in obj and 'lang' in obj:
                    # Use combination of lang and title as unique key
                    key = f"{obj['lang']}::{obj['title']}"
                    processed.add(key)
            except json.JSONDecodeError:
                continue
    return processed


def write_jsonl_line(fp, obj: Dict) -> None:
    """Write a single JSONL line with flush and fsync for safety."""
    json_line = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
    fp.write(json_line + "\n")
    fp.flush()
    os.fsync(fp.fileno())


def process_dataset(
    input_path: str,
    output_path: str,
    loaded_tokenizers,
    start_from_scratch: bool = False,
) -> None:
    """Process the dataset and generate tokenizer evaluation metrics."""
    
    # Load already processed entries for resumability
    processed_keys: Set[str] = set()
    if not start_from_scratch:
        processed_keys = load_processed_titles(output_path)
        print(f"Resuming: {len(processed_keys)} entries already processed.")
    
    # Open output in append mode (or write mode if starting fresh)
    out_mode = 'w' if start_from_scratch else 'a'
    
    completed = 0
    skipped = 0
    errors = 0
    
    with io.open(output_path, out_mode, encoding='utf-8') as out_fp:
        with io.open(input_path, 'r', encoding='utf-8') as in_fp:
            for line_idx, raw_line in enumerate(in_fp, 1):
                line = raw_line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[{line_idx}] JSON parse error: {e}", file=sys.stderr)
                    errors += 1
                    continue
                
                if not isinstance(obj, dict):
                    print(f"[{line_idx}] Not a dict", file=sys.stderr)
                    errors += 1
                    continue
                
                # Check required fields
                required_fields = ['lang', 'langname', 'autonym', 'title', 'text']
                missing = [f for f in required_fields if f not in obj]
                if missing:
                    print(f"[{line_idx}] Missing fields: {missing}", file=sys.stderr)
                    errors += 1
                    continue
                
                # Check if already processed
                key = f"{obj['lang']}::{obj['title']}"
                if key in processed_keys:
                    skipped += 1
                    continue
                
                # Extract text and calculate basic metrics
                text = str(obj['text'])
                character_count = len(text)
                word_count = len([w for w in text.split() if w.strip()])
                
                # Build output record with copied fields
                output_record = {
                    'lang': obj['lang'],
                    'langname': obj['langname'],
                    'autonym': obj['autonym'],
                    'title': obj['title'],
                    'character_count': character_count,
                    'word_count': word_count,
                }
                
                # Process with each tokenizer
                try:
                    wpt_scores = {}  # Track WPT for finding best tokenizer
                    
                    for tokenizer_info in loaded_tokenizers:
                        name = tokenizer_info['name']
                        tokenizer = tokenizer_info['tokenizer']
                        
                        token_count = tokenize_text(tokenizer, text)
                        
                        # Calculate ratios (avoid division by zero)
                        cpt = character_count / token_count if token_count > 0 else 0
                        wpt = word_count / token_count if token_count > 0 else 0
                        
                        # Add to output record
                        output_record[f"{name}_token_count"] = token_count
                        output_record[f"{name}_cpt"] = round(cpt, 3)
                        output_record[f"{name}_wpt"] = round(wpt, 3)
                        
                        # Track WPT for best tokenizer determination
                        wpt_scores[name] = wpt
                    
                    # Find best tokenizer based on WPT (highest is best)
                    best_tokenizer = max(wpt_scores, key=wpt_scores.get)
                    output_record['best_tokenizer'] = best_tokenizer
                    
                    # Write to output
                    write_jsonl_line(out_fp, output_record)
                    processed_keys.add(key)
                    completed += 1
                    
                    if completed % 10 == 0:
                        print(f"[{line_idx}] Processed {completed} entries...")
                    
                except Exception as e:
                    print(f"[{line_idx}] Error processing {key}: {e}", file=sys.stderr)
                    errors += 1
                    continue
    
    print(f"\nDone. Completed: {completed}, Skipped: {skipped}, Errors: {errors}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate multilingual tokenizers on WikiCat dataset'
    )
    parser.add_argument(
        '--input', '-i',
        help='Path to input JSONL file (default: auto-download to eval/)'
    )
    parser.add_argument(
        '--output', '-o',
        default='wikicat-tokenizer-eval.jsonl',
        help='Path to output JSONL file'
    )
    parser.add_argument(
        '--eval-dir',
        default='.',
        help='Directory for eval files (default: current directory)'
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Start from scratch and overwrite output'
    )
    args = parser.parse_args()
    
    # Ensure eval directory exists
    os.makedirs(args.eval_dir, exist_ok=True)
    
    # Determine input path
    if args.input:
        input_path = args.input
    else:
        input_path = download_dataset(args.eval_dir)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load tokenizers
    loaded_tokenizers = load_tokenizers()
    
    # Process dataset
    output_path = os.path.join(args.eval_dir, args.output)
    try:
        process_dataset(
            input_path,
            output_path,
            loaded_tokenizers,
            start_from_scratch=args.fresh
        )
        print(f"\nOutput written to: {output_path}")
    except KeyboardInterrupt:
        print('\n\nInterrupted. Progress is saved; you can resume by re-running.', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
