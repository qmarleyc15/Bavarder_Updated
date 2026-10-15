"""
Build context-window training examples from TCOF .trs files.

Each example = the previous few turns (children and adults, in order) plus the
adult reply that followed. Reuses your existing parser class (TRSExtractor), so
put this file in the same folder as your parser, and pass the parser's filename
without ".py" as --parser-module.

Usage:
    python build_context_pairs.py path/to/trs_folder --parser-module your_parser_filename
"""

import argparse
import importlib
import json
import re
from collections import Counter
from pathlib import Path


def words(text):
    return re.findall(r"\w+", text.lower())


def overlap(child, adult):
    """Fraction of the adult's words that also appear in the child's line."""
    c, a = set(words(child)), words(adult)
    return sum(w in c for w in a) / len(a) if a else 0.0


def grams(text, n=5):
    w = text.lower().split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def clip(text, max_words):
    """Keep the last max_words words of a long turn."""
    return " ".join(text.split()[-max_words:])


def build_examples(extractor, folder, k, max_ctx_words):
    examples = []
    for trs in sorted(folder.glob("*.trs")):
        turns = extractor.extract_conversations(trs)
        for i in range(1, len(turns)):
            prev, cur = turns[i - 1], turns[i]
            # same pairing rule as before: an adult replying to a child
            if cur["speaker_role"] != "Adult" or prev["speaker_role"] != "Child":
                continue
            if not extractor.is_valid_pair(prev["text"], cur["text"]):
                continue
            context = []
            for t in turns[max(0, i - k):i]:
                role = {"Child": "user", "Adult": "assistant"}.get(t["speaker_role"])
                if role is None:
                    continue
                context.append({"role": role, "content": clip(t["text"], max_ctx_words)})
            examples.append({
                "file": trs.name,
                "context": context,
                "reply": cur["text"],
                "last_child": prev["text"],
            })
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trs_folder")
    ap.add_argument("--parser-module", default="trs_extractor",
                    help="your parser's filename without .py")
    ap.add_argument("--context-turns", type=int, default=3,
                    help="how many previous turns to include")
    ap.add_argument("--max-ctx-words", type=int, default=60,
                    help="clip each context turn to its last N words")
    ap.add_argument("--echo-threshold", type=float, default=0.7)
    ap.add_argument("--rep-cap", type=int, default=20,
                    help="drop replies containing a 5-word phrase seen more than this many times")
    ap.add_argument("--out", default="context_pairs.json")
    args = ap.parse_args()

    module = importlib.import_module(args.parser_module)
    extractor = module.TRSExtractor()

    ex = build_examples(extractor, Path(args.trs_folder),
                        args.context_turns, args.max_ctx_words)
    print(f"Built {len(ex)} raw examples")

    ex = [e for e in ex if overlap(e["last_child"], e["reply"]) < args.echo_threshold]
    print(f"After near-echo filter: {len(ex)}")

    counts = Counter(g for e in ex for g in grams(e["reply"]))
    ex = [e for e in ex if all(counts[g] <= args.rep_cap for g in grams(e["reply"]))]
    print(f"After repetition cap: {len(ex)}")

    for e in ex:
        del e["last_child"]

    n_files = len({e["file"] for e in ex})
    avg_ctx = sum(len(e["context"]) for e in ex) / max(1, len(ex))
    print(f"Source files: {n_files}   avg context turns: {avg_ctx:.2f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(ex, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.out}")

    print("\nThree random examples:")
    import random
    random.seed(1)
    for e in random.sample(ex, min(3, len(ex))):
        for t in e["context"]:
            who = "ENFANT" if t["role"] == "user" else "ADULTE"
            print(f"  {who}: {t['content']}")
        print(f"  --> ADULTE: {e['reply']}\n")


if __name__ == "__main__":
    main()
