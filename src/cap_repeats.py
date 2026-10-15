"""
Drop pairs whose adult reply contains a 5-word phrase that appears more than
--cap times across the whole dataset. This removes sentences that one session
repeats over and over, which small models otherwise memorize.

Usage:
    python cap_repeats.py conversation_pairs_filtered.json --out conversation_pairs_final.json
"""

import argparse
import json
from collections import Counter


def grams(text, n=5):
    w = text.lower().split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_file")
    ap.add_argument("--cap", type=int, default=20,
                    help="drop replies containing a 5-word phrase seen more than this many times")
    ap.add_argument("--out", default="conversation_pairs_final.json")
    args = ap.parse_args()

    with open(args.pairs_file, encoding="utf-8") as f:
        pairs = json.load(f)

    counts = Counter(g for p in pairs for g in grams(p["adult"]))
    kept = [p for p in pairs if all(counts[g] <= args.cap for g in grams(p["adult"]))]
    print(f"{len(pairs)} -> {len(kept)} pairs")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
