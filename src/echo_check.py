"""
Measure how often the adult reply just repeats the child's words ("echo"),
and write a filtered pairs file with near-echoes removed.

Usage:
    python echo_check.py conversation_pairs.json --threshold 0.7
In Colab:
    !python echo_check.py conversation_pairs.json --threshold 0.7
"""

import argparse
import json
import random
import re


def words(text):
    return re.findall(r"\w+", text.lower())


def overlap(child, adult):
    """Fraction of the adult's words that also appear in the child's line."""
    c, a = set(words(child)), words(adult)
    if not a:
        return 0.0
    return sum(1 for w in a if w in c) / len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_file")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="drop pairs with overlap >= this value")
    ap.add_argument("--out", default="conversation_pairs_filtered.json")
    args = ap.parse_args()

    with open(args.pairs_file, encoding="utf-8") as f:
        pairs = json.load(f)

    scores = [overlap(p["child"], p["adult"]) for p in pairs]
    n = len(pairs)

    print(f"Total pairs: {n}\n")
    print("Echo rate (share of pairs with overlap >= threshold):")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        k = sum(1 for s in scores if s >= t)
        print(f"  >= {t:.1f}: {k:5d}  ({k / n:.1%})")

    # print examples for easy access
    random.seed(0)
    near = [(p, s) for p, s in zip(pairs, scores) if args.threshold <= s]
    print(f"\nSample of pairs that would be REMOVED (overlap >= {args.threshold}):")
    for p, s in random.sample(near, min(8, len(near))):
        print(f"  [{s:.2f}] child: {p['child']}")
        print(f"         adult: {p['adult']}\n")

    kept = [p for p, s in zip(pairs, scores) if s < args.threshold]
    print(f"Kept {len(kept)} of {n} pairs ({len(kept) / n:.1%}).")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
    print(f"Saved filtered pairs to {args.out}")


if __name__ == "__main__":
    main()
