"""Compute usage statistics that the analysis correlates against effort.

Two sources:

    * **ASL-LEX 2.0** (Sehyr et al., 2021): per-handshape sign-frequency
      sums over the native, foreign (initialized + loan), and combined
      lexicons.
    * **Wikipedia** (10,000 random English articles, low-frequency words
      only): per-letter token/type frequencies and pairwise contextual
      confusability ``H(X | C)``.

This module provides both library functions and a CLI. To run end-to-end::

    python -m src.frequencies --asl_lex data/raw/asl-lex.csv \\
                              --out_dir data/freq

ASL-LEX is not redistributed here -- request it from
https://asl-lex.org/. The 10k-article Wikipedia sample is built on the fly
by :func:`build_wikipedia_word_freq` (requires the ``datasets`` library).
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# ASL-LEX uses a richer handshape inventory than the 22 fingerspelling
# handshapes; we collapse to the FS-handshape labels by stripping prefixes
# (e.g. ``closed_b -> b``, ``flat_o -> o``) and keeping only those whose
# stripped name is one of these single-letter handshapes.
ALPHABET_SHAPES = list("abcdefghiklmnoprstuvwxy")  # 22 letters of ASL FS
# 'closed_b' and 'closed_e' are interchangeable with 'b' and 'e' in ASL-LEX.
HANDSHAPE_ALIASES = {"closed_b": "b", "closed_e": "e", "bent_1": "x"}


def _normalize_handshape(label):
    if not isinstance(label, str):
        return None
    label = HANDSHAPE_ALIASES.get(label, label)
    stripped = label[label.rfind("_") + 1 :]  # 'flat_o' -> 'o'
    return stripped if stripped in ALPHABET_SHAPES else None


def asl_handshape_frequencies(asl_lex_csv, out_dir):
    """Dump per-handshape sign frequencies, split by native vs. foreign.

    Files written:
        ``asl_freq.json``, ``asl_freq_type.json`` -- all signs.
        ``nat_asl_freq.json``, ``nat_asl_freq_type.json`` -- native only.
        ``for_asl_freq.json``, ``for_asl_freq_type.json`` -- initialized + loan.
    """
    df = pd.read_csv(asl_lex_csv, encoding_errors="replace")

    foreign_cols = ["Initialized.2.0", "FingerspelledLoanSign.2.0"]
    is_foreign = df[foreign_cols].fillna(0).sum(axis=1) > 0

    sums = {"all": defaultdict(float), "native": defaultdict(float), "foreign": defaultdict(float)}
    types = {"all": Counter(), "native": Counter(), "foreign": Counter()}

    for i, row in df.iterrows():
        letter = _normalize_handshape(row["Handshape.2.0"])
        if letter is None:
            continue
        freq = row.get("SignFrequency(M)", 0) or 0
        scope = "foreign" if is_foreign.iloc[i] else "native"
        sums["all"][letter] += freq
        sums[scope][letter] += freq
        types["all"][letter] += 1
        types[scope][letter] += 1

    os.makedirs(out_dir, exist_ok=True)
    for prefix, scope in [("asl", "all"), ("nat_asl", "native"), ("for_asl", "foreign")]:
        with open(os.path.join(out_dir, f"{prefix}_freq.json"), "w") as f:
            json.dump(dict(sums[scope]), f)
        with open(os.path.join(out_dir, f"{prefix}_freq_type.json"), "w") as f:
            json.dump(dict(types[scope]), f)


def build_wikipedia_word_freq(num_articles=10000, seed=42):
    """Sample ``num_articles`` random English Wikipedia articles and return a
    Counter of word frequencies, dropping hapax legomena."""
    from datasets import load_dataset  # local import; heavy dep

    dataset = load_dataset("wikipedia", "20220301.en")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset["train"]), size=num_articles, replace=False)
    articles = dataset["train"].select(indices.tolist())
    words = []
    for article in articles:
        for w in article["text"].split():
            w = re.sub(r"[^a-zA-Z]", "", w).lower()
            if w:
                words.append(w)
    counts = Counter(words)
    return Counter({w: c for w, c in counts.items() if c > 1})


def english_letter_frequencies(words_freq, drop_top_k=20000):
    """Per-letter frequency over rare words, both token-weighted and type-only.

    Following the paper, we drop the ``drop_top_k`` most common types and
    treat the remainder as a proxy for words likely to be fingerspelled.
    """
    rare = words_freq.most_common()[drop_top_k:]
    rare_types = [w for w, _ in rare]
    rare_tokens = "".join(w for w, c in rare for _ in range(c))
    rare_types_text = "".join(rare_types)

    by_token = Counter(rare_tokens)
    by_type = Counter(rare_types_text)
    return dict(by_token), dict(by_type)


def english_letter_confusability(words_freq, drop_top_k=20000, max_context=4):
    """Pairwise contextual confusability ``H({x1,x2} | C)``.

    The context of a letter at position ``i`` of ``w`` is
    ``w[max(0, i - max_context):i]``. Returned dict keys are sorted bigrams
    (``"ae"`` not ``"ea"``).
    """
    rare = words_freq.most_common()[drop_top_k:]
    rare_types = [w for w, _ in rare]

    # char_context_model[char][context] = count
    char_context = defaultdict(Counter)
    # word_context_model[context][next_char] = count
    word_context = defaultdict(Counter)

    for word in rare_types:
        word = re.sub(r"[^a-zA-Z]", "", word).lower()
        for i, ch in enumerate(word):
            ctx = word[max(0, i - max_context) : i]
            if len(ctx) < 2:
                continue
            word_context[ctx][ch] += 1
            char_context[ch][ctx] += 1

    return _conditional_entropy(char_context)


def _conditional_entropy(char_context):
    """Compute H({x1,x2}|C) for every letter pair from a char->context Counter."""
    entropy = {}
    chars = sorted(char_context.keys())
    for i, c1 in enumerate(chars):
        ctx1 = char_context[c1]
        for c2 in chars[i + 1 :]:
            ctx2 = char_context[c2]
            joint = defaultdict(float)
            ctx_marginal = defaultdict(float)
            for c in ctx1.keys() & ctx2.keys():
                joint[(c, c1)] += ctx1[c]
                joint[(c, c2)] += ctx2[c]
                ctx_marginal[c] += ctx1[c] + ctx2[c]
            joint_total = sum(joint.values())
            ctx_total = sum(ctx_marginal.values())
            if joint_total == 0:
                entropy[c1 + c2] = 0.0
                continue
            joint = {k: v / joint_total for k, v in joint.items()}
            ctx_marginal = {k: v / ctx_total for k, v in ctx_marginal.items()}
            ent = 0.0
            for c in ctx1.keys() & ctx2.keys():
                for ch in (c1, c2):
                    p = joint[(c, ch)]
                    if p > 0:
                        ent += p * np.log2(p / ctx_marginal[c])
            entropy[c1 + c2] = -ent
    return entropy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asl_lex", required=True, help="Path to asl-lex.csv")
    parser.add_argument("--out_dir", default="data/freq")
    parser.add_argument(
        "--skip_wikipedia",
        action="store_true",
        help="Only recompute the ASL-LEX-derived stats.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    asl_handshape_frequencies(args.asl_lex, args.out_dir)

    if not args.skip_wikipedia:
        words_freq = build_wikipedia_word_freq()
        with open(os.path.join(args.out_dir, "wikipedia_words.json"), "w") as f:
            json.dump(words_freq, f)
        token_freq, type_freq = english_letter_frequencies(words_freq)
        with open(os.path.join(args.out_dir, "wiki_char_word.json"), "w") as f:
            json.dump(token_freq, f)
        with open(os.path.join(args.out_dir, "wiki_char_type.json"), "w") as f:
            json.dump(type_freq, f)
        confusability = english_letter_confusability(words_freq)
        with open(os.path.join(args.out_dir, "wiki_ent_type.json"), "w") as f:
            json.dump(confusability, f)


if __name__ == "__main__":
    main()
