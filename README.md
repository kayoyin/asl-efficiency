# American Sign Language Handshapes Reflect Pressures for Communicative Efficiency

Code and data supporting the paper [American Sign Language Handshapes Reflect Pressures for Communicative Efficiency](https://arxiv.org/abs/2406.04024) (ACL 2024).

The paper measures **articulatory effort** (finger independence) and **perceptual effort** (handshape distance) for ASL fingerspelling handshapes, and correlates them against ASL and English usage statistics. We find that handshapes frequent in *native* ASL signs are easier to produce, while no such pressure is observed for handshapes derived from English (initialized + loan signs).

## Repository layout

```
src/
  rotation.py              MediaPipe landmarks -> joint-angle representation
  effort.py                Articulatory & perceptual effort metrics
  utils.py                 Misc. helpers (IQR outlier removal)
  extract_landmarks.py     Segment fingerspelled letters from the Google corpus
  compute_scores.py        End-to-end: landmarks -> angles -> effort scores
  frequencies.py           ASL-LEX & Wikipedia usage statistics
notebooks/
  analysis.ipynb           Reproduces correlations and Figure 5
data/
  angles/                  Joint-angle JSONs for each Google corpus split
  freq/                    Pre-computed ASL & English usage statistics
plots/                     Output figures
```

## Quick start

```bash
pip install -r requirements.txt
```

To regenerate effort scores from the joint-angle JSONs and produce the
analysis:

```bash
# 1. Recompute scores for every split.
for split in supp0_5 supp6_118 supp119_614 train0_22; do
  python -m src.compute_scores \
    --in_path data/angles/${split}.json \
    --split_name ${split} \
    --angles_dir data/angles \
    --scores_dir data/scores/${split}
done

# 2. Open the analysis notebook to produce correlations + figures.
jupyter notebook notebooks/analysis.ipynb
```

## Reproducing the full pipeline

The committed `data/angles/*.json` files are the joint-angle encodings of
the dominant-hand landmarks for ~1,000 isolated fingerspelled letters
extracted from the Google ASL Fingerspelling Recognition Corpus
(Chow et al., 2023). To regenerate them from scratch:

1. **Download the corpus** from
   <https://www.kaggle.com/competitions/asl-fingerspelling>. The download
   contains MediaPipe landmark sequences keyed by `sequence_id` plus a CSV
   of phrase labels.
2. **Segment isolated letters** with the heuristic algorithm described in
   §3.3 of the paper (sharpest local minima in hand-velocity), then
   manually correct the alignment:
   ```bash
   python -m src.extract_landmarks \
     --landmarks_dir <corpus>/train_landmarks \
     --label_csv     <corpus>/train.csv \
     --split_name    train0_22 \
     --out_path      data/google/train0_22.json
   ```
   Repeat for each split (`supp0_5`, `supp6_118`, `supp119_614`).
3. **Encode joint angles + compute effort scores** with
   `src/compute_scores.py` (see the snippet above).

To regenerate the usage statistics:

```bash
# Requires asl-lex.csv from https://asl-lex.org/ (request access).
python -m src.frequencies --asl_lex data/raw/asl-lex.csv --out_dir data/freq
```

`src/frequencies.py` downloads a 10,000-article sample of English Wikipedia
(via the `datasets` library, seed=42) and writes per-letter frequencies as
well as pairwise contextual confusability `H({x1,x2}|C)`.

## What the metrics mean

* **Finger independence** (`src.effort.finger_independence`) sums the
  pairwise angular distances between corresponding joints of the four
  non-thumb fingers (MCP / PIP / DIP). Thumb effort -- the angular
  distance between this hand's thumb configuration and the closest
  resting-hand thumb -- is added with weight 2.
* **Handshape distance** (`src.effort.handshape_distance`) is the mean
  per-joint angular distance between two handshapes. It approximates
  perceptual disambiguability between two letters.
* **Conditional entropy** `H({x1,x2}|C)` over English contexts
  (`src.frequencies.english_letter_confusability`) measures how
  distinguishable two letters are from the n-gram (n<=4) preceding them.

## Citation

```bibtex
@inproceedings{yin24acl,
    title     = {American Sign Language Handshapes Reflect Pressures for Communicative Efficiency},
    author    = {Yin, Kayo and Regier, Terry and Klein, Dan},
    booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
    year      = {2024}
}
```

## License

Data and code in this repository are released for research use. ASL-LEX,
the Google ASL Fingerspelling Recognition Corpus, and the Wikipedia
content used to generate the statistics here are governed by their
respective licenses; please consult those sources directly.
