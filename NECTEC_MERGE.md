# NECTEC_MERGE.md — learned sign-segment merging on NECTEC ground truth

Status: **plan only, nothing implemented.** Decisions settled in the grilling
session of 2026-09-14 are marked ✅. Defaults I picked without asking are marked
🔶 and listed in §9. Veto or change any of them before implementation starts.

---

## 0. Goal and hypothesis

**Hypothesis:** stock `pose_to_segments` splits TSL signs into more pieces than
the real gloss boundaries have, so merging adjacent SIGN segments brings the
output closer to ground truth.

**Deliverable:** a merge step, learned from NECTEC's human gloss boundaries,
that turns `segments.json` into `segments_learned.json`. Output is still spans
only (CLAUDE.md §1).

Method: ✅ **(a)** threshold baseline first, then ✅ **(b)** pairwise classifier.
Keep (b) only if it beats (a) on held-out signers.

---

## 1. Scope decisions (✅ settled)

| # | Decision |
|---|---|
| Q1 | NECTEC is used for its **span boundaries only**. Gloss text is discarded at parse time and never written to disk. NECTEC takes the place of Phase 2's manual annotation for this experiment, which unblocks Phase 3 (metrics) and Phase 4 (adaptation). CLAUDE.md gets amended (§8). |
| Q2 | Crop the interpreter panel with a **static crop box stored in config**, pad to square, scale to 512×512, resample to 30 fps. Then **verify that the crop box contains the hands** (§3.3). |
| Q3 | Map the `Gloss Labeling` tier to `sign`. **Do not use** `Gloss` as a tier (its timing is exactly `CC_Aligned`'s, so it follows the speech, not the signing), nor `CC` / `CC_Aligned`. Exception: `Gloss` spans serve as a coverage mask, see Q6. |
| Q4 | Use **only the 4 videos that have both an mp4 and an `.eaf`.** Skip the orphan ค่าเฉลี่ย `.eaf` (its `14.mp4` is missing) and the 16 unannotated mp4s. |
| Q5 | (a) grid-search a merge threshold as the baseline, then (b) a pairwise merge classifier. Both are evaluated leave-one-signer-out. |
| Q6 | **Scored region = inside `Gloss` sentence spans only**, minus excluded spans. `Gloss` is used purely as a coverage mask ("the annotators looked here"). It never becomes a label or a segment, so Q3 still holds. |
| Q7 | If Gate 0 fails (§4), **stop and report**. (a) and (b) are not built. |
| Q8 | Selection: mean F1@0.50 over the 3 LOSO folds, **and** (b) must beat (a) on F1@0.50 in **at least 2 of 3 folds** to be kept. |
| Q9 | Pairs where either segment matches no GT sign are **excluded from training**. |
| Q10 | The learned merge is **not applied to ThaiSignVis** until Phase 2 ground truth exists. |
| Q11 | Crop edges that **cannot move** are listed as `fixed_edges` in config. Hand contact there is reported but not gated; the 1% rule applies only to movable edges, and the 60% hand-presence gate stays. Fixed for all 4: right (end of the video at x=1920) and bottom (subtitle bar at y=901). `moon` also has left fixed, because the teacher stands just past the panel edge. |

---

## 2. The data, as it actually is (verified 2026-09-14)

- **Raw videos are 1920×1080 at 60/1 fps CFR, `start_time=0`.** This is not
  the 512×512/30 fps that ThaiSignVis has. The existing `probe()` assert would
  reject them, correctly.
- Each frame is a full broadcast layout: slides on the left, the **interpreter
  in a grey panel on the right**, subtitles burned into a bottom bar. The
  science video has a **second person (the teacher) in the slide area**, and the
  math videos have a teacher picture-in-picture. Running pose on the full frame
  is unsafe.
- Each `.eaf` has 4 tiers. Only `Gloss Labeling` is used: 850–1,840 spans per
  video, median duration ≈ 400 ms, gaps between spans, no overlaps.
- The `._*.eaf` / `._*.pfsx` files are macOS AppleDouble junk. They are
  unparseable and must be skipped by the `._` prefix.
- The `.eaf` `MEDIA_URL` points at the original numbered files (`04.mp4`,
  `07.mp4`, …), not the local names. Pairing is by **same stem, one directory
  up** (`elan_final_ed/X.eaf` ↔ `X.mp4`). The ELAN spans fit inside each
  video's duration, which is consistent with that pairing.
- `scikit-learn==1.9.0` is **already in `requirements.lock`**, so no new
  dependency is needed.

| proposed slug | source (session folder / file stem) | raw frames @60 | min | GT signs | signer 🔶 |
|---|---|---|---|---|---|
| `nectec_compare` | คณิตศาสตร์ เล่ม 1บทที่ 1 / การเปรียบเทียบและเรียงลำดับ (11.07 นาที) | 39,988 | 11.1 | 852 | A |
| `nectec_muldiv` | คณิตศาสตร์ เล่ม 1บทที่ 4 / การคูณ หาร ไม่มีวงเล็บ (17.27 นาที) | 62,862 | 17.5 | 1,208 | B |
| `nectec_mixed` | คณิตศาสตร์ เล่ม 1บทที่ 4 / การบวก ลบ คูณ หารระคนแบบที่มีและไม่มีวงเล็บ (23.15 นาที) | 83,725 | 23.3 | 1,843 | B |
| `nectec_moon` | วิทยาศาสตร์ เล่ม 2 หน่วยที่ 5 บทที่ 1 / 18_การขึ้นและตกและรูปร่างของดวงจันทร์ (18.33 นาที) (1) | 66,722 | 18.5 | 972 | C |

Signer IDs come from eyeballing one frame per video: `muldiv` and `mixed` look
like the same man. Confirm this on more frames during §3.3, because the CV
folds depend on it. The CLAUDE.md "1 video = 1 signer" rule **does not hold
here**, so NECTEC needs an explicit `signer` field in config.

---

## 3. Stage N1 — preprocessing (crop, resample, ground truth)

### 3.1 Config

A new `config_nectec.yaml`, fed to the **unchanged** `src/pipeline.py` through
its existing `--config` flag:

```yaml
raw_root: sample/NECTEC                 # Thai paths appear only here
data_root: outputs/nectec_prep          # preprocessed 512x512@30 mp4s; pipeline.py reads these
output_root: outputs
crop: {x: 1310, y: 170, w: 610, h: 730} # 🔶 one box for all 4; per-video override allowed
run: [nectec_compare, nectec_muldiv, nectec_mixed, nectec_moon]
videos:
  nectec_compare:
    raw: "คณิตศาสตร์ เล่ม 1บทที่ 1/การเปรียบเทียบและเรียงลำดับ (11.07 นาที).mp4"
    eaf: "คณิตศาสตร์ เล่ม 1บทที่ 1/elan_final_ed/การเปรียบเทียบและเรียงลำดับ (11.07 นาที).eaf"
    path: nectec_compare.mp4            # relative to data_root, which is what pipeline.py expects
    signer: A
  # ...
```

The crop numbers above are **estimates from one downscaled frame per video**.
They get measured properly in §3.3 before anything is committed.

### 3.2 `src/nectec_prep.py`

For each video in `run`:

1. `ffprobe` the raw file and **assert 1920×1080, 60/1 fps, start_time 0**.
   Fail loudly on anything else.
2. Run ffmpeg once, cached (skip if the output exists, unless `--force`):
   `crop=w:h:x:y, pad=S:S:(S-w)/2:(S-h)/2:black, scale=512:512, fps=30`
   with `S = max(w,h)`. The output must pass the existing `probe()` assert.
   Frame count should be ≈ raw/2.
3. Parse `Gloss Labeling` with pympi into
   `outputs/{id}/ground_truth.json`, using the same schema as `segments.json`:
   `{video_id, fps: 30, source: "nectec:Gloss Labeling", segments: [{start_frame, end_frame, tier:"sign"}]}`.
   Round ms to frames with the §5.5 formula. The value is **dropped**. Assert no
   overlaps and no zero-length spans after rounding (if two spans collide at
   30 fps, log the count and fail).
   Also write `"scored_spans": [[start, end], ...]` from the `Gloss` tier
   (✅ Q6), rounded the same way, with values dropped. Assert that every
   `Gloss Labeling` span lies inside some scored span, and log any that don't.

Then run `python src/pipeline.py --config config_nectec.yaml`, which is stock
Phase 1: pose, landmark gate, excluded spans, `.eaf` and `segments.json`.
Estimated cost: ~70 min of video → ~126k frames → **~1h45m of pose extraction**,
plus re-encoding.

### 3.3 Crop verification (✅ required by Q2)

Run this before the full pose extraction, on a 2-minute clip from each video,
then again on the full `.pose` files:

1. **Contact sheet:** for each video, a grid of crops at the frames with the
   most extreme wrist positions (min/max x, min/max y of the pose wrists), plus
   10 evenly spaced frames. Save it to `reports/nectec_crop/{id}.jpg` for a
   human to look at.
2. **Edge-contact rate:** the fraction of frames where any hand landmark lies
   within 2% of a crop edge (excluding the padded bars). Fail if it is above
   1% on the **movable** edges of any video, and widen the box. Contact on
   `fixed_edges` is reported only (✅ Q11).

   Clip results (2026-09-14):
   - `compare` 0.5%, `muldiv` 0.0%: both pass.
   - `mixed` 4.7% at the bottom (subtitle bar).
   - `moon`: the default box gave 9.4%. Extending it to y=40 fixed the top.
     Widening the left edge to x=1160 put the teacher in the crop, so that was
     reverted. What's left is left 5.1% and right 2.8%, both fixed edges.
   - Hand presence during GT signs was ≥97% on every video.
   - Signers confirmed from the contact sheets: `muldiv` and `mixed` are the
     same man.
3. **Hand presence inside GT sign spans:** the left and right hand presence rate
   over frames covered by `ground_truth.json`. It must pass the existing 60%
   gate. 🔶 Also report it against ThaiSignVis's 98% as a sanity reference.
4. **Wrong-person check:** confirm that pose landmarks stay inside the panel on
   `nectec_moon`, the video with the teacher on screen. Because of the crop
   this should hold by construction; check it anyway.

If any check fails, widen the box, re-run the clip checks, and only then start
the full extraction.

---

## 4. Stage N2 — Gate 0: is the hypothesis even true?

`src/merge_learn.py gate` compares the stock `segments.json` (sign tier) with
`ground_truth.json` for each video, and writes the results to
`reports/nectec_merge.md`:

- predicted signs/min vs GT signs/min, and their ratio
- **over-segmentation:** the share of GT signs covered by ≥2 predicted segments
- **under-segmentation:** the share of predicted segments that cover ≥2 GT signs
- **oracle merge ceiling:** apply every merge the GT says is correct (§6.1
  labels) and score the result. This is the best any merge-only method can do.
- F1@{0.10, 0.25, 0.50} and frame accuracy for stock output (§5)

**Stop condition (✅ Q7):** if the predicted/GT ratio is ≤ 1.1 on at least 3 of
the 4 videos, **or** the oracle merge ceiling improves mean F1@0.50 by less
than 0.05 over stock, **stop and report**. Merging would be the wrong fix, and
(a)/(b) don't get built.

---

## 5. Evaluation protocol (shared by gate, (a) and (b))

- **Scored region (✅ Q6):** frames inside `scored_spans` (from the `Gloss`
  tier), minus `excluded_spans.json`. Predicted segments are clipped to this
  region before scoring, the same way `_subtract_spans` truncates them. GT sign
  spans outside it (there should be none, §3.2) are dropped and counted.
- **Metrics** (CLAUDE.md §5.7):
  - frame-level accuracy (sign vs not-sign)
  - segmental F1@{0.10, 0.25, 0.50}: one-to-one greedy matching by IoU
  - sign count ratio
- **Selection (✅ Q8):** mean **F1@0.50** across folds, and (b) is kept only if
  it also beats (a) on F1@0.50 in **≥ 2 of 3 folds**. Ties go to (a), since it
  is simpler.
- **Folds:** **leave-one-signer-out**, 3 folds: {A}, {B: muldiv+mixed}, {C}.
  Train on 2 signers, test on the third. Report every fold separately as well as
  the mean. With 3 signers, a mean on its own hides too much.
- No inter-annotator agreement exists for NECTEC, so there is no human ceiling
  to report. Say so in the report instead of inventing one.

---

## 6. Stage N3 — methods

### 6.1 Pair labels (shared)

Take every pair of adjacent predicted sign segments `(p_i, p_{i+1})`, in
start-frame order, inside the scored region.
Assign each predicted segment to the GT sign it overlaps most (or to nothing).

- **merge = 1:** both segments are assigned to the *same* GT sign.
- **merge = 0:** they are assigned to different GT signs.
- **unlabeled (✅ Q9):** either segment is assigned to no GT sign. Such pairs are left
  out of training. At inference they are still predicted, so the model must
  handle them.

### 6.2 (a) Threshold baseline

- Reuse `merge()` from `src/merge_segments.py` as is.
- Grid: 🔶 `max_gap ∈ 0..30` frames (0–1 s), plus optionally
  `min_dur ∈ {0, 3, 6, 9}` (a segment shorter than this always merges into its
  nearer neighbour). If the one-parameter version wins, drop `min_dur`.
- For each fold, pick the best parameters on the train signers by F1@0.50, then
  score the held-out signer.
- Also report the existing hand-picked `max_gap=5` as a reference row.

### 6.3 (b) Pairwise classifier

Per-pair features, all computed from `segments.json` and the `.pose` file
(no RGB):

| feature | source |
|---|---|
| gap length (frames) | segments |
| duration of `p_i`, `p_{i+1}`, and their ratio | segments |
| mean and max wrist speed inside the gap, normalized by shoulder width | pose `[0:33]` wrists 15/16, shoulders 11/12 |
| wrist speed at the last frames of `p_i` vs the first frames of `p_{i+1}` | pose |
| hand-presence fraction inside the gap (L, R) | confidence `[501:522]`, `[522:543]` |
| wrist displacement across the gap (start → end) | pose |

- Models: 🔶 `LogisticRegression` (standardized) and
  `HistGradientBoostingClassifier`, both from the already-installed sklearn.
  Keep whichever does better on the CV.
- Apply the merge greedily left to right: if `P(merge) > τ`, merge, then
  recompute features against the merged segment. τ is tuned on the train signers
  by F1@0.50, **not** fixed at 0.5.
- Class imbalance: `class_weight="balanced"` for LR; for HGB, tuning τ covers it.
- No hyperparameter search beyond τ and model choice: 3 signers is too few.
- Seeds: fix `random_state` for HGB. This is Phase 4, so seeds apply now
  (CLAUDE.md §4).

### 6.4 Output

- `outputs/{id}/segments_learned.json`: same schema, sign tier merged, sentence
  tier untouched, plus a `"merge": {"method": "a"|"b", "params": ...}` field.
- `models/merge_b.joblib` 🔶: the final model, refit on all 4 videos, and only
  if (b) wins.
- `reports/nectec_merge.md`: the gate table, then the per-fold table for stock /
  `max_gap=5` / (a) / (b) / oracle ceiling, then a verdict.

---

## 7. File plan (nothing written yet)

| file | new/changed | purpose |
|---|---|---|
| `config_nectec.yaml` | new | Thai paths, crop box, signer ids |
| `src/nectec_prep.py` | new | raw probe, ffmpeg crop/resample, EAF → `ground_truth.json`, crop verification |
| `src/merge_learn.py` | new | `gate` / `baseline` / `classifier` subcommands, metrics, LOSO CV |
| `src/merge_segments.py` | reused, unchanged | `merge()` |
| `src/pipeline.py` | **unchanged** | `run_video()` is called from `nectec_prep.py segment`. Not `main()`, which overwrites `reports/phase1_review.md` (human answers). |
| `.gitignore` | changed | ignores `outputs/nectec_prep/` and `outputs/nectec_clip/` (regenerable mp4s) |
| `CLAUDE.md` | changed | §8 amendments, §2 tree |

🔶 Metrics live inside `merge_learn.py` for now. Move them to their own module
when Phase 3 scores ThaiSignVis against Phase 2 annotations.

---

## 8. CLAUDE.md amendments (at implementation time)

- §1: add NECTEC as a second dataset with **gloss boundaries**. "Nothing names a
  sign" still holds for our outputs, because gloss text is dropped at parse
  time. The "zero gloss labels" claim applies to ThaiSignVis only.
- §1: NECTEC facts (1080p60, broadcast layout, a signer shared across videos,
  AppleDouble junk).
- §3: Phases 3–4 are running on NECTEC; Phase 2 is still open for ThaiSignVis.
- §5.1: "no crop" applies to ThaiSignVis only; NECTEC uses the static panel crop.
- §2: the new files.

---

## 9. Remaining defaults (🔶), to veto before implementation

Settled in round 2 and moved to §1: scored region (Q6), Gate 0 (Q7), selection
(Q8), unlabeled pairs (Q9), no ThaiSignVis transfer (Q10).

1. **Signer grouping** A / B(muldiv+mixed) / C. This is a fact to verify, not a
   choice: check it on more frames during §3.3. If `muldiv` and `mixed` turn
   out to be different people, use 4 folds.
2. **One crop box for all 4 videos**, with per-video override supported.
3. **Crop fail threshold:** hand edge-contact above 1%.
4. **Grid ranges:** `max_gap 0..30`, `min_dur {0,3,6,9}`.
5. **Models:** LR + HGB from sklearn; the final model is saved with joblib.
6. **Metrics** live inside `merge_learn.py` until Phase 3 needs them elsewhere.

## 10. Execution order

1. `python -m src.nectec_prep prep --clip` then `verify --clip`: crop
   verification on 2-min clips (§3.3). A human looks at
   `reports/nectec_crop/*_clip.jpg`.
2. `python -m src.nectec_prep prep`, then `segment` (~2 h CPU), then `verify`
   on the full videos.
3. `python -m src.merge_learn gate` (§4). **Stop here if it fails** and report.
   Run `python -m src.merge_learn selftest` first to check the metric code.
4. `merge_learn.py baseline` (a).
5. `merge_learn.py classifier` (b), then selection (Q8), then `reports/nectec_merge.md`.
6. Amend CLAUDE.md (§8).
