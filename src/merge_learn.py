"""Metrics and Gate 0 for learned sign merging on NECTEC (NECTEC_MERGE.md §4-§5).

    python -m src.merge_learn gate       stock segments vs ground truth + oracle merge ceiling
    python -m src.merge_learn selftest   metric sanity checks on toy spans

Everything is in frames. Scoring happens only inside ground_truth.json's
`scored_spans` minus excluded_spans.json, and both predictions and GT are clipped
to that region before any metric.
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import json
from pathlib import Path

import numpy as np

from src.pipeline import REPO_ROOT, load_config

IOUS = (0.10, 0.25, 0.50)
GATE_RATIO = 1.1  # pred/GT sign count at or below this means "not over-segmented"
GATE_MIN_GAIN = 0.05  # oracle must add at least this much mean F1@0.50

Span = tuple[int, int]


def runs(mask: np.ndarray) -> list[Span]:
    """Contiguous True runs as [start, end) spans."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return [(int(s), int(e)) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def to_mask(spans, n: int) -> np.ndarray:
    mask = np.zeros(n, bool)
    for s, e in spans:
        mask[s:e] = True
    return mask


def clip(spans: list[Span], region: np.ndarray) -> list[Span]:
    """Intersect each span with the region; a span crossing a hole splits in two."""
    return [(s + a, s + b) for s, e in spans for a, b in runs(region[s:e])]


def overlap_matrix(pred: list[Span], gt: list[Span]) -> np.ndarray:
    """[P, G] overlap in frames."""
    if not pred or not gt:
        return np.zeros((len(pred), len(gt)), int)
    p, g = np.array(pred), np.array(gt)
    return np.clip(np.minimum(p[:, None, 1], g[None, :, 1]) - np.maximum(p[:, None, 0], g[None, :, 0]), 0, None)


def f1_at(pred: list[Span], gt: list[Span], threshold: float) -> float:
    """Segmental F1 as in MS-TCN: each prediction takes its best-IoU GT; TP if IoU >= t and that GT is unclaimed."""
    if not pred and not gt:
        return 1.0
    inter = overlap_matrix(pred, gt).astype(float)
    tp = 0
    if pred and gt:
        p, g = np.array(pred), np.array(gt)
        union = np.maximum(p[:, None, 1], g[None, :, 1]) - np.minimum(p[:, None, 0], g[None, :, 0])
        iou = inter / union
        hit = np.zeros(len(gt), bool)
        for row in iou:
            j = int(row.argmax())
            if row[j] >= threshold and not hit[j]:
                hit[j] = True
                tp += 1
    return 2 * tp / (len(pred) + len(gt))


def frame_accuracy(pred: list[Span], gt: list[Span], region: np.ndarray) -> float:
    n = len(region)
    return float(((to_mask(pred, n) == to_mask(gt, n)) & region).sum()) / max(int(region.sum()), 1)


def assign(pred: list[Span], gt: list[Span]) -> list[int | None]:
    """Index of the GT sign each prediction overlaps most, or None."""
    m = overlap_matrix(pred, gt)
    return [int(row.argmax()) if row.size and row.max() > 0 else None for row in m]


def oracle_merge(pred: list[Span], gt: list[Span]) -> list[Span]:
    """Merge every adjacent pair assigned to the same GT sign: the best any merge-only method can do."""
    out: list[Span] = []
    prev = None
    for (s, e), a in zip(pred, assign(pred, gt)):
        if out and a is not None and a == prev:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
        prev = a
    return out


def score(pred: list[Span], gt: list[Span], region: np.ndarray) -> dict:
    return {**{f"f1@{t:.2f}": f1_at(pred, gt, t) for t in IOUS}, "frame_acc": frame_accuracy(pred, gt, region)}


def load_video(video_id: str, config: dict) -> dict:
    """Stock sign predictions and GT, both clipped to the scored region."""
    out_dir = REPO_ROOT / config["output_root"] / video_id
    read = lambda name: json.loads((out_dir / name).read_text(encoding="utf-8"))
    gt_doc, seg_doc, excl_doc = read("ground_truth.json"), read("segments.json"), read("excluded_spans.json")
    if not (gt_doc["fps"] == seg_doc["fps"] == excl_doc["fps"]):
        raise AssertionError(f"{video_id}: fps mismatch gt={gt_doc['fps']} segments={seg_doc['fps']} excluded={excl_doc['fps']}")

    gt_raw = [(s["start_frame"], s["end_frame"]) for s in gt_doc["segments"]]
    pred_raw = sorted((s["start_frame"], s["end_frame"]) for s in seg_doc["segments"] if s["tier"] == "sign")
    n = max(e for _, e in gt_raw + pred_raw + [tuple(x) for x in gt_doc["scored_spans"]] + [tuple(x) for x in excl_doc["spans"]])
    region = to_mask(gt_doc["scored_spans"], n) & ~to_mask(excl_doc["spans"], n)
    gt = clip(gt_raw, region)
    if len(gt) != len(gt_raw):
        print(f"  [{video_id}] GT signs {len(gt_raw)} -> {len(gt)} after clipping to scored region")
    return {
        "video_id": video_id,
        "signer": config["videos"][video_id]["signer"],
        "fps": gt_doc["fps"],
        "region": region,
        "gt": gt,
        "pred": clip(pred_raw, region),
    }


def gate(config: dict, video_ids: list[str]) -> None:
    rows = []
    for video_id in video_ids:
        v = load_video(video_id, config)
        pred, gt, region = v["pred"], v["gt"], v["region"]
        m = overlap_matrix(pred, gt) > 0
        oracle = oracle_merge(pred, gt)
        rows.append({
            "video_id": video_id,
            "signer": v["signer"],
            "scored_min": region.sum() / v["fps"] / 60,
            "gt": len(gt),
            "pred": len(pred),
            "ratio": len(pred) / len(gt),
            "over": float((m.sum(axis=0) >= 2).mean()),
            "under": float((m.sum(axis=1) >= 2).mean()) if pred else 0.0,
            "oracle_n": len(oracle),
            "stock": score(pred, gt, region),
            "oracle": score(oracle, gt, region),
        })

    mean = lambda f: float(np.mean([f(r) for r in rows]))
    not_over = sum(r["ratio"] <= GATE_RATIO for r in rows)
    gain = mean(lambda r: r["oracle"]["f1@0.50"]) - mean(lambda r: r["stock"]["f1@0.50"])
    passed = not_over < 3 and gain >= GATE_MIN_GAIN

    lines = [
        "# NECTEC learned merge\n",
        "## Gate 0: is stock pose_to_segments over-segmenting?\n",
        "Scored region = `Gloss` spans minus excluded spans. over-seg = share of GT signs covered by >=2 predictions; "
        "under-seg = share of predictions covering >=2 GT signs. Oracle = merge every adjacent pair the GT says belongs "
        "to one sign (ceiling for any merge-only method).\n",
        "| video | signer | scored min | GT signs | pred signs | pred/GT | over-seg | under-seg | "
        "stock F1@.10 | F1@.25 | F1@.50 | frame acc | oracle signs | oracle F1@.10 | F1@.25 | F1@.50 | frame acc |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    fmt = lambda s: " | ".join(f"{s[k]:.3f}" for k in ("f1@0.10", "f1@0.25", "f1@0.50", "frame_acc"))
    for r in rows:
        lines.append(
            f"| {r['video_id']} | {r['signer']} | {r['scored_min']:.1f} | {r['gt']} | {r['pred']} | {r['ratio']:.2f} | "
            f"{r['over']:.1%} | {r['under']:.1%} | {fmt(r['stock'])} | {r['oracle_n']} | {fmt(r['oracle'])} |"
        )
    mean_score = lambda key: {k: mean(lambda r: r[key][k]) for k in rows[0][key]}
    lines.append(
        f"| **mean** | | | | | {mean(lambda r: r['ratio']):.2f} | {mean(lambda r: r['over']):.1%} | "
        f"{mean(lambda r: r['under']):.1%} | {fmt(mean_score('stock'))} | | {fmt(mean_score('oracle'))} |"
    )
    lines += [
        "",
        f"- Videos with pred/GT <= {GATE_RATIO}: **{not_over}/{len(rows)}** (stop if >= 3)",
        f"- Oracle gain in mean F1@0.50: **{gain:+.3f}** (stop if < {GATE_MIN_GAIN})",
        f"- **Gate 0: {'PASS, build (a) and (b)' if passed else 'FAIL, stop: merging is not the fix'}**",
        "",
    ]
    out = REPO_ROOT / "reports" / "nectec_merge.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[3:]))
    print(f"-> {out}")


def selftest() -> None:
    gt = [(0, 10), (20, 30)]
    pred = [(0, 4), (5, 10), (20, 30)]
    assert runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    assert clip([(0, 10)], to_mask([(0, 3), (6, 20)], 20)) == [(0, 3), (6, 10)]
    assert f1_at(pred, gt, 0.5) == 0.8  # (0,4) IoU .4 FP, (5,10) IoU .5 TP, (20,30) TP
    assert f1_at(pred, gt, 0.1) == 0.8  # (0,4) claims GT 0 first, (5,10) is then a duplicate -> FP
    assert f1_at([], gt, 0.5) == 0.0
    assert assign(pred + [(12, 15)], gt) == [0, 0, 1, None]
    assert oracle_merge(pred, gt) == [(0, 10), (20, 30)]
    assert oracle_merge([(0, 4), (12, 15), (5, 10)], gt) == [(0, 4), (12, 15), (5, 10)]  # None breaks the chain
    assert f1_at(oracle_merge(pred, gt), gt, 0.5) == 1.0
    region = to_mask([(0, 30)], 30)
    assert frame_accuracy([(0, 10)], gt, region) == 20 / 30
    print("selftest ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["gate", "selftest"])
    parser.add_argument("--config", default=str(REPO_ROOT / "config_nectec.yaml"))
    args = parser.parse_args()
    if args.stage == "selftest":
        selftest()
        return
    config = load_config(Path(args.config))
    gate(config, config["run"])


if __name__ == "__main__":
    main()
