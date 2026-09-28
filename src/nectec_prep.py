"""NECTEC preprocessing, crop verification and stock segmentation (NECTEC_MERGE.md §3).

    python -m src.nectec_prep prep    [--clip] [--force]   raw mp4 -> 512x512@30 crop; .eaf -> ground_truth.json
    python -m src.nectec_prep verify  [--clip]             hands-inside-crop checks + contact sheets
    python -m src.nectec_prep segment [--force]            pipeline.run_video() on the cropped mp4s, then overlay
    python -m src.nectec_prep overlay                      add GT_SIGN / GT_SENTENCE tiers to the .eaf

--clip works on a 2-minute excerpt per video, so the crop can be checked before
the ~2 h full pose extraction.
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pympi
from pose_format import Pose

from src.pipeline import (
    LEFT_HAND_LANDMARKS,
    REPO_ROOT,
    RIGHT_HAND_LANDMARKS,
    SUBPROCESS_ENV,
    extract_pose,
    load_config,
    probe,
    run_video,
)

CLIP_START_S = 120
CLIP_LEN_S = 120
OUT_SIZE = 512
HANDS = slice(501, 543)
EDGE_BAND = 0.02 * OUT_SIZE  # px from the crop edge that counts as "touching"
EDGE_FAIL = 0.01  # fail if more than this share of hand frames touch an edge
HAND_GATE = 0.60  # same threshold as pipeline.landmark_report


def crop_box(video_id: str, config: dict) -> dict:
    return {**config["crop"], **config["videos"][video_id].get("crop", {})}


def media_paths(video_id: str, config: dict, clip: bool) -> tuple[Path, Path]:
    """(512x512 mp4, .pose) for the full video or its clip."""
    if clip:
        clip_dir = REPO_ROOT / config["output_root"] / "nectec_clip"
        return clip_dir / f"{video_id}.mp4", clip_dir / f"{video_id}.pose"
    video = REPO_ROOT / config["data_root"] / config["videos"][video_id]["path"]
    return video, REPO_ROOT / config["output_root"] / video_id / f"{video_id}.pose"


def probe_raw(video_path: Path) -> int:
    """Assert the NECTEC broadcast format (1920x1080, 60/1 fps, start 0); return frame count."""
    if not video_path.exists():
        raise FileNotFoundError(f"raw video not found: {video_path}")
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,start_time",
        "-of", "json", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=SUBPROCESS_ENV, check=True)
    s = json.loads(result.stdout)["streams"][0]
    got = (int(s["width"]), int(s["height"]), s["r_frame_rate"], float(s["start_time"]))
    if got != (1920, 1080, "60/1", 0.0):
        raise AssertionError(f"{video_path}: expected (1920, 1080, '60/1', 0.0), got {got}")
    return int(s["nb_frames"])


def transcode(raw: Path, out: Path, pose: Path, box: dict, clip: bool, force: bool) -> None:
    """Static crop -> pad to square -> 512x512 -> 30 fps. Written via a .part file so a killed run never caches.

    Re-encoding deletes `pose`: a .pose extracted from the previous crop would otherwise be served from cache.
    """
    if out.exists() and not force:
        print(f"  [crop] cached: {out}")
        return
    if pose.exists():
        pose.unlink()
        print(f"  [crop] re-encoding, dropped stale pose: {pose}")
    out.parent.mkdir(parents=True, exist_ok=True)
    side = max(box["w"], box["h"])
    vf = (
        f"crop={box['w']}:{box['h']}:{box['x']}:{box['y']},"
        f"pad={side}:{side}:(ow-iw)/2:(oh-ih)/2:black,scale={OUT_SIZE}:{OUT_SIZE},fps=30"
    )
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if clip:
        cmd += ["-ss", str(CLIP_START_S), "-t", str(CLIP_LEN_S)]
    part = out.with_suffix(".part.mp4")
    cmd += ["-i", str(raw), "-vf", vf, "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", str(part)]
    subprocess.run(cmd, check=True, env=SUBPROCESS_ENV)
    part.replace(out)


def ms_spans(eaf: pympi.Elan.Eaf, tier: str, fps: float) -> list[tuple[int, int]]:
    """Tier -> sorted frame spans. The annotation text is dropped here and never leaves this function."""
    return sorted((round(s * fps / 1000), round(e * fps / 1000)) for s, e, *_ in eaf.get_annotation_data_for_tier(tier))


def write_ground_truth(video_id: str, config: dict, fps: float) -> None:
    eaf_path = REPO_ROOT / config["raw_root"] / config["videos"][video_id]["eaf"]
    if not eaf_path.exists():
        raise FileNotFoundError(f"eaf not found: {eaf_path}")
    eaf = pympi.Elan.Eaf(str(eaf_path))
    signs = ms_spans(eaf, "Gloss Labeling", fps)
    scored = ms_spans(eaf, "Gloss", fps)  # coverage mask only (NECTEC_MERGE.md Q6), never a label

    for name, spans in (("Gloss Labeling", signs), ("Gloss", scored)):
        zero = [sp for sp in spans if sp[1] <= sp[0]]
        overlap = [(a, b) for a, b in zip(spans, spans[1:]) if b[0] < a[1]]
        if zero or overlap:
            raise ValueError(f"{video_id} {name}: {len(zero)} zero-length, {len(overlap)} overlapping after rounding to {fps} fps")

    outside = [sp for sp in signs if not any(cs <= sp[0] and sp[1] <= ce for cs, ce in scored)]
    if outside:
        print(f"  [gt] {len(outside)} sign span(s) outside every Gloss span: {outside}")

    out = REPO_ROOT / config["output_root"] / video_id / "ground_truth.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "video_id": video_id,
        "fps": fps,
        "source": "nectec:Gloss Labeling",
        "segments": [{"start_frame": s, "end_frame": e, "tier": "sign"} for s, e in signs],
        "scored_spans": scored,
    }, indent=2), encoding="utf-8")
    print(f"  [gt] {len(signs)} signs, {len(scored)} scored spans -> {out}")


def prep(video_id: str, config: dict, clip: bool, force: bool) -> None:
    raw = REPO_ROOT / config["raw_root"] / config["videos"][video_id]["raw"]
    raw_frames = probe_raw(raw)
    video, pose = media_paths(video_id, config, clip)
    transcode(raw, video, pose, crop_box(video_id, config), clip, force)
    fps, _, _, n_frames = probe(video)  # asserts 512x512 @ 30
    expected = CLIP_LEN_S * fps if clip else raw_frames / 2
    if abs(n_frames - expected) > 2:
        raise AssertionError(f"{video}: {n_frames} frames, expected ~{expected:.0f}")
    print(f"  [crop] {n_frames} frames @ {fps} fps -> {video}")
    write_ground_truth(video_id, config, fps)


def verify(video_id: str, config: dict, clip: bool) -> dict:
    """Are the hands inside the crop? Edge contact, hand presence during GT signs, nose inside box, contact sheet."""
    video, pose_path = media_paths(video_id, config, clip)
    extract_pose(video, pose_path, force=False)
    pose = Pose.read(pose_path.read_bytes())
    fps = pose.body.fps
    xy = np.asarray(pose.body.data.data)[:, 0, :, :2]
    conf = pose.body.confidence[:, 0]
    n = len(conf)

    # Crop content area in 512x512 output pixels (the rest is black padding).
    box = crop_box(video_id, config)
    scale = OUT_SIZE / max(box["w"], box["h"])
    x0 = (max(box["w"], box["h"]) - box["w"]) / 2 * scale
    y0 = (max(box["w"], box["h"]) - box["h"]) / 2 * scale
    x1, y1 = x0 + box["w"] * scale, y0 + box["h"] * scale

    hx, hy, hm = xy[:, HANDS, 0], xy[:, HANDS, 1], conf[:, HANDS] > 0
    hand_frames = max(int(hm.any(axis=1).sum()), 1)
    edges = {
        "left": hm & (hx < x0 + EDGE_BAND),
        "right": hm & (hx > x1 - EDGE_BAND),
        "top": hm & (hy < y0 + EDGE_BAND),
        "bottom": hm & (hy > y1 - EDGE_BAND),
    }
    edge_rates = {k: float(v.any(axis=1).sum()) / hand_frames for k, v in edges.items()}
    # Only edges the crop could still widen are gated; fixed ones are reported (NECTEC_MERGE.md Q11).
    fixed = set(config["videos"][video_id].get("fixed_edges", config["fixed_edges"]))
    if fixed - edges.keys():
        raise ValueError(f"{video_id}: unknown fixed_edges {fixed - edges.keys()}")
    gated = [v.any(axis=1) for k, v in edges.items() if k not in fixed]
    edge_rate = float(np.logical_or.reduce(gated).sum()) / hand_frames if gated else 0.0

    gt = json.loads((REPO_ROOT / config["output_root"] / video_id / "ground_truth.json").read_text(encoding="utf-8"))
    offset = round(CLIP_START_S * fps) if clip else 0
    in_sign = np.zeros(n, bool)
    for seg in gt["segments"]:
        in_sign[max(seg["start_frame"] - offset, 0):max(min(seg["end_frame"] - offset, n), 0)] = True
    if not in_sign.any():
        raise RuntimeError(f"{video_id}: no GT sign frames inside the verified range")
    left = float((conf[in_sign][:, LEFT_HAND_LANDMARKS].sum(axis=1) > 0).mean())
    right = float((conf[in_sign][:, RIGHT_HAND_LANDMARKS].sum(axis=1) > 0).mean())

    nose_seen = conf[:, 0] > 0
    nose_out = nose_seen & ((xy[:, 0, 0] < x0) | (xy[:, 0, 0] > x1) | (xy[:, 0, 1] < y0) | (xy[:, 0, 1] > y1))
    nose_out_rate = float(nose_out.sum()) / max(int(nose_seen.sum()), 1)

    failures = []
    if edge_rate > EDGE_FAIL:
        failures.append(f"hand contact on movable edges {edge_rate:.1%} > {EDGE_FAIL:.0%}")
    if left < HAND_GATE or right < HAND_GATE:
        failures.append(f"hand presence during GT signs L={left:.1%} R={right:.1%} < {HAND_GATE:.0%}")
    if nose_out_rate > 0:
        failures.append(f"nose outside crop on {nose_out_rate:.1%} of frames")

    sheet = contact_sheet(video, video_id, hx, hy, hm, (x0, y0, x1, y1), clip)
    result = {
        "video_id": video_id, "clip": clip, "frames": n,
        "edge_contact_rate": edge_rate, "edge_contact_by_side": edge_rates, "fixed_edges": sorted(fixed),
        "left_hand_in_gt_signs": left, "right_hand_in_gt_signs": right,
        "nose_outside_rate": nose_out_rate, "contact_sheet": str(sheet.relative_to(REPO_ROOT)),
        "failures": failures,
    }
    sides = " ".join(f"{k}{'*' if k in fixed else ''}={v:.1%}" for k, v in edge_rates.items())
    print(f"  [verify] movable-edge={edge_rate:.1%} ({sides}; *fixed) L={left:.1%} R={right:.1%} nose_out={nose_out_rate:.1%} "
          f"-> {'PASS' if not failures else 'FAIL: ' + '; '.join(failures)}")
    return result


def contact_sheet(video: Path, video_id: str, hx, hy, hm, box, clip: bool) -> Path:
    """4x4 grid: the 4 frames with the most extreme hand positions, then 12 evenly spaced frames."""
    n = len(hm)
    extremes = [
        int(np.where(hm, hx, np.inf).min(axis=1).argmin()),
        int(np.where(hm, hx, -np.inf).max(axis=1).argmax()),
        int(np.where(hm, hy, np.inf).min(axis=1).argmin()),
        int(np.where(hm, hy, -np.inf).max(axis=1).argmax()),
    ]
    frames = extremes + [int(i) for i in np.linspace(0, n - 1, 12)]
    labels = ["min x", "max x", "min y", "max y"] + [""] * 12

    cap = cv2.VideoCapture(str(video))
    tiles = []
    for f, label in zip(frames, labels):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"{video}: could not read frame {f}")
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 1)
        for x, y in zip(hx[f][hm[f]], hy[f][hm[f]]):
            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
        cv2.putText(img, f"{f} {label}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        tiles.append(cv2.resize(img, (256, 256)))
    cap.release()

    grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(4)])
    out = REPO_ROOT / "reports" / "nectec_crop" / f"{video_id}{'_clip' if clip else ''}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    return out


def overlay_gt(video_id: str, config: dict) -> None:
    """Add GT_SIGN / GT_SENTENCE tiers to the segmenter's .eaf, so ELAN stacks prediction
    and ground truth on one timeline.

    Values stay empty: gloss text is dropped at parse time and never written (Q1/Q3).
    GT_SENTENCE comes from the `Gloss` tier, whose timing follows the speech, not the
    signing -- it marks the annotated stretches, so do not read it as sentence truth.
    Idempotent: existing GT_* tiers are removed first, so re-running never duplicates.
    """
    out_dir = REPO_ROOT / config["output_root"] / video_id
    eaf_path = out_dir / f"{video_id}.eaf"
    gt = json.loads((out_dir / "ground_truth.json").read_text(encoding="utf-8"))
    fps = gt["fps"]

    eaf = pympi.Elan.Eaf(str(eaf_path))
    tiers = {
        "GT_SIGN": [(s["start_frame"], s["end_frame"]) for s in gt["segments"]],
        "GT_SENTENCE": [tuple(sp) for sp in gt["scored_spans"]],
    }
    for name, spans in tiers.items():
        if name in eaf.get_tier_names():
            eaf.remove_tier(name)
        eaf.add_tier(name)
        for start, end in spans:
            start_ms, end_ms = round(start * 1000 / fps), round(end * 1000 / fps)
            if end_ms > start_ms:
                eaf.add_annotation(name, start_ms, end_ms)
    eaf.to_file(str(eaf_path))
    eaf_path.with_suffix(".bak").unlink(missing_ok=True)
    print(f"  [overlay] GT_SIGN {len(tiers['GT_SIGN'])}, GT_SENTENCE {len(tiers['GT_SENTENCE'])} -> {eaf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prep", "verify", "segment", "overlay"])
    parser.add_argument("--video", help="one video_id; default is config `run`")
    parser.add_argument("--clip", action="store_true", help=f"{CLIP_LEN_S}s excerpt from {CLIP_START_S}s (prep/verify)")
    parser.add_argument("--force", action="store_true", help="ignore the crop/pose cache")
    parser.add_argument("--config", default=str(REPO_ROOT / "config_nectec.yaml"))
    args = parser.parse_args()

    config = load_config(Path(args.config))
    video_ids = [args.video] if args.video else config["run"]

    failed = {}
    results = []
    for video_id in video_ids:
        print(f"[{video_id}] {args.stage}{' (clip)' if args.clip else ''}")
        if args.stage == "prep":
            prep(video_id, config, args.clip, args.force)
        elif args.stage == "verify":
            r = verify(video_id, config, args.clip)
            results.append(r)
            if r["failures"]:
                failed[video_id] = r["failures"]
        elif args.stage == "segment":
            try:
                run_video(video_id, config, args.force)  # not pipeline.main(): that rewrites reports/phase1_review.md
                overlay_gt(video_id, config)
            except RuntimeError as e:
                failed[video_id] = [str(e)]
        else:
            overlay_gt(video_id, config)

    if args.stage == "verify":
        out = REPO_ROOT / "reports" / "nectec_crop" / f"verify{'_clip' if args.clip else ''}.json"
        merged = {r["video_id"]: r for r in json.loads(out.read_text(encoding="utf-8"))} if out.exists() else {}
        merged.update({r["video_id"]: r for r in results})  # --video reruns must not drop the other videos
        out.write_text(json.dumps(list(merged.values()), indent=2), encoding="utf-8")
        print(f"\nverify results -> {out}")
    if failed:
        raise SystemExit(f"FAILED: {json.dumps(failed, indent=2)}")


if __name__ == "__main__":
    main()
