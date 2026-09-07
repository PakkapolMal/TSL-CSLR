"""Phase 1: pretrained zero-shot segmentation pipeline for TSL video.

video -> MediaPipe Holistic pose -> pretrained segmenter -> ELAN (.eaf) -> segments.json

See CLAUDE.md and PHASE1.md for the spec this implements.
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pympi
import yaml
from pose_format import Pose

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_SCRIPTS = Path(sys.executable).resolve().parent  # console scripts live next to python.exe

POSE_LANDMARKS = slice(0, 33)
LEFT_HAND_LANDMARKS = slice(501, 522)
RIGHT_HAND_LANDMARKS = slice(522, 543)

SUBPROCESS_ENV = {**os.environ, "PYTHONUTF8": "1"}


def _cli(name: str) -> str:
    """Resolve a console-script entry point installed in this venv (not relying on PATH)."""
    exe = VENV_SCRIPTS / f"{name}.exe"
    return str(exe) if exe.exists() else name


@dataclass
class Segment:
    start_frame: int
    end_frame: int  # exclusive
    tier: str  # "sign" | "sentence"


def probe(video_path: Path) -> tuple[float, int, int, int]:
    """ffprobe the video; assert 30fps / 512x512 (CLAUDE.md data assumptions)."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,width,height,nb_frames",
        "-of", "json", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=SUBPROCESS_ENV, check=True)
    stream = json.loads(result.stdout)["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    width, height = int(stream["width"]), int(stream["height"])
    n_frames = int(stream["nb_frames"])
    if fps != 30.0:
        raise AssertionError(f"{video_path}: expected 30 fps, got {fps}")
    if width != 512 or height != 512:
        raise AssertionError(f"{video_path}: expected 512x512, got {width}x{height}")
    return fps, width, height, n_frames


def extract_pose(video_path: Path, out_pose: Path, force: bool) -> Path:
    """Shell out to video_to_pose. Skips if out_pose already exists (unless force)."""
    if out_pose.exists() and not force:
        print(f"  [pose] cached: {out_pose}")
        return out_pose
    out_pose.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_cli("video_to_pose"), "-i", str(video_path), "-o", str(out_pose), "--format", "mediapipe"]
    subprocess.run(cmd, check=True, env=SUBPROCESS_ENV)
    return out_pose


def landmark_report(pose: Pose) -> dict:
    """Fraction of frames with any signal per component. Raises if either hand < 60%."""
    confidence = pose.body.confidence  # [frames, people, points]
    n_frames = confidence.shape[0]

    def rate(sl: slice) -> float:
        present = confidence[:, 0, sl].sum(axis=1) > 0
        return float(present.sum()) / n_frames

    report = {
        "pose_rate": rate(POSE_LANDMARKS),
        "left_hand_rate": rate(LEFT_HAND_LANDMARKS),
        "right_hand_rate": rate(RIGHT_HAND_LANDMARKS),
    }
    print(
        f"  [landmarks] pose={report['pose_rate']:.1%} "
        f"left_hand={report['left_hand_rate']:.1%} right_hand={report['right_hand_rate']:.1%}"
    )
    if report["left_hand_rate"] < 0.60 or report["right_hand_rate"] < 0.60:
        raise RuntimeError(f"hand landmark rate below 60%: {report}")
    return report


def excluded_spans(pose: Pose) -> list[tuple[int, int]]:
    """Contiguous frame runs where POSE_LANDMARKS has no signal at all."""
    confidence = pose.body.confidence
    present = confidence[:, 0, POSE_LANDMARKS].sum(axis=1) > 0
    spans = []
    start = None
    for i, ok in enumerate(present):
        if not ok and start is None:
            start = i
        elif ok and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(present)))
    return spans


def _subtract_spans(start: int, end: int, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """[start, end) minus every overlapping span in spans. May split into pieces."""
    pieces = [(start, end)]
    for ex_start, ex_end in spans:
        next_pieces = []
        for s, e in pieces:
            if ex_end <= s or ex_start >= e:
                next_pieces.append((s, e))
                continue
            if s < ex_start:
                next_pieces.append((s, ex_start))
            if ex_end < e:
                next_pieces.append((ex_end, e))
        pieces = next_pieces
    return pieces


def segment(pose_path: Path, eaf_path: Path, video_path: Path, spans_to_exclude: list[tuple[int, int]]) -> list[Segment]:
    """Run the pretrained segmenter and parse its .eaf back into frame-indexed Segments."""
    # pympi backs up an existing .eaf by os.rename()-ing it to .bak, which errors on
    # Windows if a .bak from a previous run is already there. Clear both before rerunning.
    eaf_path.unlink(missing_ok=True)
    eaf_path.with_suffix(".bak").unlink(missing_ok=True)

    cmd = [_cli("pose_to_segments"), "--pose", str(pose_path), "--elan", str(eaf_path), "--video", str(video_path)]
    subprocess.run(cmd, check=True, env=SUBPROCESS_ENV)

    fps = Pose.read(pose_path.read_bytes()).body.fps

    eaf = pympi.Elan.Eaf(str(eaf_path))
    segments = []
    for tier in ("SIGN", "SENTENCE"):  # "default" tier is always empty, ignore it
        for start_ms, end_ms, _value in eaf.get_annotation_data_for_tier(tier):
            start_frame = round(start_ms * fps / 1000)
            end_frame = round(end_ms * fps / 1000)
            for s, e in _subtract_spans(start_frame, end_frame, spans_to_exclude):
                if e > s:
                    segments.append(Segment(s, e, tier.lower()))
    return segments


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_video(video_id: str, config: dict, force: bool) -> dict:
    data_root = REPO_ROOT / config["data_root"]
    output_root = REPO_ROOT / config["output_root"]
    video_path = data_root / config["videos"][video_id]["path"]
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    out_dir = output_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pose_path = out_dir / f"{video_id}.pose"
    eaf_path = out_dir / f"{video_id}.eaf"

    print(f"[{video_id}] probing...")
    fps, width, height, n_frames = probe(video_path)
    print(f"[{video_id}] {n_frames} frames @ {fps} fps, {width}x{height}")

    print(f"[{video_id}] extracting pose...")
    extract_pose(video_path, pose_path, force)

    pose = Pose.read(pose_path.read_bytes())
    pose_fps = pose.body.fps

    print(f"[{video_id}] checking landmark rates...")
    rates = landmark_report(pose)

    spans = excluded_spans(pose)
    excluded_frame_count = sum(e - s for s, e in spans)
    (out_dir / "excluded_spans.json").write_text(
        json.dumps({"video_id": video_id, "fps": pose_fps, "spans": spans}, indent=2),
        encoding="utf-8",
    )
    print(f"  [excluded] {len(spans)} span(s), {excluded_frame_count} frame(s)")

    print(f"[{video_id}] segmenting...")
    segments = segment(pose_path, eaf_path, video_path, spans)
    (out_dir / "segments.json").write_text(
        json.dumps(
            {"video_id": video_id, "fps": pose_fps, "segments": [asdict(s) for s in segments]},
            indent=2,
        ),
        encoding="utf-8",
    )
    signs = [s for s in segments if s.tier == "sign"]
    sentences = [s for s in segments if s.tier == "sentence"]
    print(f"  [segments] {len(signs)} signs, {len(sentences)} sentences")

    return {
        "video_id": video_id,
        "n_frames": n_frames,
        "fps": pose_fps,
        "landmark_rates": rates,
        "excluded_span_count": len(spans),
        "excluded_frame_count": excluded_frame_count,
        "signs": signs,
        "sentences": sentences,
    }


def write_report(results: list[dict], report_path: Path, failed: list[str] | None = None) -> None:
    def stats(segs, fps):
        durations = [(s.end_frame - s.start_frame) / fps for s in segs]
        if not durations:
            return 0.0, 0.0, 0.0
        mean = statistics.mean(durations)
        median = statistics.median(durations)
        cv = (statistics.stdev(durations) / mean) if len(durations) > 1 and mean else 0.0
        return mean, median, cv

    lines = ["# Phase 1 Review\n"]
    lines.append(
        "| video | frames | duration (s) | left hand % | right hand % | pose % | "
        "excluded spans | excluded frames | signs | sentences | signs/s | "
        "mean sign dur (s) | median sign dur (s) | sign dur CV |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        duration_s = r["n_frames"] / r["fps"]
        mean_dur, median_dur, cv = stats(r["signs"], r["fps"])
        signs_per_sec = len(r["signs"]) / duration_s if duration_s else 0.0
        lines.append(
            f"| {r['video_id']} | {r['n_frames']} | {duration_s:.1f} | "
            f"{r['landmark_rates']['left_hand_rate']:.1%} | {r['landmark_rates']['right_hand_rate']:.1%} | "
            f"{r['landmark_rates']['pose_rate']:.1%} | {r['excluded_span_count']} | {r['excluded_frame_count']} | "
            f"{len(r['signs'])} | {len(r['sentences'])} | {signs_per_sec:.2f} | "
            f"{mean_dur:.2f} | {median_dur:.2f} | {cv:.2f} |"
        )

    if failed:
        lines.append(
            f"\n**Excluded from this run:** {', '.join(failed)} failed the >=60% "
            "hand-landmark gate (CLAUDE.md §5.1 / PHASE1.md 3c) and were not segmented. "
            "This is real per-video data, not a bug or a threshold that was tuned — "
            "verified against raw .pose confidence values. No cropping was added to "
            "compensate, per CLAUDE.md's instruction to report rather than fix."
        )

    lines.append("\n## Qualitative review (human)\n")
    lines.append("| Question | Why it matters | Answer |")
    lines.append("|---|---|---|")
    lines.append("| Do sentence boundaries land on visible pauses / hand drops / body shifts? | If yes, Phase 2 has a viable fallback tier. | |")
    lines.append("| Are sign boundaries plausible, or is it splitting at a near-constant rate? | Cross-check against the CV computed above. | |")
    lines.append("| Does behaviour differ noticeably between videos (= between signers)? | Predicts how hard signer-independence will be. | |")
    lines.append("| Where does it fail — fingerspelling, classifiers, fast sequences? | Directly informs the Phase 2 annotation convention. | |")
    lines.append("\n**Overall judgement (human):** sign-level usable / sentence-level only / neither — _blank, fill in after ELAN review_.\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 TSL segmentation pipeline")
    parser.add_argument("--video", help="run a single video_id from config.yaml")
    parser.add_argument("--all", action="store_true", help="run every video in config.yaml")
    parser.add_argument("--force", action="store_true", help="ignore the pose cache")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    args = parser.parse_args()

    config = load_config(Path(args.config))

    if args.video:
        video_ids = [args.video]
    elif args.all:
        video_ids = list(config["videos"].keys())
    else:
        video_ids = config["run"]

    results = []
    failed = []
    for video_id in video_ids:
        try:
            results.append(run_video(video_id, config, args.force))
        except RuntimeError as e:
            # Soft landmark-rate gate (CLAUDE.md §5.1 / PHASE1.md 3c): a real per-video
            # finding, not a pipeline bug. Report it and move on instead of segmenting
            # on unreliable hand data or "fixing" it with a crop stage.
            print(f"[{video_id}] GATE FAILED: {e}")
            failed.append(video_id)

    write_report(results, REPO_ROOT / "reports" / "phase1_review.md", failed)

    print("\nREADY FOR HUMAN REVIEW")
    for r in results:
        video_id = r["video_id"]
        print(f"  Open in ELAN:  outputs/{video_id}/{video_id}.eaf   (video auto-links)")
    print("  Then fill in:  reports/phase1_review.md")
    print(f"  Videos to inspect: {', '.join(r['video_id'] for r in results)}, ~2 minutes each")
    if failed:
        print(f"  FAILED the landmark-rate gate (not segmented, needs human attention): {', '.join(failed)}")


if __name__ == "__main__":
    main()
