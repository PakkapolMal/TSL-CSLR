"""Merge SIGN-tier segments separated by a small gap, for review/visualization only.

Does NOT touch outputs/{video_id}/segments.json (the pipeline's Phase 1 output).
Writes outputs/{video_id}/segments_merged.json instead.

Measured against NECTEC ground truth (2026-09-14): the max_gap=5 default LOWERS
segmental F1@0.50 from 0.718 to 0.644, and no gap value beats stock by more than
+0.001 when tuned honestly. See NECTEC_MERGE.md 4.1b. Keep this for eyeballing
short gaps in the viewer; do not treat its output as better segmentation.

ponytail: fixed --max-gap threshold with no ground truth to tune it against.
This is exploratory only, per CLAUDE.md Phase 1 (no threshold tuning). Revisit
once Phase 2 annotations exist to actually validate a gap value.
"""
import argparse
import json
from pathlib import Path


def merge(segments: list[dict], tier: str, max_gap: int) -> list[dict]:
    """Merge consecutive segments of `tier` when the gap between them is <= max_gap."""
    others = [s for s in segments if s["tier"] != tier]
    target = sorted((s for s in segments if s["tier"] == tier), key=lambda s: s["start_frame"])

    merged = []
    for seg in target:
        if merged and seg["start_frame"] - merged[-1]["end_frame"] <= max_gap:
            merged[-1]["end_frame"] = seg["end_frame"]
        else:
            merged.append(dict(seg))

    return sorted(others + merged, key=lambda s: s["start_frame"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--tier", default="sign", choices=["sign", "sentence"])
    parser.add_argument("--max-gap", type=int, default=5, help="frames; gaps <= this are merged")
    args = parser.parse_args()

    in_path = args.outputs_dir / args.video_id / "segments.json"
    data = json.loads(in_path.read_text(encoding="utf-8"))

    before = sum(1 for s in data["segments"] if s["tier"] == args.tier)
    data["segments"] = merge(data["segments"], args.tier, args.max_gap)
    after = sum(1 for s in data["segments"] if s["tier"] == args.tier)

    out_path = in_path.with_name("segments_merged.json")
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"{args.video_id}: {args.tier} {before} -> {after} segments (max_gap={args.max_gap}) -> {out_path}")


if __name__ == "__main__":
    main()
