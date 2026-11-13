"""Synthesize long text in a single process by splitting, inferring, and joining.

Automates the reliable long-text workflow (see docs/findings.md):
1. Split long text at sentence boundaries into ~70-80 character segments.
2. Load tokenizers (fast torch-free path when tokenizer.json is available) and
   ONNX sessions once in a single process.
3. Precompute shared caption and speaker conditions once.
4. Synthesize each segment and concatenate with a fixed silence gap (default 0.4s).

Usage:
  .venv/bin/python scripts/speak_long_text.py \
      --irodori-dir <irodori-tts> \
      --caption "若い女性の声。明るく親しみやすく話す。" \
      --ref-latents ref/my-voice.npz \
      --truncation-factor 0.85 \
      --text "長い文章がここに続きます。文ごとに自動で分割されます。" \
      --out full.wav
"""
import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from infer_onnx import (  # noqa: E402
    SEED,
    SR,
    encode_caption_state,
    load_sessions,
    prepare_speaker_condition,
    synthesize_waveform,
)
from join_wavs import join_audio_arrays  # noqa: E402
from tokenize_text import load_tokenizers, tokenize_segment  # noqa: E402

DEFAULT_GAP = 0.4
DEFAULT_THREADS = min(6, max(2, (os.cpu_count() or 2) // 2))


def split_text_into_sentences(text: str, max_chars: int = 80) -> list[str]:
    """Split text into sentence-level segments (~50-80 chars max)."""
    text = text.strip()
    if not text:
        return []

    raw_sentences = re.split(r'(?<=[。！？!?\n])', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    refined = []
    for s in sentences:
        if len(s) <= max_chars:
            refined.append(s)
        else:
            sub_parts = re.split(r'(?<=[、,])', s)
            current = ""
            for p in sub_parts:
                p = p.strip()
                if not p:
                    continue
                if len(current) + len(p) <= max_chars:
                    current += p
                else:
                    if current:
                        refined.append(current)
                    current = p
            if current:
                refined.append(current)

    return refined


def main() -> None:
    ap = argparse.ArgumentParser(description="Split long text and synthesize in a single process")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Text to speak")
    group.add_argument("--text-file", help="Path to UTF-8 text file to speak")

    ap.add_argument("--caption", required=True, help="Voice caption")
    ap.add_argument("--out", required=True, help="Output wav file path")
    ap.add_argument("--variant", choices=["int8", "f32"], default="int8")
    ap.add_argument("--seed", type=int, default=SEED, help="Random seed (default 42)")
    ap.add_argument("--truncation-factor", type=float, default=None,
                    help="Scale initial noise by this factor (e.g. 0.85)")
    ap.add_argument("--gap", type=float, default=DEFAULT_GAP,
                    help="Silence gap between segments in seconds (default 0.4)")
    ap.add_argument("--max-chars", type=int, default=80, help="Max characters per segment")
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                    help="ONNX Runtime intra-op threads")
    ap.add_argument("--models-dir", default=os.environ.get("IRODORI_MODELS_DIR"),
                    help="Directory holding int8/ and f32/ model subdirs "
                         "(default: <repo-root>/models)")
    ap.add_argument("--irodori-dir", default=os.environ.get("IRODORI_TTS_DIR"),
                    help="Path to a checkout of Aratako/Irodori-TTS (or set IRODORI_TTS_DIR)")
    ap.add_argument("--tokenizer-dir", default=None,
                    help="Tokenizer directory (default: <models-dir>/int8/tokenizer if present, "
                         "else Hugging Face cache)")
    ap.add_argument("--ref-wavs", nargs="+", default=None,
                    help="Reference wav path(s) for speaker conditioning; unset = no-ref")
    ap.add_argument("--ref-latents", default=None,
                    help="Pre-encoded reference latent npz; mutually exclusive with --ref-wavs")
    ap.add_argument("--ref-normalize-db", type=float, default=-16.0,
                    help="Per-clip loudness target in dB (official ref_normalize_db)")
    ap.add_argument("--ref-max-seconds", type=float, default=120.0,
                    help="Cap on combined reference length (config ref_max_seconds)")
    ap.add_argument("--keep-parts", action="store_true", help="Keep individual segment wav files")
    args = ap.parse_args()

    if not args.irodori_dir:
        ap.error("--irodori-dir (or IRODORI_TTS_DIR) is required")
    if args.ref_wavs and args.ref_latents:
        ap.error("--ref-wavs and --ref-latents are mutually exclusive")

    text = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else args.text
    segments = split_text_into_sentences(text, max_chars=args.max_chars)
    if not segments:
        print("Error: No valid text segments found.", file=sys.stderr)
        sys.exit(1)

    models_dir = Path(args.models_dir) if args.models_dir else REPO_ROOT / "models"
    tokenizer_dir = args.tokenizer_dir
    if tokenizer_dir is None:
        bundled = models_dir / "int8" / "tokenizer"
        if bundled.is_dir():
            tokenizer_dir = str(bundled)

    print(f"Total {len(segments)} segments to synthesize:")
    for i, seg in enumerate(segments, 1):
        print(f"  [{i}/{len(segments)}] ({len(seg)} chars): {seg}")

    t_start = time.perf_counter()

    # 1. Load tokenizers once and tokenize all segments
    t0 = time.perf_counter()
    tok_bundle = load_tokenizers(args.irodori_dir, tokenizer_dir)
    pre_caption_ids = tok_bundle.encode_caption(args.caption)
    tokenized_segments = [
        tokenize_segment(
            tok_bundle,
            seg,
            args.caption,
            precomputed_caption=pre_caption_ids,
            include_duration_features=False,
        )
        for seg in segments
    ]
    t_tok = time.perf_counter() - t0

    # 2. Load ONNX sessions once and precompute caption + speaker conditions
    sessions, load_timings = load_sessions(models_dir, variant=args.variant, threads=args.threads)
    cap_ids_list, cap_len = pre_caption_ids
    precomputed_caption_state = encode_caption_state(sessions["enc"], cap_ids_list, cap_len)
    speaker_cond = prepare_speaker_condition(
        sessions,
        ref_latents=args.ref_latents,
        ref_wavs=args.ref_wavs,
        ref_normalize_db=args.ref_normalize_db,
        ref_max_seconds=args.ref_max_seconds,
    )

    # 3. Synthesize each segment in-process
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="irodori_parts_")) if args.keep_parts else None

    part_wavs = []
    t_synth_start = time.perf_counter()
    for i, tok in enumerate(tokenized_segments, 1):
        wav, seg_timings, _ = synthesize_waveform(
            sessions,
            tok,
            speaker_cond,
            seed=args.seed,
            truncation_factor=args.truncation_factor,
            precomputed_caption=precomputed_caption_state,
        )
        part_wavs.append(wav)
        if temp_dir is not None:
            part_path = temp_dir / f"part_{i:03d}.wav"
            sf.write(part_path, wav, SR, subtype="PCM_16")
        print(f"  Generated segment {i}/{len(segments)} ({seg_timings['generation_total']:.2f}s)")
    t_synth = time.perf_counter() - t_synth_start

    # 4. Join wavs and write final output
    full_wav = part_wavs[0] if len(part_wavs) == 1 else join_audio_arrays(part_wavs, SR, gap_sec=args.gap)
    sf.write(out_path, full_wav, SR, subtype="PCM_16")

    if temp_dir is not None:
        print(f"Preserved part wavs in: {temp_dir}")

    t_total = time.perf_counter() - t_start
    print(
        f"\nSuccessfully generated full audio: {out_path} "
        f"(total {t_total:.2f}s: tok {t_tok:.2f}s, load {load_timings['load_total']:.2f}s, synth {t_synth:.2f}s)"
    )


if __name__ == "__main__":
    main()

