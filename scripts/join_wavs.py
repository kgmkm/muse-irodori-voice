"""Concatenate segment wavs with a fixed silence gap.

Long texts generated in one shot can drop or insert words (see
docs/findings.md). The reliable workflow is: split the text at sentence
boundaries into ~70-80 character segments, synthesize each segment
separately, then join the pieces with this script.

Usage:
  python3 scripts/join_wavs.py part1.wav part2.wav --out full.wav
  python3 scripts/join_wavs.py part1.wav part2.wav --gap 0.4 --out full.wav

All inputs must share the same sample rate and channel count.
"""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf


def join_audio_arrays(pieces: list[np.ndarray], sr: int, gap_sec: float = 0.4) -> np.ndarray:
    """Concatenate waveform arrays along time axis with a fixed silence gap."""
    if not pieces:
        raise ValueError("pieces must contain at least one array")
    gap_samples = int(round(gap_sec * sr))
    if pieces[0].ndim == 1:
        gap = np.zeros((gap_samples,), dtype=np.float32)
    else:
        gap = np.zeros((gap_samples, pieces[0].shape[1]), dtype=np.float32)

    out_parts = []
    for i, piece in enumerate(pieces):
        if i:
            out_parts.append(gap)
        out_parts.append(piece)
    return np.concatenate(out_parts, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+', help='wav files in playback order')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gap', type=float, default=0.4,
                    help='silence gap between segments, seconds (default 0.4)')
    args = ap.parse_args()

    pieces = []
    sr_ref, ch_ref = None, None
    for path in args.inputs:
        data, sr = sf.read(path, dtype='float32', always_2d=True)
        if sr_ref is None:
            sr_ref, ch_ref = sr, data.shape[1]
        elif sr != sr_ref or data.shape[1] != ch_ref:
            raise SystemExit(
                f'{path}: format mismatch (sr={sr}, ch={data.shape[1]}; '
                f'expected sr={sr_ref}, ch={ch_ref})')
        pieces.append(data)

    joined = join_audio_arrays(pieces, sr_ref, gap_sec=args.gap)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, joined, sr_ref, subtype='PCM_16')
    print(f'{args.out}: {len(joined) / sr_ref:.2f}s '
          f'({len(pieces)} segments, gap {args.gap}s, sr {sr_ref})')


if __name__ == '__main__':
    main()
