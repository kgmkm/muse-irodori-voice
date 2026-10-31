"""Encode reference clips into a reusable "fixed voice" latent npz.

Reuses infer_onnx.encode_reference_latents (same preprocessing as the
--ref-wavs path: 48kHz mono, pyloudnorm -16 LUFS + ensure_max, then
dacvae_encode.onnx). The ONNX encoder follows the DACVAE posterior
*sampling* path, so each clip is encoded N_AVG times and averaged to
approximate the official posterior mean. For a one-off fixed reference,
8 averages (default here) sit closer to the mean than the 4 used for
interactive --ref-wavs runs.

Usage:
  python3 scripts/make_ref_latents.py \
      --clips clip_pos.wav clip_neg.wav clip_pos2.wav \
      --out ref/my-voice.npz

Output npz keys:
  latent       (1, T, 32) float32 concatenated reference latent
  clip_paths   (K,) unicode, clip paths in order
  clip_frames  (K,) int64, frames contributed per clip
  total_frames () int64
  meta_json    () unicode, JSON with encode settings / per-clip stats

Tip (see docs/findings.md): mix clips with different emotional registers
(e.g. one cheerful take + one worried take, ~30 s total). A single cheerful
reference drags even sad text toward a cheerful delivery ("emotion leak");
a mixed set keeps the identity while leaving emotion to the text/caption.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_onnx import encode_reference_latents, model_dirs  # noqa: E402
import onnxruntime as ort  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', nargs='+', required=True,
                    help='reference wav clips, concatenated in this order')
    ap.add_argument('--out', required=True, help='output npz path')
    ap.add_argument('--n-avg', type=int, default=8,
                    help='encodes averaged per clip (default 8)')
    ap.add_argument('--normalize-db', type=float, default=-16.0)
    ap.add_argument('--max-ref-seconds', type=float, default=120.0)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--models-dir', default=os.environ.get('IRODORI_MODELS_DIR'),
                    help='directory holding int8/ and f32/ model subdirs '
                         '(default: <repo-root>/models)')
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    models_dir = Path(args.models_dir) if args.models_dir else repo_root / 'models'
    dirs, _shared = model_dirs(models_dir)

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    enc = ort.InferenceSession(str(dirs['f32'] / 'dacvae_encode.onnx'),
                               so, providers=['CPUExecutionProvider'])
    t0 = time.perf_counter()
    latent, info = encode_reference_latents(enc, args.clips, args.normalize_db,
                                            args.max_ref_seconds, n_avg=args.n_avg)
    encode_seconds = time.perf_counter() - t0
    meta = {
        'created_by': 'scripts/make_ref_latents.py',
        'encode_runs_per_clip': args.n_avg,
        'normalize_db': args.normalize_db,
        'max_ref_seconds': args.max_ref_seconds,
        'encode_seconds': encode_seconds,
        'note': ('Reference latents are re-encodable but sampling-based: '
                 're-running this script yields a statistically equivalent, '
                 'not identical, latent. Once saved, the npz IS the fixed '
                 'voice; keep it unchanged for reproducible identity.'),
        'per_clip': info,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        latent=latent.astype(np.float32),
        clip_paths=np.array([str(c) for c in args.clips]),
        clip_frames=np.array([i['frames'] for i in info], dtype=np.int64),
        total_frames=np.int64(latent.shape[1]),
        meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    print(json.dumps({
        'out': str(out), 'latent_shape': list(latent.shape),
        'total_frames': int(latent.shape[1]),
        'clip_frames': [i['frames'] for i in info],
        'encode_seconds': encode_seconds,
        'latent_mean': float(latent.mean()), 'latent_std': float(latent.std()),
    }, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
