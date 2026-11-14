"""Irodori-TTS v4.1-Small-MF ONNX Runtime (CPU) inference pipeline.

Mirrors the official PyTorch flow (irodori_tts/inference_runtime.py) using the
split ONNX graphs exported for the Onsei app:
  text_encoder -> speaker_encoder -> duration -> context_kv
  -> MeanFlow sampling with dit_step (4 steps) -> dacvae_decode -> tail trim.

Token ids come from a token JSON produced by scripts/tokenize_text.py using the
official tokenizer/normalizer (text: rare-token split ON, caption: split OFF,
exactly as the official InferenceRuntime does).

Model layout (see README.md / SKILL.md for download sources):
  <models-dir>/int8/  text_encoder_int8.onnx, dit_step_int8.onnx, and the
                      shared float32 graphs: duration.onnx, context_kv.onnx,
                      speaker_encoder.onnx, dacvae_decode.onnx
  <models-dir>/f32/   text_encoder.onnx, dit_step.onnx, dacvae_encode.onnx

<models-dir> defaults to <repo-root>/models, overridable with --models-dir
or the IRODORI_MODELS_DIR environment variable.

Usage:
  python3 scripts/infer_onnx.py --variant int8 --tokens tokens.json --out out.wav
  python3 scripts/infer_onnx.py --variant int8 --tokens tokens.json \
      --ref-latents ref/my-voice.npz --out out.wav

Options:
  --tokens PATH           token JSON to use (default: ./tokens.json)
  --truncation-factor F   scale the initial Gaussian noise by F right after it is
                          drawn, mirroring irodori_tts/rf.py (x_t = x_t * factor).
                          Unset = disabled (bit-identical to factor 1.0).
                          0.85 was a good quality/stability balance in our tests.
  --ref-wavs P [P ...]    reference wav(s) for speaker conditioning. Each clip is
                          loudness-normalized (pyloudnorm BS.1770, target
                          --ref-normalize-db, official default -16 dB, plus the
                          peak<=1.0 safety scaling) and encoded with
                          dacvae_encode.onnx; latents are concatenated in order
                          and capped at ref_max_seconds (config: 120 s = 3000
                          frames), mirroring InferenceRuntime._encode_reference_wavs
                          / _load_reference_latent. Unset = no-ref path.
  --ref-latents NPZ       pre-encoded reference latent saved as npz (key
                          "latent", shape (1,T,32) float32, as produced by
                          scripts/make_ref_latents.py). Skips dacvae_encode
                          entirely and feeds the stored latent to
                          speaker_encoder. Mutually exclusive with --ref-wavs.
                          This is the recommended way to "fix" a voice: encode
                          once, reuse forever (fast and deterministic).
"""
import argparse, json, math, os, resource, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 42
NUM_STEPS = 4          # MeanFlow default for v4.1-Small-MF
HOP = 1920             # codec hop_length (48kHz / 25fps latent)
SR = 48000
MIN_SEC, MAX_SEC = 0.5, 30.0

FILES = {
    # variant: (encoder file, dit file); the other four graphs are shared float32.
    'int8': ('text_encoder_int8.onnx', 'dit_step_int8.onnx'),
    'f32': ('text_encoder.onnx', 'dit_step.onnx'),
}


def model_dirs(models_dir: Path):
    dirs = {'int8': models_dir / 'int8', 'f32': models_dir / 'f32'}
    shared = models_dir / 'int8'  # duration/context_kv/speaker_encoder/dacvae_decode (float32)
    return dirs, shared


def find_flattening_point(latent: np.ndarray, window_size: int = 20,
                          std_threshold: float = 0.05, mean_threshold: float = 0.1) -> int:
    """Numpy mirror of irodori_tts.inference_runtime.find_flattening_point."""
    total = latent.shape[0]
    padded = np.concatenate([latent, np.zeros((window_size, latent.shape[1]), np.float32)], axis=0)
    for i in range(padded.shape[0] - window_size):
        w = padded[i:i + window_size]
        if w.std() < std_threshold and abs(w.mean()) < mean_threshold:
            return i
    return total


def encode_reference_latents(enc_sess, wav_paths, normalize_db, max_ref_seconds,
                             n_avg=4):
    """Mirror InferenceRuntime._encode_reference_wavs + the latent cap in
    _load_reference_latent. Returns (ref_latent (1,T,32) float32, info list).

    Note: dacvae_encode.onnx implements the DACVAE posterior *sampling* path
    (two runs on the same input differ), while the official runtime encodes
    references with the deterministic posterior *mean*. Averaging n_avg
    independent encodes per clip approximates that mean (measured: latent corr
    vs official 0.90 @K=1 -> 0.97 @K=4)."""
    import pyloudnorm as pyln
    from scipy.signal import resample_poly
    max_ref_frames = max(1, math.ceil(max_ref_seconds * SR / HOP)) if max_ref_seconds > 0 else None
    pieces, info = [], []
    for path in wav_paths:
        data, sr = sf.read(path, dtype='float32', always_2d=True)  # (T, C)
        wav = data.mean(axis=1) if data.shape[1] != 1 else data[:, 0]
        if len(wav_paths) == 1 and max_ref_seconds > 0:
            cap = max(1, int(max_ref_seconds * sr))
            if wav.shape[0] > cap:
                wav = wav[:cap]
        if sr != SR:  # official resamples (torchaudio sinc) before normalizing
            g = math.gcd(int(sr), SR)
            wav = resample_poly(wav.astype(np.float64), SR // g, int(sr) // g).astype(np.float32)
        loud_before = None
        if normalize_db is not None:
            meter = pyln.Meter(SR)
            loud_before = float(meter.integrated_loudness(wav.astype(np.float64)))
            gain = 10.0 ** ((float(normalize_db) - loud_before) / 20.0)
            wav = (wav * np.float32(gain)).astype(np.float32)
            peak = float(np.abs(wav).max()) if wav.size else 0.0
            if peak > 1.0:  # audiotools ensure_max_of_audio(1.0)
                wav = (wav * np.float32(1.0 / peak)).astype(np.float32)
        latent = None
        for _ in range(max(1, int(n_avg))):
            run = enc_sess.run(None, {'waveform': wav[None, None, :]})[0]
            latent = run if latent is None else latent + run
        latent = (latent / max(1, int(n_avg))).astype(np.float32)
        info.append({'path': str(path), 'sr': int(sr), 'seconds': float(wav.shape[0] / SR),
                     'loudness_before_db': loud_before, 'frames': int(latent.shape[1]),
                     'encode_runs': int(n_avg),
                     'latent_mean': float(latent.mean()), 'latent_std': float(latent.std())})
        pieces.append(latent)
        if max_ref_frames is not None and sum(p.shape[1] for p in pieces) >= max_ref_frames:
            break
    ref_latent = np.concatenate(pieces, axis=1)
    if max_ref_frames is not None and ref_latent.shape[1] > max_ref_frames:
        ref_latent = ref_latent[:, :max_ref_frames]
    return ref_latent, info


def load_sessions(models_dir: Path, variant: str = 'int8', threads: int = 2) -> tuple[dict, dict[str, float]]:
    """Load all ONNX sessions once for single or batch inference."""
    dirs, shared = model_dirs(models_dir)
    timings: dict[str, float] = {}

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1

    def load(path: Path) -> ort.InferenceSession:
        t0 = time.perf_counter()
        s = ort.InferenceSession(str(path), so, providers=['CPUExecutionProvider'])
        timings[f'load:{path.name}'] = time.perf_counter() - t0
        return s

    enc_name, dit_name = FILES[variant]
    t0 = time.perf_counter()
    sessions = {
        'enc': load(dirs[variant] / enc_name),
        'spk': load(shared / 'speaker_encoder.onnx'),
        'dur': load(shared / 'duration.onnx'),
        'ctx': load(shared / 'context_kv.onnx'),
        'dit': load(dirs[variant] / dit_name),
        'dec': load(shared / 'dacvae_decode.onnx'),
        '_so': so,
        '_dirs': dirs,
    }
    timings['load_total'] = time.perf_counter() - t0
    return sessions, timings


def encode_caption_state(enc_sess: ort.InferenceSession, caption_ids: list[int], caption_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode caption token ids once so multiple segments can reuse the state."""
    cap_ids = np.array([caption_ids], dtype=np.int64)
    cap_mask = np.ones((1, caption_len), dtype=np.float32)
    caption_state = enc_sess.run(['caption_state'], {'input_ids': cap_ids, 'mask': cap_mask})[0]
    return caption_state, cap_mask


def prepare_speaker_condition(
    sessions: dict,
    ref_latents: str | Path | None = None,
    ref_wavs: list[str] | None = None,
    ref_normalize_db: float = -16.0,
    ref_max_seconds: float = 120.0,
) -> dict:
    """Compute speaker conditioning tensors once for reuse across segments."""
    timings: dict[str, float] = {}
    ref_info = None
    ref_latent = None
    spk = sessions['spk']

    if ref_latents:
        t0 = time.perf_counter()
        with np.load(ref_latents) as z:
            ref_latent = np.ascontiguousarray(z['latent'], dtype=np.float32)
            ref_info = {'source': 'ref_latents_npz', 'path': str(ref_latents),
                        'total_frames': int(ref_latent.shape[1]),
                        'clip_frames': ([int(v) for v in z['clip_frames']]
                                        if 'clip_frames' in z.files else None),
                        'clips': ([str(v) for v in z['clip_paths']]
                                  if 'clip_paths' in z.files else None)}
        timings['load_ref_latents'] = time.perf_counter() - t0
    elif ref_wavs:
        t0 = time.perf_counter()
        enc_ref = ort.InferenceSession(
            str(sessions['_dirs']['f32'] / 'dacvae_encode.onnx'),
            sessions['_so'],
            providers=['CPUExecutionProvider'],
        )
        ref_latent, ref_info = encode_reference_latents(
            enc_ref, ref_wavs, ref_normalize_db, ref_max_seconds)
        timings['encode_reference_latents'] = time.perf_counter() - t0

    if ref_latent is not None:
        t0 = time.perf_counter()
        ref_mask = np.ones(ref_latent.shape[:2], dtype=np.float32)
        speaker_state, speaker_mask = spk.run(None, {'ref_latent': ref_latent, 'mask': ref_mask})
        timings['encode_speaker(ref)'] = time.perf_counter() - t0
        has_speaker = np.ones(1, dtype=np.float32)
    else:
        t0 = time.perf_counter()
        # no-ref: official runtime builds ref_latent = zeros(B, speaker_patch_size, latent_dim), mask all False
        speaker_state, speaker_mask = spk.run(None, {
            'ref_latent': np.zeros((1, 4, 32), dtype=np.float32),
            'mask': np.zeros((1, 4), dtype=np.float32)})
        timings['encode_speaker(noref)'] = time.perf_counter() - t0
        has_speaker = np.zeros(1, dtype=np.float32)

    return {
        'speaker_state': speaker_state,
        'speaker_mask': speaker_mask,
        'has_speaker': has_speaker,
        'ref_latent': ref_latent,
        'ref_info': ref_info,
        'timings': timings,
    }


def synthesize_waveform(
    sessions: dict,
    tok: dict,
    speaker_cond: dict,
    seed: int = SEED,
    truncation_factor: float | None = None,
    precomputed_caption: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, float], dict]:
    """Synthesize a single segment waveform using loaded sessions and cached conditions."""
    enc = sessions['enc']
    dur = sessions['dur']
    ctx = sessions['ctx']
    dit = sessions['dit']
    dec = sessions['dec']

    timings: dict[str, float] = {}
    t_gen = time.perf_counter()

    # --- text (+ caption if not precomputed) ---
    t0 = time.perf_counter()
    text_ids = np.array([tok['text_ids']], dtype=np.int64)
    text_mask = np.ones((1, tok['text_len']), dtype=np.float32)
    text_state = enc.run(['text_state'], {'input_ids': text_ids, 'mask': text_mask})[0]
    if precomputed_caption is None:
        caption_state, cap_mask = encode_caption_state(enc, tok['caption_ids'], tok['caption_len'])
        timings['encode_text+caption'] = time.perf_counter() - t0
    else:
        caption_state, cap_mask = precomputed_caption
        timings['encode_text'] = time.perf_counter() - t0

    speaker_state = speaker_cond['speaker_state']
    speaker_mask = speaker_cond['speaker_mask']
    has_speaker = speaker_cond['has_speaker']

    # --- duration ---
    t0 = time.perf_counter()
    log_frames = float(dur.run(None, {
        'text_state': text_state, 'text_mask': text_mask,
        'speaker_state': speaker_state, 'has_speaker': has_speaker,
        'caption_state': caption_state, 'caption_mask': cap_mask,
        'has_caption': np.ones(1, dtype=np.float32)})[0][0])
    pred_frames = math.expm1(log_frames)
    latent_steps = int(round(pred_frames))
    latent_steps = max(int(math.ceil(MIN_SEC * 25)), min(int(MAX_SEC * 25), latent_steps))
    target_samples = latent_steps * HOP
    timings['duration'] = time.perf_counter() - t0

    # --- context KV cache ---
    t0 = time.perf_counter()
    kv_out = ctx.run(None, {'text_state': text_state, 'speaker_state': speaker_state,
                            'caption_state': caption_state})
    kv = {o.name: a for o, a in zip(ctx.get_outputs(), kv_out)}
    timings['context_kv'] = time.perf_counter() - t0

    # --- MeanFlow sampling (sample_euler_meanflow equivalent) ---
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, latent_steps, 32)).astype(np.float32)
    # Truncation (official rf.py: x_t = x_t * truncation_factor, applied to the
    # initial noise before sampling). Only when explicitly requested, so the
    # default path stays bit-identical to before.
    if truncation_factor is not None:
        if truncation_factor <= 0:
            raise ValueError(f"truncation_factor must be > 0, got {truncation_factor}")
        x = x * np.float32(truncation_factor)
    schedule = np.linspace(1.0, 0.0, NUM_STEPS + 1, dtype=np.float32)
    base_feed = {'text_mask': text_mask, 'speaker_mask': speaker_mask.astype(np.float32),
                 'caption_mask': cap_mask, 'speaker_kv_scales': np.ones(12, dtype=np.float32)}
    base_feed.update(kv)
    step_times = []
    for i in range(NUM_STEPS):
        t_cur, t_next = float(schedule[i]), float(schedule[i + 1])
        feed = dict(base_feed)
        feed['x_t'] = x
        feed['t'] = np.array([t_cur], dtype=np.float32)
        feed['delta_t'] = np.array([t_cur - t_next], dtype=np.float32)
        ts0 = time.perf_counter()
        v = dit.run(None, feed)[0]
        step_times.append(time.perf_counter() - ts0)
        x = x + v * (t_next - t_cur)
    timings['sample_meanflow(4 steps)'] = time.perf_counter() - t0
    timings['dit_step_each'] = step_times  # type: ignore[assignment]

    # --- decode + tail trim (official: trim on latent, cap at target_samples) ---
    t0 = time.perf_counter()
    wav = dec.run(None, {'latent': x})[0][0, 0]  # (samples,)
    fp = find_flattening_point(x[0])
    max_samples = target_samples
    if fp * HOP > 0:
        max_samples = min(max_samples, fp * HOP)
    wav = wav[:max_samples]
    timings['decode+trim'] = time.perf_counter() - t0
    timings['generation_total'] = time.perf_counter() - t_gen

    meta = {
        'log_frames': log_frames,
        'pred_frames': pred_frames,
        'latent_steps': latent_steps,
        'flattening_point': fp,
    }
    return wav, timings, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['int8', 'f32'], default='int8')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--models-dir', default=os.environ.get('IRODORI_MODELS_DIR'),
                    help='directory holding int8/ and f32/ model subdirs '
                         '(default: <repo-root>/models)')
    ap.add_argument('--tokens', default=None,
                    help='token json path (default: ./tokens.json)')
    ap.add_argument('--truncation-factor', type=float, default=None,
                    help='scale initial noise by this factor (official rf.py semantics); '
                         'unset = disabled')
    ap.add_argument('--ref-wavs', nargs='+', default=None,
                    help='reference wav path(s) for speaker conditioning; '
                         'unset = no-ref')
    ap.add_argument('--ref-latents', default=None,
                    help='npz with pre-encoded reference latent (key "latent", '
                         'shape (1,T,32)); skips dacvae_encode. Mutually '
                         'exclusive with --ref-wavs.')
    ap.add_argument('--ref-normalize-db', type=float, default=-16.0,
                    help='per-clip loudness target in dB (official ref_normalize_db)')
    ap.add_argument('--ref-max-seconds', type=float, default=120.0,
                    help='cap on combined reference length (config ref_max_seconds)')
    args = ap.parse_args()
    if args.ref_wavs and args.ref_latents:
        ap.error('--ref-wavs and --ref-latents are mutually exclusive')

    models_dir = Path(args.models_dir) if args.models_dir else REPO_ROOT / 'models'
    tok_path = Path(args.tokens) if args.tokens else Path('tokens.json')
    tok = json.loads(tok_path.read_text())

    t_all = time.perf_counter()
    sessions, load_timings = load_sessions(models_dir, variant=args.variant, threads=args.threads)
    speaker_cond = prepare_speaker_condition(
        sessions,
        ref_latents=args.ref_latents,
        ref_wavs=args.ref_wavs,
        ref_normalize_db=args.ref_normalize_db,
        ref_max_seconds=args.ref_max_seconds,
    )
    wav, synth_timings, meta = synthesize_waveform(
        sessions,
        tok,
        speaker_cond,
        seed=args.seed,
        truncation_factor=args.truncation_factor,
    )

    timings = {}
    timings.update(load_timings)
    timings.update(speaker_cond['timings'])
    timings.update(synth_timings)

    sf.write(args.out, wav, SR, subtype='PCM_16')
    timings['total_wall'] = time.perf_counter() - t_all
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    ref_latent = speaker_cond['ref_latent']
    report = {
        'variant': args.variant, 'seed': args.seed, 'out': args.out,
        'tokens': str(tok_path), 'truncation_factor': args.truncation_factor,
        'ref_wavs': args.ref_wavs, 'ref_latents': args.ref_latents,
        'ref_info': speaker_cond['ref_info'],
        'ref_total_frames': None if ref_latent is None else int(ref_latent.shape[1]),
        'speaker_len': int(speaker_cond['speaker_state'].shape[1]),
        'log_frames': meta['log_frames'], 'pred_frames': meta['pred_frames'],
        'latent_steps': meta['latent_steps'], 'flattening_point': meta['flattening_point'],
        'wav_seconds': len(wav) / SR, 'wav_peak': float(np.abs(wav).max()),
        'wav_rms': float(np.sqrt((wav ** 2).mean())),
        'peak_rss_mb': peak_kb / 1024.0, 'timings': timings,
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
