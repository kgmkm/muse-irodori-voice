"""Tokenize text + caption with the official Irodori-TTS tokenizer.

Produces the token JSON consumed by scripts/infer_onnx.py.

IMPORTANT (measured pitfall): the official runtime enables rare-token
splitting ONLY on the text tokenizer. The caption tokenizer must NOT split.
Applying split to the caption measurably shifts duration prediction
(e.g. 189 -> 178 frames in our reference test). This script reproduces the
official setup exactly.

Requires a checkout of the official Irodori-TTS repository (for its
text_normalization / tokenizer modules) and the MF checkpoint's bundled
tokenizer, which the official repo's setup downloads into the Hugging Face
cache. Point --irodori-dir at the checkout (or set IRODORI_TTS_DIR); point
--tokenizer-dir at a tokenizer directory if you have one outside the cache.

Run with the Irodori-TTS venv python (needs torch), e.g.:
  /path/to/Irodori-TTS/.venv/bin/python scripts/tokenize_text.py \
      --text "こんにちは。今日もよろしくお願いします。" \
      --caption "若い女性の声。明るく親しみやすく話す。" \
      --out tokens.json
"""
import argparse, glob, json, os, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--text', required=True)
ap.add_argument('--caption', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--irodori-dir',
                default=os.environ.get('IRODORI_TTS_DIR'),
                help='path to a checkout of Aratako/Irodori-TTS '
                     '(or set IRODORI_TTS_DIR)')
ap.add_argument('--tokenizer-dir', default=None,
                help='tokenizer directory from the MF checkpoint; default: '
                     'look up the Hugging Face cache snapshot')
ap.add_argument('--max-text-len', type=int, default=256)
ap.add_argument('--max-caption-len', type=int, default=512)
args = ap.parse_args()

if not args.irodori_dir:
    ap.error('--irodori-dir (or IRODORI_TTS_DIR) is required')
sys.path.insert(0, str(Path(args.irodori_dir).expanduser()))

from irodori_tts.text_normalization import normalize_text        # noqa: E402
from irodori_tts.tokenizer import PretrainedTextTokenizer        # noqa: E402
from irodori_tts.duration import build_duration_features         # noqa: E402
import torch                                                     # noqa: E402

if args.tokenizer_dir:
    snap = str(Path(args.tokenizer_dir).expanduser())
else:
    hits = glob.glob(os.path.expanduser(
        '~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4.1-Small-MF'
        '/snapshots/*/tokenizer'))
    if not hits:
        ap.error('tokenizer not found in the Hugging Face cache; '
                 'run the official Irodori-TTS setup once, or pass --tokenizer-dir')
    snap = hits[0]

text_tok = PretrainedTextTokenizer.from_pretrained(
    repo_id=snap, add_bos=True, local_files_only=True)
text_tok.enable_rare_token_split()          # text: split ON (official)
cap_tok = PretrainedTextTokenizer.from_pretrained(
    repo_id=snap, add_bos=True, local_files_only=True)   # caption: split OFF

norm_text = normalize_text(args.text).strip()
text_ids, text_mask = text_tok.batch_encode([norm_text], max_length=args.max_text_len)
cap_ids, cap_mask = cap_tok.batch_encode([args.caption], max_length=args.max_caption_len)
feats = build_duration_features(
    [norm_text], token_counts=text_mask.sum(dim=1),
    max_text_len=args.max_text_len,
    has_speaker=torch.zeros((1,), dtype=torch.bool))
tl, cl = int(text_mask.sum()), int(cap_mask.sum())
out = {
    "normalized_text": norm_text,
    "text_ids": text_ids[0, :tl].tolist(), "text_len": tl,
    "caption_ids": cap_ids[0, :cl].tolist(), "caption_len": cl,
    "duration_features": feats[0].tolist(),
    "pad_id": text_tok.pad_token_id, "bos_id": text_tok.bos_token_id,
}
Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("out", args.out, "text_len", tl, "caption_len", cl)
print("text_ids", out["text_ids"])
print("caption_ids", out["caption_ids"])
