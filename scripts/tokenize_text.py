"""Tokenize text + caption with the official Irodori-TTS tokenizer.

Produces the token JSON consumed by scripts/infer_onnx.py.

IMPORTANT (measured pitfall): the official runtime enables rare-token
splitting ONLY on the text tokenizer. The caption tokenizer must NOT split.
Applying split to the caption measurably shifts duration prediction
(e.g. 189 -> 178 frames in our reference test). This script reproduces the
official setup exactly.

Requires a checkout of the official Irodori-TTS repository (for its
text_normalization / token_split modules) and the MF checkpoint's bundled
tokenizer (either downloaded into models/int8/tokenizer by download_models.sh
or cached by Hugging Face). Point --irodori-dir at the checkout (or set
IRODORI_TTS_DIR); point --tokenizer-dir at a tokenizer directory if you have
one outside the cache.

When tokenizer.json and tokenizer_config.json are present, this script uses
a fast torch-free path via `tokenizers` and direct submodule loading. It falls
back to PretrainedTextTokenizer (transformers + torch) otherwise.

Run with the venv python, e.g.:
  .venv/bin/python scripts/tokenize_text.py \
      --irodori-dir <irodori-tts> \
      --tokenizer-dir models/int8/tokenizer \
      --text "こんにちは。今日もよろしくお願いします。" \
      --caption "若い女性の声。明るく親しみやすく話す。" \
      --out tokens.json
"""
import argparse, glob, importlib.util, json, os, sys
from pathlib import Path


def _load_module_direct(module_name: str, file_path: Path):
    """Load a submodule directly by path without importing package __init__ (avoids heavy torch import)."""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_tokenizer_dir(tokenizer_dir: str | None) -> str:
    if tokenizer_dir:
        return str(Path(tokenizer_dir).expanduser())
    hits = glob.glob(os.path.expanduser(
        '~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4.1-Small-MF'
        '/snapshots/*/tokenizer'))
    if not hits:
        raise FileNotFoundError(
            'tokenizer not found in the Hugging Face cache; '
            'run the official Irodori-TTS setup once, or pass --tokenizer-dir'
        )
    return hits[0]


class TokenizerBundle:
    """Unified wrapper supporting fast tokenizers-only path and PretrainedTextTokenizer fallback."""

    def __init__(self, normalize_text_fn, encode_text_fn, encode_caption_fn,
                 pad_id: int, bos_id: int, irodori_dir: Path):
        self.normalize_text = normalize_text_fn
        self._encode_text = encode_text_fn
        self._encode_caption = encode_caption_fn
        self.pad_id = int(pad_id)
        self.bos_id = int(bos_id)
        self.irodori_dir = irodori_dir

    def encode_caption(self, caption: str, max_caption_len: int = 512) -> tuple[list[int], int]:
        ids = self._encode_caption(caption, max_caption_len)
        return ids, len(ids)

    def encode_text(self, text: str, max_text_len: int = 256) -> tuple[str, list[int], int]:
        norm_text = self.normalize_text(text).strip()
        ids = self._encode_text(norm_text, max_text_len)
        return norm_text, ids, len(ids)


def load_tokenizers(irodori_dir: str | Path, tokenizer_dir: str | None) -> TokenizerBundle:
    """Load official normalizer and text/caption tokenizers once (torch-free fast path when available)."""
    irodori_path = Path(irodori_dir).expanduser().resolve()
    pkg_dir = irodori_path / "irodori_tts"
    tn_mod = _load_module_direct("irodori_text_normalization", pkg_dir / "text_normalization.py")
    snap = Path(resolve_tokenizer_dir(tokenizer_dir))

    tok_json = snap / "tokenizer.json"
    tok_cfg_json = snap / "tokenizer_config.json"
    if tok_json.is_file() and tok_cfg_json.is_file():
        cfg = json.loads(tok_cfg_json.read_text(encoding="utf-8"))
        bos_tok_name = cfg.get("bos_token")
        pad_tok_name = cfg.get("pad_token") or cfg.get("eos_token")
        if bos_tok_name and pad_tok_name:
            try:
                from tokenizers import Tokenizer
            except ImportError:
                Tokenizer = None
            if Tokenizer is not None:
                ts_mod = _load_module_direct("irodori_token_split", pkg_dir / "token_split.py")
                raw_tok = Tokenizer.from_file(str(tok_json))
                raw_tok.no_padding()
                raw_tok.no_truncation()
                bos_id = raw_tok.token_to_id(bos_tok_name)
                pad_id = raw_tok.token_to_id(pad_tok_name)
                if bos_id is not None and pad_id is not None:
                    splitter = ts_mod.RareTokenSplitter(raw_tok)

                    def _fast_encode_text(norm_text: str, max_len: int) -> list[int]:
                        if max_len <= 1:
                            return [bos_id]
                        return [bos_id] + splitter.encode(norm_text)[: max_len - 1]

                    def _fast_encode_caption(cap: str, max_len: int) -> list[int]:
                        if max_len <= 1:
                            return [bos_id]
                        ids = raw_tok.encode(cap, add_special_tokens=False).ids[: max_len - 1]
                        return [bos_id] + ids

                    return TokenizerBundle(
                        tn_mod.normalize_text,
                        _fast_encode_text,
                        _fast_encode_caption,
                        pad_id=pad_id,
                        bos_id=bos_id,
                        irodori_dir=irodori_path,
                    )

    # Fallback: official PretrainedTextTokenizer via transformers + torch
    if str(irodori_path) not in sys.path:
        sys.path.insert(0, str(irodori_path))
    from irodori_tts.tokenizer import PretrainedTextTokenizer

    text_tok = PretrainedTextTokenizer.from_pretrained(
        repo_id=str(snap), add_bos=True, local_files_only=True)
    text_tok.enable_rare_token_split()          # text: split ON (official)
    cap_tok = PretrainedTextTokenizer.from_pretrained(
        repo_id=str(snap), add_bos=True, local_files_only=True)   # caption: split OFF

    def _pt_encode_text(norm_text: str, max_len: int) -> list[int]:
        ids, mask = text_tok.batch_encode([norm_text], max_length=max_len)
        tl = int(mask.sum())
        return ids[0, :tl].tolist()

    def _pt_encode_caption(cap: str, max_len: int) -> list[int]:
        ids, mask = cap_tok.batch_encode([cap], max_length=max_len)
        cl = int(mask.sum())
        return ids[0, :cl].tolist()

    return TokenizerBundle(
        tn_mod.normalize_text,
        _pt_encode_text,
        _pt_encode_caption,
        pad_id=text_tok.pad_token_id,
        bos_id=text_tok.bos_token_id,
        irodori_dir=irodori_path,
    )


def tokenize_segment(
    bundle: TokenizerBundle,
    text: str,
    caption: str,
    max_text_len: int = 256,
    max_caption_len: int = 512,
    precomputed_caption: tuple[list[int], int] | None = None,
    include_duration_features: bool = True,
) -> dict:
    norm_text, text_id_list, tl = bundle.encode_text(text, max_text_len=max_text_len)
    if precomputed_caption is None:
        cap_id_list, cl = bundle.encode_caption(caption, max_caption_len=max_caption_len)
    else:
        cap_id_list, cl = precomputed_caption

    out = {
        "normalized_text": norm_text,
        "text_ids": text_id_list, "text_len": tl,
        "caption_ids": cap_id_list, "caption_len": cl,
        "pad_id": bundle.pad_id, "bos_id": bundle.bos_id,
    }
    if include_duration_features:
        try:
            import torch
        except ImportError:
            print("note: skipped duration_features (torch is not installed)", file=sys.stderr)
        else:
            if str(bundle.irodori_dir) not in sys.path:
                sys.path.insert(0, str(bundle.irodori_dir))
            from irodori_tts.duration import build_duration_features

            feats = build_duration_features(
                [norm_text],
                token_counts=torch.tensor([tl], dtype=torch.long),
                max_text_len=max_text_len,
                has_speaker=torch.zeros((1,), dtype=torch.bool),
            )
            out["duration_features"] = feats[0].tolist()
    return out


def main() -> None:
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
    ap.add_argument('--no-duration-features', action='store_true',
                    help='skip duration_features (unused by infer_onnx.py) to avoid importing torch')
    args = ap.parse_args()

    if not args.irodori_dir:
        ap.error('--irodori-dir (or IRODORI_TTS_DIR) is required')

    try:
        bundle = load_tokenizers(args.irodori_dir, args.tokenizer_dir)
    except FileNotFoundError as exc:
        ap.error(str(exc))

    out = tokenize_segment(
        bundle, args.text, args.caption,
        max_text_len=args.max_text_len, max_caption_len=args.max_caption_len,
        include_duration_features=not args.no_duration_features,
    )
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("out", args.out, "text_len", out["text_len"], "caption_len", out["caption_len"])
    print("text_ids", out["text_ids"])
    print("caption_ids", out["caption_ids"])


if __name__ == '__main__':
    main()
