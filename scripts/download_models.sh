#!/bin/bash
# Download the ONNX graphs into ./models/{int8,f32}.
# Sources (weights are NOT redistributed by this repository; always fetch
# them from the original publishers):
#   int8 + shared float graphs: Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8
#   float graphs + reference encoder: raratu/Onsei-iOS-Models (v4.1-small-mf/)
set -u
# Pitfall: if NO_PROXY contains IPv6 notation, httpx/urllib stacks can fail
# with "Invalid port" errors behind some proxies. Pin it to plain hosts.
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
cd "$(dirname "$0")/.."
dl() { # url dest
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [ -s "$dest" ]; then echo "skip existing $dest"; return 0; fi
  for i in 1 2 3; do
    curl -L -C - --retry 3 --retry-delay 5 -o "$dest" "$url" && return 0
    echo "retry $i for $dest"; sleep 3
  done
  return 1
}
B=https://huggingface.co/Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8/resolve/main
dl "$B/text_encoder_int8.onnx" models/int8/text_encoder_int8.onnx
dl "$B/dit_step_int8.onnx" models/int8/dit_step_int8.onnx
dl "$B/duration.onnx" models/int8/duration.onnx
dl "$B/context_kv.onnx" models/int8/context_kv.onnx
dl "$B/speaker_encoder.onnx" models/int8/speaker_encoder.onnx
dl "$B/dacvae_decode.onnx" models/int8/dacvae_decode.onnx
dl "$B/config.json" models/int8/config.json
dl "$B/manifest.json" models/int8/manifest.json
dl "$B/tokenizer/tokenizer.json" models/int8/tokenizer/tokenizer.json
dl "$B/tokenizer/tokenizer_config.json" models/int8/tokenizer/tokenizer_config.json
F=https://huggingface.co/raratu/Onsei-iOS-Models/resolve/main/v4.1-small-mf
dl "$F/text_encoder.onnx" models/f32/text_encoder.onnx
dl "$F/dit_step.onnx" models/f32/dit_step.onnx
dl "$F/dacvae_encode.onnx" models/f32/dacvae_encode.onnx
dl "$F/config.json" models/f32/config.json
dl "$F/tokenizer/tokenizer.json" models/f32/tokenizer/tokenizer.json
dl "$F/tokenizer/tokenizer_config.json" models/f32/tokenizer/tokenizer_config.json
echo DONE
ls -lh models/int8 models/f32
