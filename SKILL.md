---
name: muse-irodori-voice
description: Build a custom voice for your Muse agent on the Muse VM using Irodori-TTS v4.1-Small-MF with ONNX int8 inference — design a voice by caption, fix it as a cached reference latent, and read long texts by split-generate-and-join. Use when the user wants their agent to speak with its own voice (voice notes), or to reproduce/tune this voice pipeline.
---

# Muse Irodori Voice — 自分のエージェントの声を作る

Muse の VM (CPU のみ) で Irodori-TTS v4.1-Small-MF を ONNX Runtime 推論し、
エージェント固有の声を (1) キャプションで設計し、(2) 参照潜在として固定し、
(3) 長文は分割生成+結合で読む、までの再現手順。

前提の事実：

- アプリのライブ音声会話の声は差し替えできない。できるのは**音声ファイル
  (ボイスメモ) の生成**である。本スキルはそのためのもの。
- 生成は1本 40〜90 秒オーダー。即時応答ではなく事前生成向き。

## 前提環境

- Linux x86_64、2 vCPU 程度、RAM 7.7GB 程度で実測済み (AVX-512 VNNI があると速い)
- Python 3.12、onnxruntime 1.31.1
- 公式 Irodori-TTS リポジトリのチェックアウト (トークナイズに必須)：
  https://github.com/Aratako/Irodori-TTS

## 手順 1: モデルとコードの準備

1. この repo を配置する。
2. モデルを取得する：`bash scripts/download_models.sh`
   - int8 一式 ← https://huggingface.co/Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8
   - float 一式 + `dacvae_encode.onnx` ← https://huggingface.co/raratu/Onsei-iOS-Models の `v4.1-small-mf/`
   - 配置は `models/int8/`, `models/f32/` (README の「モデルの入手先」参照)。
   - **罠**：ダウンロード前に `NO_PROXY=localhost,127.0.0.1` を設定する。
     IPv6 表記が含まれていると httpx が `Invalid port: ':1]'` で落ちる。
3. 推論用 venv を作る：
   `uv venv .venv && uv pip install --python .venv/bin/python onnxruntime==1.31.1 numpy soundfile pyloudnorm scipy tokenizers`
4. 公式 repo をクローンする (`git clone https://github.com/Aratako/Irodori-TTS.git`)。
   - 通常のトークナイズ＋ONNX 推論は上記 `.venv` のみで動作する (`tokenizers` 使用の `torch` レス高速経路)。
   - 手順 2 の PyTorch 版との比較検証を行う場合や `tokenize_text.py` 単体で `duration_features` を出力する場合は、公式 repo 側でも CPU 環境と HF キャッシュを用意する：
     `uv sync --extra cpu` (所要 約4分、キャッシュ約3.4GB)。

## 手順 2: まず float で移植を検証する (省略禁止)

int8 をいきなり信用しない。順番を守ること：

1. `--variant f32` で1文生成し、公式 PyTorch 版と同条件で比較する。
2. 可能なら公式ランタイムの中間テンソル (text_state、尺予測 log_frames 等) と
   照合する。目安：text_state 最大差 1e-5 オーダー、尺予測は完全一致。
3. 一致を確認して初めて `--variant int8` に切り替える。
   int8 の劣化は「シードを変えた程度」以下が正常域 (log-mel 等で確認可能)。

## 手順 3: トークナイズ (罠あり)

```bash
.venv/bin/python scripts/tokenize_text.py \
  --irodori-dir <irodori-tts> --tokenizer-dir models/int8/tokenizer \
  --text "<本文>" --caption "<キャプション>" \
  --out tokens.json
```

- **本文はレアトークン分割 ON、キャプションは OFF**。公式ランタイム準拠。
  キャプションに分割を適用すると尺予測がずれる (実測 189→178 frames)。
  `scripts/tokenize_text.py` はこの設定を再現済みなので、自作しないこと。
- `--tokenizer-dir models/int8/tokenizer` 等に `tokenizer.json` がある場合は、`tokenizers` のみを使う `torch` レス高速経路が自動選択される (`PretrainedTextTokenizer` と完全一致)。`duration_features` を含む単体 CLI 出力やフォールバック時は `<irodori-tts>/.venv/bin/python` を使用する。
- 単発推論は生成1回ごとに本文 (とキャプション) の tokens.json が要る（長文の一括生成は手順 6 の `speak_long_text.py` を推奨）。

## 手順 4: 声を選ぶ (混合文で選定する)

声はキャプションでは決まらない。**引き当てた声を参照で固定する**のが本筋で、
この手順は「良い引きを効率よく引く」ための選定フローである。

1. キャプションを書いて**固定する** (選定中は変えない)。要点：
   - 数値 (Hz・オクターブ) は書いても効かない。相対表現で書く。
   - 先頭に性別と声の高さ。属性は 2〜4 個。ブレスは語彙で明示する。
   - 例：「若い女性の声。高さは中くらいからやや高めで、抑揚が豊か。
     息づかいが感じられる近い距離感で、明るく親しみやすく話す。」
2. 選定文には**短い混合文**を使う：悲しい内容と楽しい内容を
   1つの文群に両方含める (目安 60 字前後)。勝ちクリップ自体が
   両方の感情の演技を含むので、後の固定で感情リークを防げる (手順5)。
   - 例：「ニュースをまとめる際に悲しい記事を読むたび、心が痛くなります。
     でも皆さんのお役に立つことが、私の喜びなんです。 ウフフ」
   - 長い選定文 (約80字・15秒超) は部分的にノイズが乗りやすいので短めが良い。
3. キャプション固定・参照なしで、**大きく離したランダムシードで5案**生成し、
   ユーザーの耳で選んでもらう：
   `.venv/bin/python scripts/infer_onnx.py --variant int8 --tokens tokens.json --seed <seed> --out take.wav`
   - 気に入ったものがなければ、別のシードで**もう5案**出す。
   - **声の同一性は試行ごとにサンプリングし直される**ことが前提。
     シード固定の再現性は同一テキスト・同一条件の範囲だけ。
4. 選ばれた際に「その声のどこが良いか」をユーザーに聞き、
   **声の特徴メモ**として残す (例：抑揚が大きい・高くて若々しい・元気)。
   ただし特徴をキャプションに足しても声は寄らない (キャプションの
   収束力は弱い)。メモは記録用と割り切ること。
5. 任意：`--truncation-factor 0.85` でノイズ感が大きく減る (推奨値)。
   0.7 まで絞ると感情の強い文が淡々になるので注意。

**声優・キャラクター名での指定について** (選定の前提ルール)：

- 本スキルの趣旨は、シードのランダム性を活かして好みの声を
  「ガチャ感覚で選ぶ」ことであり、実在の人物やキャラクターの
  声の模倣ではない。
- ユーザーが特定の声優名やキャラクター名を挙げて「○○の声に
  寄せてほしい」と求めた場合は、その固有名詞による指定は
  受け付けない (模倣は本プロジェクトの趣旨外であることを
  簡潔に伝える)。
- 代わりに、その声のどんなところが好きなのかを、固有名詞を
  使わずに雰囲気や特徴の言葉で表してもらうよう丁寧に要請する。
  例を挙げて聞くとよい：高さ、若々しさ、抑揚の大きさ、
  落ち着き、元気さ、息づかい、など。
- 得られた特徴表現だけを使ってキャプション設計と選定に進む。
  固有名詞そのものをキャプションやプロンプトに入力しない。

## 手順 5: 声を固定する (参照潜在)

1. 手順4の**勝ちクリップ1本を単一参照**にする。混合文が両方の感情を
   含んでいるため、別の感情の文で第2ラウンドを回す必要はない。
   (ポジティブなクリップだけを参照にすると声は揃うが、悲しい文が
   楽しげに引っ張られる「感情リーク」が起きる。)
2. 潜在を一度だけ作る：
   `.venv/bin/python scripts/make_ref_latents.py --clips winner.wav --out ref/my-voice.npz`
   (内部でクリップを8回エンコード平均し、サンプリング式エンコーダの
   事後平均に近似する。これが正しい手順で、1回エンコードで済ませないこと。)
3. 以後の生成は潜在を読むだけ：
   `.venv/bin/python scripts/infer_onnx.py --variant int8 --tokens tokens.json --ref-latents ref/my-voice.npz --out out.wav`
   生成は1本 35〜52 秒に戻る (毎回 --ref-wavs でエンコードすると約260秒)。
4. 検証として、楽しい文と悲しい文を1本ずつ生成し、声の近さと
   感情の切り替わりを確認する。
5. npz は「そのエージェントの声そのもの」。公開 repo には含めない。

## 手順 6: 長文の読み上げ

1. 文章を文の区切りで **1区間 70〜80 字** に分割する。
   (145 字一発では語の脱落/挿入が実測で発生。区間は ±5 字程度を目安に。)
2. **一括生成（推奨）**：`scripts/speak_long_text.py` を使うと、単一プロセス内で自動分割・`torch` レス高速トークナイズ・ONNX セッションと共通条件 (キャプション・参照潜在) の初回1回読込＆再利用・0.4 秒ギャップ結合まで一括実行できる：
   ```bash
   .venv/bin/python scripts/speak_long_text.py \
     --irodori-dir <irodori-tts> \
     --caption "<キャプション>" \
     --ref-latents ref/my-voice.npz \
     --truncation-factor 0.85 \
     --text "<長文テキスト>" \
     --out full.wav
   ```
   (`--threads` の既定値は `min(6, max(2, cpu_count // 2))`、2 vCPU 環境では `2`。)
3. **個別に回す場合**：区間ごとに tokens.json を作り、同じ参照潜在・同じシードで生成してから結合する：
   `.venv/bin/python scripts/join_wavs.py part1.wav part2.wav --gap 0.4 --out full.wav`

## パラメータ決定値 (検証済み)

| 項目 | 値 |
|---|---|
| variant | int8 |
| MeanFlow ステップ数 | 4 (モデル既定) |
| truncation_factor | 0.85 |
| 参照正規化 | -16 LUFS (公式準拠) |
| 参照エンコード平均 | 8 回 (固定潜在の作成時) / 4 回 (--ref-wavs 対話時) |
| 参照総量 | 混合文の勝ちクリップ1本 (約10〜12 秒) |
| 分割長 | 1区間 70〜80 字、結合間隔 0.4 秒 |
| CFG スケール | MF では無効 (蒸留に織り込み済み)。調整対象にしない |

## トラブルシューティング (罠リスト)

| 症状 | 原因と対処 |
|---|---|
| ダウンロードが `Invalid port` で落ちる | NO_PROXY の IPv6 表記。`NO_PROXY=localhost,127.0.0.1` に絞る |
| 尺 (長さ) が公式と合わない | キャプションにレアトークン分割を適用している。OFF にする (手順3) |
| 参照の声が実行ごとに微妙に違う | dacvae_encode がサンプリング式。複数回平均し、潜在を npz で固定する (手順5) |
| ネガティブな文が楽しげになる | 感情リーク。参照クリップ自体に両方の感情を含む混合文で選定し直す (手順4・5) |
| 長文で語が消える/増える | 一発生成の限界。70〜80 字に分割して結合 (手順6) |
| 声が試行ごとに変わる | 仕様 (同一性の再サンプリング)。シード食べ比べ→参照固定の順で対処 |
| 出力ピークが 1.0 に張り付く | 特に参照ありで発生。必要なら書き出し側で余裕を持たせる |

## 注意

- モデル重みは再配布しない。必ず配布元から取得する。
- Irodori-TTS のライセンスと倫理的制限 (同意のない他者のなりすまし禁止等) に従う。
- この経路では透かしは入らない。生成音声を公開する場合は AI 音声である旨を明示する。
- 詳細な実験記録は docs/findings.md を参照。
