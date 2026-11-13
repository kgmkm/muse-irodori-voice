# muse-irodori-voice

Muse の VM 上で **Irodori-TTS v4.1-Small-MF** の **ONNX int8 推論**を動かし、
「自分のエージェント固有の声」を作るためのキットです。

キャプション (文章) で声質を設計し、気に入った声は参照潜在として固定。
文章ごとに分割生成して結合することで、長文でも安定した読み上げができます。
推論パイプライン・参照潜在の固定スクリプト・検証済みのパラメータ・ハマりどころを
一式まとめています。

> AI エージェント (Muse) がそのまま手順を辿って再現するための手順書は
> [SKILL.md](SKILL.md)、実験の詳細な知見は [docs/findings.md](docs/findings.md) にあります。

## 特徴

- **ONNX Runtime (CPU) のみで動作**：PyTorch 公式実装の推論フローを分割 ONNX グラフで
  再現。公式の中間テンソルと段ごとに照合し、float 経路の一致を検証済み。
- **高速**：公式 PyTorch 版の約 1/3 の生成時間 (下表)。Muse の標準 VM で実測。
- **声の固定**：参照音声を潜在 (npz) として保存し、毎回読み込む方式。
  声の同一性がテキストに依存せず安定し、生成も速くなる。
- **感情リーク対策**：悲しい内容と楽しい内容を両方含む「混合文」で声を選定し、
  その勝ちクリップを単一参照にすることで、楽しげな文も心配な文も
  「同じ声のまま」読める。
- **長文対応**：1区間 70〜80 字に分割生成し、0.4 秒間隔で結合するワークフロー。
  `scripts/speak_long_text.py` を使うと、単一プロセスでトークナイザと ONNX モデルを
  1回だけ読み込み、共通条件を使い回して高速に一括生成できる。

## 実測 (Muse VM: 2 vCPU AVX-512 VNNI / RAM 7.7GB / onnxruntime 1.31.1 / Python 3.12)

短い一文 (約7.5 秒の音声) での比較：

| 構成 | 生成のみ | 合計 | ピークメモリ | 備考 |
|---|---|---|---|---|
| PyTorch 公式 (MF) | 107 秒 | 約4分 | 3.9 GB | 透かし入り |
| GGUF Q6_K (speech.cpp) | 169 秒 | 約3分 | 1.26 GB | 音質不良で不採用 |
| ONNX float32 | 50 秒 | 84 秒 | 4.62 GB | 公式とほぼ一致を検証 |
| **ONNX int8 (本キット)** | **37 秒** | **約59 秒** | **2.67 GB** | — |
| ONNX int8 + 参照潜在キャッシュ | — | 1本 35〜52 秒 | 約2.8 GB | 毎回エンコードだと約260 秒 |

## セットアップ

前提：

- Linux x86_64 (AVX-512 VNNI があると速い)、Python 3.12
- トークナイズ用に公式 [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
  のチェックアウト (`tokenizers` による高速経路を使えば公式 venv 構築不要、float 検証時のみ公式 venv を使用)
- ディスク数 GB (モデル一式 + 公式 repo)

手順：

```bash
# 1. ONNX モデルを取得 (重みは本 repo には含まれません)
bash scripts/download_models.sh     # -> models/int8, models/f32

# 2. 推論用 venv (tokenizers を含めると torch レスで高速トークナイズが可能)
uv venv .venv
uv pip install --python .venv/bin/python onnxruntime==1.31.1 numpy soundfile pyloudnorm scipy tokenizers

# 3. トークナイズ (models/int8/tokenizer を使う場合は .venv のみ・torch レスで動作)
.venv/bin/python scripts/tokenize_text.py \
    --irodori-dir <irodori-tts> \
    --tokenizer-dir models/int8/tokenizer \
    --no-duration-features \
    --text "こんにちは。今日もよろしくお願いします。" \
    --caption "若い女性の声。明るく親しみやすく話す。" \
    --out tokens.json

# 4. 生成 (参照なし)
.venv/bin/python scripts/infer_onnx.py --variant int8 --tokens tokens.json --out out.wav
```

## 使い方

### 声を選ぶ (参照なし)

キャプションで声質を指定します。数値 (Hz 等) は効かないので相対表現で書き、
属性は 2〜4 個に絞ります (詳細は findings.md)。
ただしキャプションは声を「決める」手段ではなく、声の同一性は生成ごとに
サンプリングし直されます。選定文には悲しい内容と楽しい内容を両方含む
**短い混合文** (目安60字前後) を使い、キャプション固定・参照なしで
大きく離したランダムシードの **5案** から選んでもらいます
(気に入らなければ別の5案)。選ばれた声の「どこが良いか」を聞いて
特徴メモとして残すのも有効です (記録用で、キャプションに足しても
声は寄りません)。

### 声を固定する (参照潜在)

勝ちクリップ1本を参照にして、潜在を一度だけ作って保存します：

```bash
.venv/bin/python scripts/make_ref_latents.py \
    --clips winner.wav \
    --out ref/my-voice.npz

.venv/bin/python scripts/infer_onnx.py --variant int8 --tokens tokens.json \
    --ref-latents ref/my-voice.npz --out out.wav
```

混合文の勝ちクリップは両方の感情の演技を含んでいるため、単一参照でも
感情リークが起きにくいのがポイントです (楽しげなクリップだけを参照に
すると、心配な文まで楽しげに引きずられます)。固定後は楽しい文と
悲しい文を1本ずつ生成して、声の近さと感情の切り替わりを検証します。
npz は各自の声そのものなので、**この repo には含めないでください**
(各自が自分の声で作るものです)。

### 主要パラメータの決定値

| パラメータ | 値 | 根拠 |
|---|---|---|
| seed | 任意 (固定推奨) | 再現性は同一テキスト・同一条件の範囲 |
| truncation_factor | 0.85 | ノイズ激減と表現力のバランス (0.7 は感情文が淡々に) |
| 参照の正規化 | -16 LUFS | 公式 `ref_normalize_db` 準拠 |
| 参照エンコード平均回数 | 8 回 (固定用) | サンプリング式エンコーダの事後平均近似 |
| 長文の分割 | 1区間 70〜80 字 | 145 字一発では語の脱落/挿入が発生 |
| 結合の間隔 | 0.4 秒 | `scripts/join_wavs.py --gap 0.4` |

### 長文を読む

```bash
# 単一プロセスで分割・高速トークナイズ・モデル1回読込・条件再利用・結合まで一括実行
.venv/bin/python scripts/speak_long_text.py \
    --irodori-dir <irodori-tts> \
    --caption "若い女性の声。明るく親しみやすく話す。" \
    --ref-latents ref/my-voice.npz \
    --truncation-factor 0.85 \
    --text "長い文章。文ごとに自動分割して結合します。" \
    --out full.wav

# 個別に生成したパートを結合だけ行う場合
.venv/bin/python scripts/join_wavs.py part1.wav part2.wav --gap 0.4 --out full.wav
```

### ユースケース

声は「事前生成してボイスメモとして届ける」形が前提です (生成は1本
40〜90 秒オーダーで、リアルタイム会話には向きません)。その前提で、
次のような使い方ができます。

- **毎朝のニュースをボイスでお届け**：エージェントがまとめた朝のニュース
  ダイジェストを音声化し、スケジュール実行で決まった時刻に届ける。
  朝の支度をしながら「聞く朝刊」として使える。
- **定期配信の読み上げ**：今日の予定、週末のイベント情報など、エージェントが
  集めた定期情報をテキストと一緒に音声でも届ける。目が離せない場面でも
  内容を受け取れる。
- **長い文章の「聞く化」**：記事・メール・会議メモなどの要約を分割生成で
  音声化する。移動中や作業中のインプットに向く。
- **SNS・動画投稿用のナレーション素材**：静止画や動画に、自分のエージェントの
  声でナレーションを当てる。自己紹介動画のような短い尺と相性が良い。
- **おやすみ前の朗読**：短いお話や詩を、落ち着いた雰囲気のキャプションで
  生成する。声が固定されているので、毎回同じ声で読める。
- **語学学習のシャドーイング教材**：同じ文章を、安定した同じ声で何度でも
  生成できる。お手本音声を自分のエージェントの声で用意する使い方。

## 罠と検証方法 (要約)

- キャプションのトークナイズは**レアトークン分割 OFF** (本文のみ ON)。
  間違えると尺予測がずれる。
- `dacvae_encode.onnx` は**サンプリング式**で、同じ入力でも潜在が毎回変わる。
  複数回平均で公式の事後平均に近似する。
- 環境変数 `NO_PROXY` に IPv6 表記があるとダウンロードが落ちることがある。
- 移植の検証は「公式 PyTorch の中間テンソルと段ごとに照合 → float 一致を確認 →
  int8 に載せ替え」の順で行う。

詳細は [docs/findings.md](docs/findings.md) を参照してください。

## モデルの入手先

重みは再配布しません。各配布元から取得してください (download_models.sh が自動化)：

- int8 + 共有 float グラフ：[Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8](https://huggingface.co/Ryoppi/Irodori-TTS-v4.1-Small-MF-ONNX-int8) (Hugging Face)
- float グラフ + 参照エンコーダ：[raratu/Onsei-iOS-Models](https://huggingface.co/raratu/Onsei-iOS-Models) の `v4.1-small-mf/` フォルダ
- 公式実装・元モデル：[Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) / `Aratako/Irodori-TTS-v4.1-Small-MF`

配置：

```
models/
  int8/  text_encoder_int8.onnx, dit_step_int8.onnx, duration.onnx,
         context_kv.onnx, speaker_encoder.onnx, dacvae_decode.onnx, ...
  f32/   text_encoder.onnx, dit_step.onnx, dacvae_encode.onnx, ...
```

## 利用上のルール: 声優・キャラクターの声について

本キットの声の選定は、Irodori-TTS のシードごとの違い・ランダム性を活かして、
ユーザー好みの声を「ガチャ感覚で選ぶ」ことが趣旨です。
実在の人物やキャラクターの声を模倣するための仕組みではありません。

そのため、次を禁じます：

- 本キット (声の選定フロー) を使い、特定の声優の声や特定のキャラクターの
  声に寄せようとする行為。
- それを意図したプロンプトで AI エージェントを誘導する行為、
  および、そうして作られた音声そのものをアップロード・公開する行為。

声の希望は、固有名詞 (声優名・キャラクター名) ではなく、その声の雰囲気や
特徴を表す言葉で伝えてください (例: 高めで若々しい、抑揚が大きい、
落ち着いた、息づかいが感じられる、など)。

## ライセンス / クレジット

- 本 repo の**コード**は MIT License (LICENSE 参照)。
- モデル重みのライセンスは各配布元に従います。Irodori-TTS 本体は MIT ですが
  倫理的制限 (同意のない他者のなりすまし等の禁止) が付随します。必ず確認してください。
- この ONNX 経路では透かし (SilentCipher) は入りません。生成音声を公開する場合は
  AI 音声である旨の明示を推奨します。
- 謝辞：Irodori-TTS 作者の Aratako 氏、ONNX 変換を公開された Ryoppi 氏・raratu 氏、
  そして本キットの検証環境である Muse に感謝します。
