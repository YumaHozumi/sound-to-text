# faster-whisper 備忘録

## 概要

faster-whisperは、OpenAIのWhisperモデルをCTranslate2で最適化した音声認識ライブラリ。オリジナルのWhisperと比較して大幅な高速化とメモリ効率の改善を実現している。

## インストール

### pip使用

```bash
pip install faster-whisper
```

### uv使用

```bash
uv add faster-whisper
```

### 依存関係

- Python 3.8以上
- オプション: soundfile (音声ファイル読み込みサポート用)

## 基本的な使い方

### シンプルな文字起こし

```python
from faster_whisper import WhisperModel

# モデルの初期化
model = WhisperModel("small", device="cpu", compute_type="int8")

# 音声ファイルの文字起こし
segments, info = model.transcribe("audio.mp3")

# 結果の取得
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

### 言語指定

```python
# 日本語を明示的に指定
segments, info = model.transcribe(
    "audio.mp3",
    language="ja"
)

# 自動言語検出
segments, info = model.transcribe("audio.mp3")
print(f"検出された言語: {info.language}, 確率: {info.language_probability:.2f}")
```

### ビームサーチのカスタマイズ

```python
# より高精度な文字起こし（遅くなる）
segments, info = model.transcribe(
    "audio.mp3",
    beam_size=10,
    best_of=5,
    patience=1.0
)
```

### VAD（音声区間検出）の使用

```python
# VADフィルタを有効化して無音部分をスキップ
segments, info = model.transcribe(
    "audio.mp3",
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500)
)
```

## モデルサイズ

| モデル | パラメータ数 | 必要VRAM | 相対速度 | 精度 |
|--------|-------------|----------|----------|------|
| tiny   | 39M         | ~1GB     | ~32x     | 低   |
| base   | 74M         | ~1GB     | ~16x     | 中低 |
| small  | 244M        | ~2GB     | ~6x      | 中   |
| medium | 769M        | ~5GB     | ~2x      | 高   |
| large-v2 | 1550M     | ~10GB    | 1x       | 最高 |
| large-v3 | 1550M     | ~10GB    | 1x       | 最高 |

## デバイスとCompute Type

### デバイスの選択

```python
# CPU使用
model = WhisperModel("small", device="cpu")

# GPU使用 (CUDA)
model = WhisperModel("small", device="cuda")

# 自動選択
model = WhisperModel("small", device="auto")
```

### Compute Typeの選択

| Compute Type | 説明 | 対応デバイス |
|--------------|------|--------------|
| int8 | 8bit整数量子化 | CPU, CUDA |
| int8_float16 | 混合精度 | CUDA |
| float16 | 16bit浮動小数点 | CUDA, Apple Silicon (MPS未サポート) |
| float32 | 32bit浮動小数点 | CPU, CUDA |

```python
# CPU使用時のおすすめ設定
model = WhisperModel("small", device="cpu", compute_type="int8")

# CUDA使用時のおすすめ設定
model = WhisperModel("small", device="cuda", compute_type="int8_float16")

# Apple Silicon (M1/M2)では
model = WhisperModel("small", device="cpu", compute_type="int8")
```

## 高度な使い方

### タイムスタンプレベルの調整

```python
# 単語レベルのタイムスタンプ
segments, info = model.transcribe(
    "audio.mp3",
    word_timestamps=True
)

for segment in segments:
    for word in segment.words:
        print(f"[{word.start:.2f}s -> {word.end:.2f}s] {word.word}")
```

### プロンプトによる誘導

```python
# 初期プロンプトを使って文脈を提供
segments, info = model.transcribe(
    "audio.mp3",
    initial_prompt="これは技術的な会議の録音です。"
)
```

### 温度パラメータの調整

```python
# 複数の温度で試行し、最適な結果を選択
segments, info = model.transcribe(
    "audio.mp3",
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)
```

### 条件付き確率の取得

```python
segments, info = model.transcribe(
    "audio.mp3",
    condition_on_previous_text=True,  # 前のテキストを文脈として使用
    compression_ratio_threshold=2.4,   # 圧縮率の閾値
    log_prob_threshold=-1.0,           # 対数確率の閾値
    no_speech_threshold=0.6            # 無音判定の閾値
)
```

## パフォーマンスチューニング

### CPU最適化

```python
# スレッド数の指定
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
    cpu_threads=4,
    num_workers=1
)
```

### バッチ処理

```python
# 複数ファイルを効率的に処理
audio_files = ["audio1.mp3", "audio2.mp3", "audio3.mp3"]

model = WhisperModel("small")

for audio_file in audio_files:
    segments, info = model.transcribe(audio_file)
    # 処理...
```

## サポートされる音声形式

- WAV (.wav)
- MP3 (.mp3)
- M4A (.m4a)
- FLAC (.flac)
- OGG (.ogg)
- その他ffmpegがサポートする形式

## トラブルシューティング

### Compute Typeエラー

```
ValueError: Requested int8_float16 compute type, but the target device or backend do not support efficient int8_float16 computation.
```

**解決策**: `compute_type="int8"` に変更

### メモリ不足

- より小さいモデル（tiny, base）を使用
- `compute_type="int8"` で量子化
- バッチサイズを減らす

### 精度が低い

- より大きなモデル（medium, large）を使用
- `beam_size` を増やす（5→10）
- `language` パラメータで言語を明示的に指定
- `initial_prompt` で文脈を提供

## 参考リンク

- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- [サポートされる言語一覧](https://github.com/openai/whisper#available-models-and-languages)

## 言語コード一覧（抜粋）

| 言語 | コード |
|------|--------|
| 日本語 | ja |
| 英語 | en |
| 中国語 | zh |
| 韓国語 | ko |
| フランス語 | fr |
| ドイツ語 | de |
| スペイン語 | es |
| イタリア語 | it |
