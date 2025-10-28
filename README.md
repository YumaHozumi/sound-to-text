# sound-to-text

軽量かつ高性能な Whisper 実装である [faster-whisper](https://github.com/SYSTRAN/faster-whisper) を使って、m4a などの音声ファイルを文字起こしするためのコマンドラインツールです。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU が利用可能な場合は自動的に検出され、CPU のみの環境でも INT8 量子化により比較的軽量に動作します。

## 使い方

```bash
python transcribe.py input.m4a
```

主なオプション:

- `--model`: 使用する Whisper モデル (`tiny`, `base`, `small`, `medium`, `large-v2` など)。デフォルトは `small` です。
- `--device`: `auto` (デフォルト) で自動選択、`cpu` や `cuda` 固定も可能です。
- `--compute-type`: 量子化モード。`int8_float16` や `int8` は省メモリ、`float16` `float32` は高精度になります。
- `--beam-size`: ビームサーチの幅。大きくすると精度向上が見込めますが遅くなります。
- `--language`: 言語コード (例: `ja`, `en`) を指定すると言語を固定できます。
- `--output`: 結果をファイルに保存します。指定しない場合は標準出力に表示します。

### 例

```bash
python transcribe.py meeting.m4a --model small --output meeting.txt
```

最初の実行時はモデルのダウンロードが行われるため、インターネット接続が必要です。モデルはローカルにキャッシュされ、以後はオフラインで利用できます。