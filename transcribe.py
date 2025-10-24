"""Command line tool for transcribing audio files with faster-whisper.

This script downloads a Whisper model (if not already cached) and runs
transcription on the provided audio file.  It defaults to the ``small`` model
which offers a solid balance between accuracy and resource usage, but you can
override the model size through the command line arguments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from faster_whisper import WhisperModel


def positive_int(value: str) -> int:
    """argparse helper that ensures *value* is a positive integer."""

    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid integer") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")

    return parsed


def transcribe_audio(
    audio_path: Path,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "int8_float16",
    beam_size: int = 5,
    language: str | None = None,
) -> tuple[Iterable[str], float]:
    """Transcribe *audio_path* returning an iterable of text segments and duration."""

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=beam_size,
        language=language,
    )

    texts: list[str] = [segment.text.strip() for segment in segments]
    return texts, info.duration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio",
        type=Path,
        help="Path to the audio file (e.g. .m4a, .mp3, .wav) to transcribe.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="small",
        help=(
            "Whisper model size or path. Try 'base' or 'tiny' for faster inference, "
            "or 'medium'/'large-v2' for higher accuracy."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run the model on (auto, cpu, cuda, etc.).",
    )
    parser.add_argument(
        "--compute-type",
        default="int8_float16",
        help=(
            "Quantization mode. 'int8_float16' or 'int8' reduce memory usage, while "
            "'float16' or 'float32' provide maximum accuracy."
        ),
    )
    parser.add_argument(
        "--beam-size",
        type=positive_int,
        default=5,
        help="Beam size used during decoding (larger is slower but more accurate).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force a specific language (e.g. 'ja' or 'en'). Autodetects when omitted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the transcription text. Printed to stdout otherwise.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.audio.exists():
        parser.error(f"Audio file not found: {args.audio}")

    texts, duration = transcribe_audio(
        audio_path=args.audio,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        language=args.language,
    )

    transcript = "\n".join(filter(None, texts)) or "(no speech detected)"

    if args.output is not None:
        args.output.write_text(transcript, encoding="utf-8")
        print(f"Saved transcription to {args.output} (processed {duration:.1f}s of audio)")
    else:
        print(transcript)


if __name__ == "__main__":
    main()
