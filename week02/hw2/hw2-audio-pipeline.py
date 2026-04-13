from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import wave
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from mutagen.mp3 import MP3
except ImportError:  # pragma: no cover
    MP3 = None

try:
    import google.genai as genai
    from google.genai import errors, types
except ImportError:
    print("ERROR: google-genai package not installed.")
    print("Run: pip install google-genai")
    raise SystemExit(1)


ROOT_DIR = Path(__file__).resolve().parent
AUDIO_OUTPUT_DIR = ROOT_DIR / "audio-output"
LOG_FILE = ROOT_DIR / "api-call-log.csv"
RUN_SUMMARY_FILE = ROOT_DIR / "last-run-summary.json"

DEFAULT_TEXT = (
    "Machine learning models learn patterns from data, but they still need careful "
    "evaluation because small transcription changes can affect meaning."
)
DEFAULT_VOICES = ["Kore", "Charon"]
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav"}

TTS_COST_PER_1000_CHARS = 0.015
STT_COST_PER_MINUTE = 0.006


@dataclass
class CallRecord:
    call_type: str
    timestamp: str
    model: str
    latency_seconds: float
    input_size: str
    estimated_cost: float
    extra: dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_print(message: str = "") -> None:
    print(message)


def load_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        safe_print("ERROR: GEMINI_API_KEY is missing.")
        safe_print("Create a .env file from .env.example and add your Gemini API key.")
        raise SystemExit(1)

    return genai.Client(api_key=api_key)


def ensure_output_dir() -> None:
    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def estimate_tts_cost(text: str) -> float:
    return (len(text) / 1000) * TTS_COST_PER_1000_CHARS


def estimate_stt_cost(duration_seconds: float) -> float:
    return (duration_seconds / 60) * STT_COST_PER_MINUTE


def file_size_kb(file_path: Path) -> float:
    return file_path.stat().st_size / 1024


def get_audio_duration_seconds(file_path: Path) -> float:
    extension = file_path.suffix.lower()

    if extension == ".wav":
        with wave.open(str(file_path), "rb") as wav_file:
            frames = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            return frames / float(frame_rate)

    if extension == ".mp3":
        if MP3 is None:
            raise RuntimeError(
                "mutagen is required to read MP3 duration. Install dependencies from requirements.txt."
            )
        return float(MP3(file_path).info.length)

    raise ValueError(
        f"Unsupported audio format '{extension}'. Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
    )


def append_log(record: CallRecord) -> None:
    needs_header = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if needs_header:
            writer.writerow(
                [
                    "timestamp",
                    "call_type",
                    "model",
                    "latency_seconds",
                    "input_size",
                    "estimated_cost_usd",
                    "extra_json",
                ]
            )
        writer.writerow(
            [
                record.timestamp,
                record.call_type,
                record.model,
                f"{record.latency_seconds:.2f}",
                record.input_size,
                f"{record.estimated_cost:.6f}",
                json.dumps(record.extra, ensure_ascii=True),
            ]
        )


def retryable_call(func, retries: int = 1):
    attempts = retries + 1
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except (errors.ClientError, errors.ServerError, errors.APIError) as error:
            last_error = error
            if attempt >= attempts:
                break
            safe_print(f"  Temporary Gemini API error: {error.__class__.__name__}. Retrying once...")
            time.sleep(1)
    raise last_error


def normalize_words(text: str) -> list[str]:
    return re.findall(r"\b[\w']+\b", text.lower())


def word_overlap_accuracy(original_text: str, transcribed_text: str) -> float:
    original_words = normalize_words(original_text)
    transcribed_words = normalize_words(transcribed_text)

    if not original_words:
        return 0.0

    original_counter = Counter(original_words)
    transcribed_counter = Counter(transcribed_words)
    shared_count = sum((original_counter & transcribed_counter).values())
    return (shared_count / len(original_words)) * 100


def write_run_summary(summary: dict[str, Any]) -> None:
    with RUN_SUMMARY_FILE.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)


def mime_type_to_extension(mime_type: str) -> str:
    normalized = mime_type.lower()
    if "mpeg" in normalized or "mp3" in normalized:
        return ".mp3"
    return ".wav"


def parse_pcm_sample_rate(mime_type: str) -> int:
    match = re.search(r"rate=(\d+)", mime_type, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 24000


def write_audio_file(audio_bytes: bytes, mime_type: str, output_stub: Path) -> Path:
    extension = mime_type_to_extension(mime_type)
    output_path = output_stub.with_suffix(extension)

    normalized = mime_type.lower()
    if "l16" in normalized or "pcm" in normalized:
        sample_rate = parse_pcm_sample_rate(mime_type)
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
        return output_path

    output_path.write_bytes(audio_bytes)
    return output_path


def extract_audio_bytes(response: types.GenerateContentResponse) -> tuple[bytes, str]:
    for candidate in response.candidates or []:
        content = candidate.content
        if content is None:
            continue
        for part in content.parts or []:
            if part.inline_data and part.inline_data.data:
                mime_type = part.inline_data.mime_type or "audio/wav"
                return part.inline_data.data, mime_type

    raise ValueError("Gemini returned no audio data.")


def detect_audio_mime_type(audio_path: Path) -> str:
    extension = audio_path.suffix.lower()
    if extension == ".mp3":
        return "audio/mpeg"
    if extension == ".wav":
        return "audio/wav"
    raise ValueError(
        f"Unsupported audio format '{extension}'. Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
    )


def usage_details(response: types.GenerateContentResponse) -> dict[str, int] | None:
    usage = response.usage_metadata
    if usage is None:
        return None
    return {
        "prompt_token_count": usage.prompt_token_count or 0,
        "candidates_token_count": usage.candidates_token_count or 0,
        "total_token_count": usage.total_token_count or 0,
    }


def generate_speech(
    client: genai.Client,
    text: str,
    voice: str,
    model: str,
    output_stub: Path,
) -> dict[str, Any]:
    ensure_output_dir()

    start = time.perf_counter()

    def call_tts():
        return client.models.generate_content(
            model=model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                    )
                ),
            ),
        )

    response = retryable_call(call_tts, retries=1)
    latency = time.perf_counter() - start
    audio_bytes, mime_type = extract_audio_bytes(response)
    output_path = write_audio_file(audio_bytes, mime_type, output_stub)
    cost = estimate_tts_cost(text)
    size_kb = file_size_kb(output_path)

    record = CallRecord(
        call_type="tts",
        timestamp=utc_now_iso(),
        model=model,
        latency_seconds=latency,
        input_size=f"{len(text)} chars",
        estimated_cost=cost,
        extra={
            "voice": voice,
            "mime_type": mime_type,
            "output_file": output_path.name,
            "file_size_kb": round(size_kb, 2),
            "usage": usage_details(response),
        },
    )
    append_log(record)

    return {
        "voice": voice,
        "model": model,
        "latency_seconds": latency,
        "cost": cost,
        "file_path": str(output_path),
        "file_size_kb": size_kb,
        "mime_type": mime_type,
        "usage": usage_details(response),
    }


def extract_transcript_text(response: types.GenerateContentResponse) -> str:
    if response.text:
        return response.text.strip()

    pieces: list[str] = []
    for candidate in response.candidates or []:
        content = candidate.content
        if content is None:
            continue
        for part in content.parts or []:
            if part.text:
                pieces.append(part.text.strip())

    return "\n".join(piece for piece in pieces if piece).strip()


def transcribe_audio(client: genai.Client, audio_path: Path, model: str) -> dict[str, Any]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    extension = audio_path.suffix.lower()
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{extension}'. Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}"
        )

    duration_seconds = get_audio_duration_seconds(audio_path)
    audio_bytes = audio_path.read_bytes()
    mime_type = detect_audio_mime_type(audio_path)

    start = time.perf_counter()

    def call_stt():
        return client.models.generate_content(
            model=model,
            contents=[
                "Transcribe this audio exactly. Return only the transcript text with no extra commentary.",
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ],
        )

    response = retryable_call(call_stt, retries=1)
    latency = time.perf_counter() - start
    cost = estimate_stt_cost(duration_seconds)
    transcript_text = extract_transcript_text(response)

    record = CallRecord(
        call_type="stt",
        timestamp=utc_now_iso(),
        model=model,
        latency_seconds=latency,
        input_size=f"{duration_seconds:.2f} sec",
        estimated_cost=cost,
        extra={
            "audio_file": audio_path.name,
            "duration_seconds": round(duration_seconds, 2),
            "mime_type": mime_type,
            "usage": usage_details(response),
        },
    )
    append_log(record)

    return {
        "model": model,
        "latency_seconds": latency,
        "cost": cost,
        "duration_seconds": duration_seconds,
        "transcript_text": transcript_text,
        "audio_path": str(audio_path),
        "usage": usage_details(response),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW2 audio TTS/STT round-trip pipeline.")
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text to convert to speech and compare against transcription.",
    )
    parser.add_argument(
        "--voices",
        nargs="+",
        default=DEFAULT_VOICES,
        help="Voices to use for TTS generation. At least two are recommended.",
    )
    parser.add_argument(
        "--transcribe-file",
        default=None,
        help="Optional existing audio file to transcribe instead of the first generated sample.",
    )
    return parser.parse_args()


def print_summary(tts_runs: list[dict[str, Any]], stt_runs: list[dict[str, Any]]) -> dict[str, Any]:
    tts_total = sum(item["cost"] for item in tts_runs)
    stt_total = sum(item["cost"] for item in stt_runs)
    tts_avg_latency = sum(item["latency_seconds"] for item in tts_runs) / len(tts_runs) if tts_runs else 0.0
    stt_avg_latency = sum(item["latency_seconds"] for item in stt_runs) / len(stt_runs) if stt_runs else 0.0
    pipeline_total = tts_total + stt_total

    safe_print("\n=== Cost and Latency Summary ===")
    safe_print(
        f"  TTS calls: {len(tts_runs)} | Total cost: ${tts_total:.4f} | Avg latency: {tts_avg_latency:.2f}s"
    )
    safe_print(
        f"  STT calls: {len(stt_runs)} | Total cost: ${stt_total:.4f} | Avg latency: {stt_avg_latency:.2f}s"
    )
    safe_print(f"  Pipeline total: ${pipeline_total:.4f}")

    return {
        "tts_calls": len(tts_runs),
        "stt_calls": len(stt_runs),
        "tts_total_cost": round(tts_total, 6),
        "stt_total_cost": round(stt_total, 6),
        "pipeline_total_cost": round(pipeline_total, 6),
        "tts_avg_latency_seconds": round(tts_avg_latency, 4),
        "stt_avg_latency_seconds": round(stt_avg_latency, 4),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    client = load_client()

    tts_model = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
    stt_model = os.getenv("GEMINI_STT_MODEL", "gemini-2.5-flash")

    safe_print("=== HW2 Audio Pipeline ===\n")

    try:
        tts_runs = []
        total_steps = len(args.voices) + 2
        for index, voice in enumerate(args.voices, start=1):
            output_stub = AUDIO_OUTPUT_DIR / f"voice_{voice.lower()}_sample"
            safe_print(f"[{index}/{total_steps}] Generating speech with voice: {voice}")
            safe_print(f'  Text: "{args.text}"')

            result = generate_speech(
                client=client,
                text=args.text,
                voice=voice,
                model=tts_model,
                output_stub=output_stub,
            )
            tts_runs.append(result)

            safe_print(f"  Generated in {result['latency_seconds']:.2f}s")
            safe_print(f"  File: {result['file_path']} ({result['file_size_kb']:.1f} KB)")
            safe_print(f"  Audio format: {result['mime_type']}")
            safe_print(f"  Cost: ${result['cost']:.4f}\n")

        transcription_target = Path(args.transcribe_file) if args.transcribe_file else Path(tts_runs[0]["file_path"])
        safe_print(f"[{len(args.voices) + 1}/{total_steps}] Transcribing {transcription_target}")
        stt_result = transcribe_audio(client, transcription_target, stt_model)
        safe_print(f'  Transcript: "{stt_result["transcript_text"]}"')
        safe_print(f"  Transcribed in {stt_result['latency_seconds']:.2f}s")
        safe_print(f"  Audio duration: {stt_result['duration_seconds']:.2f}s")
        safe_print(f"  Cost: ${stt_result['cost']:.4f}\n")

        accuracy = word_overlap_accuracy(args.text, stt_result["transcript_text"])
        safe_print(f"[{total_steps}/{total_steps}] Comparing original vs transcribed text")
        safe_print(f'  Original:    "{args.text}"')
        safe_print(f'  Transcribed: "{stt_result["transcript_text"]}"')
        safe_print(f"  Word overlap accuracy: {accuracy:.1f}%")

        totals = print_summary(tts_runs, [stt_result])
        write_run_summary(
            {
                "executed_at_utc": utc_now_iso(),
                "input_text": args.text,
                "voices": args.voices,
                "tts_runs": tts_runs,
                "stt_run": stt_result,
                "accuracy_percent": round(accuracy, 2),
                "summary": totals,
            }
        )
        safe_print("\n=== Pipeline complete ===")
        safe_print(f"Saved run summary to: {RUN_SUMMARY_FILE}")

    except FileNotFoundError as error:
        safe_print(f"ERROR: {error}")
        raise SystemExit(1)
    except ValueError as error:
        safe_print(f"ERROR: {error}")
        raise SystemExit(1)
    except (errors.ClientError, errors.ServerError, errors.APIError) as error:
        safe_print(f"ERROR: Gemini API request failed after retry: {error}")
        raise SystemExit(1)
    except Exception as error:
        safe_print(f"ERROR: Pipeline failed: {error}")
        raise SystemExit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
