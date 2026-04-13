# HW2 Audio Pipeline

This script turns input text into speech with Gemini, saves audio files for two voices, transcribes one audio file back to text, compares the transcript to the original input, and logs latency and estimated cost for each API call.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Set up environment variables

1. Copy `.env.example` to `.env`
2. Add your Gemini API key
3. Keep the default models unless your instructor used different Gemini model IDs

Example `.env`:

```env
GEMINI_API_KEY=your_real_key_here
GEMINI_TTS_MODEL=gemini-2.5-flash-preview-tts
GEMINI_STT_MODEL=gemini-2.5-flash
```

## Run the pipeline

```bash
python hw2-audio-pipeline.py
```

Optional:

```bash
python hw2-audio-pipeline.py --text "Audio pipelines are useful for accessibility and transcription tasks."
python hw2-audio-pipeline.py --transcribe-file audio-output/voice_nova_sample.mp3
```

## Expected output

```text
=== HW2 Audio Pipeline ===

[1/4] Generating speech with voice: Kore
  Text: "Machine learning models learn patterns from data..."
  Generated in 2.14s
  File: audio-output/voice_kore_sample.wav (47.3 KB)
  Audio format: audio/wav
  Cost: $0.0021

[2/4] Generating speech with voice: Charon
  Text: "Machine learning models learn patterns from data..."
  Generated in 1.98s
  File: audio-output/voice_charon_sample.wav (45.8 KB)
  Audio format: audio/wav
  Cost: $0.0021

[3/4] Transcribing audio-output/voice_kore_sample.wav
  Transcript: "Machine learning models learn patterns from data..."
  Transcribed in 1.52s
  Audio duration: 8.30s
  Cost: $0.0008

[4/4] Comparing original vs transcribed text
  Original:    "Machine learning models learn patterns from data..."
  Transcribed: "Machine learning models learn patterns from data..."
  Word overlap accuracy: 100.0%

=== Cost and Latency Summary ===
  TTS calls: 2 | Total cost: $0.0042 | Avg latency: 2.06s
  STT calls: 1 | Total cost: $0.0008 | Avg latency: 1.52s
  Pipeline total: $0.0050

=== Pipeline complete ===
Saved run summary to: last-run-summary.json
```

## Notes

- `api-call-log.csv` is created automatically and stores timestamp, model, latency, input size, and estimated cost for each API call.
- `last-run-summary.json` is created automatically and is useful when writing the reflection with your own real run numbers.
- Generated files are saved inside `audio-output/`.
- Gemini may return either WAV-compatible audio bytes or another audio MIME type, so the saved extension is chosen automatically from the returned response format.
