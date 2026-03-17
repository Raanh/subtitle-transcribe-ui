# Subtitle Transcribe UI

Manual, queue-based OpenAI subtitle transcription UI.

## Key behavior

- No automatic whole-library processing
- Manual file/folder selection only
- Cost estimate before queueing
- Queue statuses + history
- Output `.openai.en.srt` next to media file
- Optional `ffsubsync` stage to create `.openai.synced.en.srt`

## Stack files

- `docker-compose.yml`
- `.env.example`
- `app/main.py`
- `app/requirements.txt`
