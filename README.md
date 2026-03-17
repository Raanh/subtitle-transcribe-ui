# Subtitle Transcribe UI

Manual, queue-based OpenAI subtitle transcription UI.

## Key behavior

- No automatic whole-library processing
- Manual file/folder selection only
- Cost estimate before queueing
- Queue statuses + history
- Raw EN stage: `.openai.raw.en.srt` (optional keep for debugging)
- Final EN stage: `.openai.en.srt` (primary subtitle for Plex)
- ffsubsync runs from raw and writes final synced EN without leaving `.synced.*` artifacts
- Manual `Translate to HR` from generated EN subtitles (`.openai.hr.srt` / safe `.openai.alt.hr.srt`)

## Stack files

- `docker-compose.yml`
- `.env.example`
- `app/main.py`
- `app/requirements.txt`
