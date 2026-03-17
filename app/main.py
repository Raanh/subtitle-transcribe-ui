import logging
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import imageio_ffmpeg
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel, Field

APP_DATA_DIR = Path(os.getenv("APP_DATA_DIR", "/data"))
DB_PATH = APP_DATA_DIR / "subtitle_transcribe_ui.db"
LOG_DIR = APP_DATA_DIR / "logs"
TMP_DIR = APP_DATA_DIR / "tmp"

VIDEO_EXTENSIONS = {
    ".mkv",
    ".mp4",
    ".avi",
    ".m4v",
    ".mov",
    ".ts",
    ".wmv",
    ".flv",
    ".webm",
}

ACTIVE_STATUSES = {
    "preparing",
    "extracting audio",
    "transcribing",
    "writing srt",
    "loading english subtitle",
    "translating",
    "writing hr srt",
}
EN_PATTERN = re.compile(r"(^|[._ \-])(en|eng|english)([._ \-]|$)", re.IGNORECASE)
HR_PATTERN = re.compile(r"(^|[._ \-])(hr|hrv|cro|croatian)([._ \-]|$)", re.IGNORECASE)

DEFAULT_SETTINGS = {
    "openai_api_key": "",
    "openai_model": "whisper-1",
    "translation_model": "gpt-4o-mini",
    "translation_chunk_size": "40",
    "price_per_minute_usd": "0.006",
    "concurrency_limit": "1",
    "overwrite_openai_outputs": "0",
    "enable_ffsubsync": "1",
    "show_tv_root": "1",
    "show_movies_root": "1",
    "show_anime_root": "1",
}

ROOT_ENV_MAP = {
    "tv": "MEDIA_ROOT_TV",
    "movies": "MEDIA_ROOT_MOVIES",
    "anime": "MEDIA_ROOT_ANIME",
}

ROOT_VISIBLE_MAP = {
    "tv": "show_tv_root",
    "movies": "show_movies_root",
    "anime": "show_anime_root",
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_storage() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def init_logging() -> logging.Logger:
    logger = logging.getLogger("subtitle-transcribe-ui")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    handler = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=2_000_000, backupCount=5)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    return logger


def init_db() -> None:
    conn = db_conn()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS queue_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                root_key TEXT NOT NULL,
                rel_path TEXT NOT NULL,
                abs_path TEXT NOT NULL,
                filename TEXT NOT NULL,
                job_type TEXT NOT NULL DEFAULT 'transcribe_en',
                status TEXT NOT NULL,
                current_step TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                elapsed_seconds REAL,
                estimated_minutes REAL,
                estimated_cost_usd REAL,
                eta_seconds REAL,
                source_subtitle_path TEXT,
                hr_subtitle_existed INTEGER NOT NULL DEFAULT 0,
                alternate_output_used INTEGER NOT NULL DEFAULT 0,
                output_subtitle_path TEXT,
                synced_subtitle_path TEXT,
                error_message TEXT
            );

            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_item_id INTEGER,
                root_key TEXT,
                rel_path TEXT,
                abs_path TEXT,
                filename TEXT,
                job_type TEXT,
                status TEXT,
                finished_at TEXT,
                elapsed_seconds REAL,
                estimated_minutes REAL,
                estimated_cost_usd REAL,
                source_subtitle_path TEXT,
                hr_subtitle_existed INTEGER,
                alternate_output_used INTEGER,
                output_subtitle_path TEXT,
                synced_subtitle_path TEXT,
                error_message TEXT
            );
            """
        )
        ensure_column(
            conn,
            "queue_items",
            "job_type",
            "TEXT NOT NULL DEFAULT 'transcribe_en'",
        )
        ensure_column(conn, "queue_items", "source_subtitle_path", "TEXT")
        ensure_column(
            conn,
            "queue_items",
            "hr_subtitle_existed",
            "INTEGER NOT NULL DEFAULT 0",
        )
        ensure_column(
            conn,
            "queue_items",
            "alternate_output_used",
            "INTEGER NOT NULL DEFAULT 0",
        )
        ensure_column(conn, "history", "job_type", "TEXT")
        ensure_column(conn, "history", "source_subtitle_path", "TEXT")
        ensure_column(conn, "history", "hr_subtitle_existed", "INTEGER")
        ensure_column(conn, "history", "alternate_output_used", "INTEGER")
        conn.commit()
    finally:
        conn.close()

    for key, value in DEFAULT_SETTINGS.items():
        set_setting_if_missing(key, value)

    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        current = get_setting("openai_api_key", "")
        if not current:
            save_setting("openai_api_key", env_key)


def ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name in existing:
        return
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")


def set_setting_if_missing(key: str, value: str) -> None:
    conn = db_conn()
    try:
        conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES(?, ?)", (key, value))
        conn.commit()
    finally:
        conn.close()


def save_setting(key: str, value: str) -> None:
    conn = db_conn()
    try:
        conn.execute(
            """
            INSERT INTO settings(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


def get_setting(key: str, default: str = "") -> str:
    conn = db_conn()
    try:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        if row:
            return row["value"]
        return default
    finally:
        conn.close()


def get_settings() -> dict[str, str]:
    conn = db_conn()
    try:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    finally:
        conn.close()
    result = dict(DEFAULT_SETTINGS)
    result.update({row["key"]: row["value"] for row in rows})
    return result


def bool_setting(key: str, default: bool) -> bool:
    val = get_setting(key, "1" if default else "0").strip().lower()
    return val in {"1", "true", "yes", "on"}


def float_setting(key: str, default: float) -> float:
    try:
        return float(get_setting(key, str(default)))
    except Exception:
        return default


def int_setting(key: str, default: int) -> int:
    try:
        return int(get_setting(key, str(default)))
    except Exception:
        return default


def media_roots() -> dict[str, dict[str, Any]]:
    settings = get_settings()
    roots: dict[str, dict[str, Any]] = {}
    for root_key, env_name in ROOT_ENV_MAP.items():
        path = Path(os.getenv(env_name, "")).resolve() if os.getenv(env_name) else None
        roots[root_key] = {
            "path": path,
            "enabled": settings.get(ROOT_VISIBLE_MAP[root_key], "1") == "1",
        }
    return roots


def ensure_root_path(root_key: str) -> Path:
    roots = media_roots()
    info = roots.get(root_key)
    if not info:
        raise HTTPException(status_code=400, detail=f"Unknown root: {root_key}")
    root_path = info["path"]
    if not root_path:
        raise HTTPException(status_code=400, detail=f"Root '{root_key}' is not configured")
    if not root_path.exists():
        raise HTTPException(status_code=400, detail=f"Root path does not exist: {root_path}")
    return root_path


def resolve_video_path(root_key: str, rel_path: str) -> Path:
    root = ensure_root_path(root_key)
    target = (root / rel_path).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="Invalid path traversal")
    return target


def subtitle_presence(video_path: Path) -> tuple[bool, bool, list[str], list[str]]:
    if not video_path.exists():
        return False, False, [], []

    base = video_path.stem.lower()
    en_hits: list[str] = []
    hr_hits: list[str] = []

    for entry in video_path.parent.iterdir():
        if not entry.is_file() or entry.suffix.lower() != ".srt":
            continue
        name_l = entry.name.lower()
        if not name_l.startswith(base):
            continue
        if EN_PATTERN.search(name_l) or ".openai.en.srt" in name_l:
            en_hits.append(entry.name)
        if HR_PATTERN.search(name_l):
            hr_hits.append(entry.name)

    return bool(en_hits), bool(hr_hits), en_hits, hr_hits


def find_best_generated_en_subtitle(video_path: Path) -> Path | None:
    base = video_path.with_suffix("")
    preferred = Path(str(base) + ".openai.synced.en.srt")
    if preferred.exists():
        return preferred

    secondary = Path(str(base) + ".openai.en.srt")
    if secondary.exists():
        return secondary

    pattern = f"{video_path.stem}.openai*.en.srt"
    candidates = [
        p
        for p in video_path.parent.glob(pattern)
        if p.is_file() and ".openai." in p.name.lower()
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_srt_entries(srt_text: str) -> list[dict[str, Any]]:
    blocks = re.split(r"\r?\n\r?\n+", srt_text.strip())
    entries: list[dict[str, Any]] = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2 or "-->" not in lines[1]:
            continue
        entries.append(
            {
                "index": lines[0].strip(),
                "timestamp": lines[1].rstrip(),
                "text_lines": lines[2:] if len(lines) > 2 else [""],
            }
        )
    return entries


def render_srt_entries(entries: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for entry in entries:
        text_lines = entry.get("text_lines") or [""]
        block = [str(entry.get("index", "")).strip(), str(entry.get("timestamp", "")).strip()]
        block.extend([str(line) for line in text_lines])
        blocks.append("\n".join(block).rstrip())
    rendered = "\n\n".join(blocks).strip()
    return rendered + ("\n" if rendered else "")


def translate_lines_batch(
    client: OpenAI,
    model: str,
    lines: list[str],
) -> list[str]:
    if not lines:
        return []

    system_prompt = (
        "You translate English subtitle lines to Croatian. "
        "Return only JSON in the form {\"translations\":[...]} with the exact same number of items. "
        "Preserve subtitle brevity, punctuation, speaker markers, and tags."
    )
    user_prompt = (
        "Translate each input line to Croatian. Keep ordering exactly the same.\n"
        f"INPUT_LINES_JSON={json_safe_dump(lines)}"
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content or ""
    translated = parse_translation_payload(content)
    if len(translated) != len(lines):
        raise RuntimeError(
            "Unexpected translation output format from OpenAI "
            f"(expected {len(lines)} lines, got {len(translated)})"
        )
    return [str(x) for x in translated]


def json_safe_dump(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_safe_load(value: str) -> dict[str, Any]:
    import json

    return json.loads(value)


def parse_translation_payload(content: str) -> list[str]:
    payload = content.strip()
    if not payload:
        raise RuntimeError("Empty translation output from OpenAI")

    # Strip simple markdown code fences if present.
    if payload.startswith("```"):
        payload = re.sub(r"^```[a-zA-Z0-9]*\n?", "", payload)
        payload = re.sub(r"\n?```$", "", payload).strip()

    candidates: list[str] = [payload]
    first_obj = payload.find("{")
    last_obj = payload.rfind("}")
    if 0 <= first_obj < last_obj:
        candidates.append(payload[first_obj : last_obj + 1])
    first_arr = payload.find("[")
    last_arr = payload.rfind("]")
    if 0 <= first_arr < last_arr:
        candidates.append(payload[first_arr : last_arr + 1])

    for candidate in candidates:
        try:
            parsed = json_safe_load(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("translations"), list):
            return [str(x) for x in parsed["translations"]]
        if isinstance(parsed, list):
            return [str(x) for x in parsed]

    raise RuntimeError("OpenAI translation output is not valid JSON")


def parse_duration_minutes(video_path: Path) -> float:
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run(
        [ffmpeg_bin, "-i", str(video_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
    if not match:
        raise RuntimeError(f"Failed to detect duration for: {video_path}")
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return (hours * 3600 + minutes * 60 + seconds) / 60.0


def queue_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    if item.get("started_at") and item["status"] in ACTIVE_STATUSES:
        started = datetime.fromisoformat(item["started_at"])
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        item["elapsed_seconds"] = elapsed
        if item.get("estimated_minutes"):
            est_total = float(item["estimated_minutes"]) * 70
            item["eta_seconds"] = max(0.0, est_total - elapsed)
    return item


def list_queue_items(limit: int = 300) -> list[dict[str, Any]]:
    conn = db_conn()
    try:
        rows = conn.execute(
            """
            SELECT * FROM queue_items
            ORDER BY
              CASE status WHEN 'completed' THEN 9 WHEN 'failed' THEN 8 ELSE 0 END,
              id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [queue_row_to_dict(row) for row in rows]
    finally:
        conn.close()


def list_history(limit: int = 300) -> list[dict[str, Any]]:
    conn = db_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def add_queue_item(
    root_key: str,
    rel_path: str,
    abs_path: Path,
    estimated_minutes: float | None,
    estimated_cost_usd: float | None,
    *,
    job_type: str = "transcribe_en",
    source_subtitle_path: str | None = None,
    hr_subtitle_existed: int = 0,
    alternate_output_used: int = 0,
) -> int:
    now = now_utc_iso()
    conn = db_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO queue_items(
                root_key, rel_path, abs_path, filename,
                job_type,
                status, current_step, created_at, updated_at,
                estimated_minutes, estimated_cost_usd,
                source_subtitle_path, hr_subtitle_existed, alternate_output_used
            ) VALUES(?, ?, ?, ?, ?, 'queued', 'queued', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                root_key,
                rel_path,
                str(abs_path),
                abs_path.name,
                job_type,
                now,
                now,
                estimated_minutes,
                estimated_cost_usd,
                source_subtitle_path,
                int(hr_subtitle_existed),
                int(alternate_output_used),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def claim_next_queue_item() -> dict[str, Any] | None:
    conn = db_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT id FROM queue_items WHERE status='queued' ORDER BY id ASC LIMIT 1"
        ).fetchone()
        if not row:
            conn.commit()
            return None

        qid = int(row["id"])
        now = now_utc_iso()
        conn.execute(
            """
            UPDATE queue_items
            SET status='preparing',
                current_step='preparing',
                started_at=?,
                updated_at=?
            WHERE id=?
            """,
            (now, now, qid),
        )
        conn.commit()
        final_row = conn.execute("SELECT * FROM queue_items WHERE id=?", (qid,)).fetchone()
        return dict(final_row) if final_row else None
    finally:
        conn.close()


def update_queue_item(
    queue_id: int,
    *,
    status: str | None = None,
    current_step: str | None = None,
    source_subtitle_path: str | None = None,
    hr_subtitle_existed: int | None = None,
    alternate_output_used: int | None = None,
    output_subtitle_path: str | None = None,
    synced_subtitle_path: str | None = None,
    error_message: str | None = None,
) -> None:
    fields = []
    values: list[Any] = []
    if status is not None:
        fields.append("status=?")
        values.append(status)
    if current_step is not None:
        fields.append("current_step=?")
        values.append(current_step)
    if source_subtitle_path is not None:
        fields.append("source_subtitle_path=?")
        values.append(source_subtitle_path)
    if hr_subtitle_existed is not None:
        fields.append("hr_subtitle_existed=?")
        values.append(int(hr_subtitle_existed))
    if alternate_output_used is not None:
        fields.append("alternate_output_used=?")
        values.append(int(alternate_output_used))
    if output_subtitle_path is not None:
        fields.append("output_subtitle_path=?")
        values.append(output_subtitle_path)
    if synced_subtitle_path is not None:
        fields.append("synced_subtitle_path=?")
        values.append(synced_subtitle_path)
    if error_message is not None:
        fields.append("error_message=?")
        values.append(error_message)

    fields.append("updated_at=?")
    values.append(now_utc_iso())
    values.append(queue_id)

    conn = db_conn()
    try:
        conn.execute(
            f"UPDATE queue_items SET {', '.join(fields)} WHERE id=?",
            values,
        )
        conn.commit()
    finally:
        conn.close()


def complete_queue_item(queue_id: int, status: str, error_message: str | None = None) -> None:
    conn = db_conn()
    try:
        row = conn.execute("SELECT * FROM queue_items WHERE id=?", (queue_id,)).fetchone()
        if not row:
            return

        now = now_utc_iso()
        elapsed = None
        if row["started_at"]:
            started = datetime.fromisoformat(row["started_at"])
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()

        conn.execute(
            """
            UPDATE queue_items
            SET status=?,
                current_step=?,
                finished_at=?,
                elapsed_seconds=?,
                error_message=?,
                updated_at=?
            WHERE id=?
            """,
            (status, status, now, elapsed, error_message, now, queue_id),
        )

        row2 = conn.execute("SELECT * FROM queue_items WHERE id=?", (queue_id,)).fetchone()
        if row2:
            conn.execute(
                """
                INSERT INTO history(
                    queue_item_id, root_key, rel_path, abs_path, filename, status, finished_at,
                    elapsed_seconds, estimated_minutes, estimated_cost_usd,
                    job_type, source_subtitle_path, hr_subtitle_existed, alternate_output_used,
                    output_subtitle_path, synced_subtitle_path, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row2["id"],
                    row2["root_key"],
                    row2["rel_path"],
                    row2["abs_path"],
                    row2["filename"],
                    row2["status"],
                    row2["finished_at"],
                    row2["elapsed_seconds"],
                    row2["estimated_minutes"],
                    row2["estimated_cost_usd"],
                    row2["job_type"],
                    row2["source_subtitle_path"],
                    row2["hr_subtitle_existed"],
                    row2["alternate_output_used"],
                    row2["output_subtitle_path"],
                    row2["synced_subtitle_path"],
                    row2["error_message"],
                ),
            )
        conn.commit()
    finally:
        conn.close()


def build_output_path(video_path: Path, overwrite_openai: bool) -> Path:
    base = video_path.with_suffix("")
    preferred = Path(str(base) + ".openai.en.srt")
    if overwrite_openai or not preferred.exists():
        return preferred
    suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(str(base) + f".openai.{suffix}.en.srt")


def build_hr_output_path(video_path: Path, force_alternate: bool) -> tuple[Path, bool]:
    base = video_path.with_suffix("")
    preferred = Path(str(base) + ".openai.hr.srt")
    if not preferred.exists() and not force_alternate:
        return preferred, False

    alt = Path(str(base) + ".openai.alt.hr.srt")
    if not alt.exists():
        return alt, True

    suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(str(base) + f".openai.alt.{suffix}.hr.srt"), True


def run_subtitle_sync(video_path: Path, input_srt: Path, logger: logging.Logger) -> Path | None:
    ffsubsync_bin = shutil.which("ffsubsync")
    if not ffsubsync_bin:
        logger.warning("ffsubsync not found in PATH, skipping sync stage")
        return None

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    ffbin_dir = TMP_DIR / "ffbin"
    ffbin_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_link = ffbin_dir / "ffmpeg"
    if not ffmpeg_link.exists():
        try:
            ffmpeg_link.symlink_to(ffmpeg_bin)
        except Exception:
            ffmpeg_link.write_text(f"#!/bin/sh\nexec '{ffmpeg_bin}' \"$@\"\n", encoding="utf-8")
            ffmpeg_link.chmod(0o755)

    synced_path = Path(str(video_path.with_suffix("")) + ".openai.synced.en.srt")
    cmd = [
        ffsubsync_bin,
        str(video_path),
        "-i",
        str(input_srt),
        "-o",
        str(synced_path),
    ]
    env = os.environ.copy()
    env["PATH"] = f"{ffbin_dir}:{env.get('PATH', '')}"
    process = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if process.returncode != 0:
        details = (process.stderr or process.stdout or "").strip()
        raise RuntimeError(details or f"ffsubsync exited with {process.returncode}")
    return synced_path


def process_translation_job(item: dict[str, Any], logger: logging.Logger) -> tuple[str, str, int]:
    queue_id = int(item["id"])
    video_path = Path(item["abs_path"])
    source_subtitle = item.get("source_subtitle_path") or ""
    source_path = Path(source_subtitle) if source_subtitle else find_best_generated_en_subtitle(video_path)
    if not source_path or not source_path.exists():
        raise RuntimeError("Missing generated English subtitle (.openai.en.srt)")

    settings = get_settings()
    api_key = settings.get("openai_api_key", "").strip()
    if not api_key:
        raise RuntimeError("OpenAI API key is not configured in settings")

    update_queue_item(
        queue_id,
        status="loading english subtitle",
        current_step="loading english subtitle",
        source_subtitle_path=str(source_path),
    )
    source_text = source_path.read_text(encoding="utf-8", errors="replace")
    entries = parse_srt_entries(source_text)
    if not entries:
        raise RuntimeError("Input EN subtitle is not a valid SRT file")

    text_indexes = [
        idx
        for idx, entry in enumerate(entries)
        if any(line.strip() for line in (entry.get("text_lines") or []))
    ]
    if not text_indexes:
        raise RuntimeError("Input EN subtitle does not contain translatable text")

    client = OpenAI(api_key=api_key)
    model = settings.get("translation_model", "gpt-4o-mini").strip() or "gpt-4o-mini"
    chunk_size = max(10, min(200, int_setting("translation_chunk_size", 40)))
    total = len(text_indexes)
    translated_done = 0

    for start in range(0, total, chunk_size):
        update_queue_item(
            queue_id,
            status="translating",
            current_step=f"translating ({min(start, total)}/{total})",
        )
        batch_indexes = text_indexes[start : start + chunk_size]
        batch_texts = ["\n".join(entries[i]["text_lines"]).strip() for i in batch_indexes]
        translated = translate_lines_batch(client, model, batch_texts)
        for entry_idx, translated_text in zip(batch_indexes, translated):
            translated_lines = [line.rstrip() for line in translated_text.splitlines()]
            entries[entry_idx]["text_lines"] = translated_lines or [""]
        translated_done += len(batch_indexes)
        update_queue_item(
            queue_id,
            status="translating",
            current_step=f"translating ({translated_done}/{total})",
        )

    update_queue_item(queue_id, status="writing hr srt", current_step="writing hr srt")
    force_alt = int(item.get("hr_subtitle_existed") or 0) == 1
    output_path, used_alt = build_hr_output_path(video_path, force_alternate=force_alt)
    output_path.write_text(render_srt_entries(entries), encoding="utf-8")

    return str(source_path), str(output_path), int(used_alt)


def process_queue_item(item: dict[str, Any], logger: logging.Logger) -> None:
    queue_id = int(item["id"])
    video_path = Path(item["abs_path"])
    tmp_audio_path = TMP_DIR / f"q{queue_id}-{int(time.time())}.m4a"
    synced_path: Path | None = None
    warning: str | None = None

    try:
        job_type = str(item.get("job_type") or "transcribe_en")
        if not video_path.exists():
            raise RuntimeError(f"Source file missing: {video_path}")
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise RuntimeError(f"Unsupported media extension: {video_path.suffix}")

        if job_type == "translate_hr":
            source_path, output_path, used_alt = process_translation_job(item, logger)
            update_queue_item(
                queue_id,
                source_subtitle_path=source_path,
                output_subtitle_path=output_path,
                alternate_output_used=used_alt,
            )
            complete_queue_item(queue_id, "completed", None)
            logger.info("Queue %s completed (translate_hr): %s", queue_id, video_path)
            return

        settings = get_settings()
        api_key = settings.get("openai_api_key", "").strip()
        if not api_key:
            raise RuntimeError("OpenAI API key is not configured in settings")

        update_queue_item(queue_id, status="preparing", current_step="preparing")

        update_queue_item(queue_id, status="extracting audio", current_step="extracting audio")
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        extract_cmd = [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "aac",
            str(tmp_audio_path),
        ]
        subprocess.run(extract_cmd, check=True)

        update_queue_item(queue_id, status="transcribing", current_step="transcribing")
        client = OpenAI(api_key=api_key)
        model = settings.get("openai_model", "whisper-1")

        with tmp_audio_path.open("rb") as af:
            response = client.audio.transcriptions.create(
                model=model,
                file=af,
                response_format="srt",
                language="en",
            )

        subtitle_text = response if isinstance(response, str) else str(response)
        if not subtitle_text.strip():
            raise RuntimeError("OpenAI returned empty subtitle output")

        update_queue_item(queue_id, status="writing srt", current_step="writing srt")
        overwrite_openai = settings.get("overwrite_openai_outputs", "0") == "1"
        output_path = build_output_path(video_path, overwrite_openai)
        output_path.write_text(subtitle_text, encoding="utf-8")
        update_queue_item(queue_id, output_subtitle_path=str(output_path))

        if settings.get("enable_ffsubsync", "1") == "1":
            try:
                update_queue_item(
                    queue_id,
                    status="writing srt",
                    current_step="writing srt (ffsubsync stage)",
                )
                synced_path = run_subtitle_sync(video_path, output_path, logger)
                if synced_path:
                    update_queue_item(queue_id, synced_subtitle_path=str(synced_path))
            except Exception as sync_exc:
                warning = f"ffsubsync failed: {sync_exc}"
                logger.warning("Queue %s: %s", queue_id, warning)

        complete_queue_item(queue_id, "completed", warning)
        logger.info("Queue %s completed: %s", queue_id, video_path)
    except Exception as exc:
        logger.exception("Queue %s failed", queue_id)
        complete_queue_item(queue_id, "failed", str(exc))
    finally:
        if tmp_audio_path.exists():
            try:
                tmp_audio_path.unlink()
            except Exception:
                pass


def count_active_jobs() -> int:
    placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
    conn = db_conn()
    try:
        row = conn.execute(
            f"""
            SELECT COUNT(*) AS c
            FROM queue_items
            WHERE status IN ({placeholders})
            """,
            tuple(ACTIVE_STATUSES),
        ).fetchone()
        return int(row["c"]) if row else 0
    finally:
        conn.close()


def worker_loop(stop_event: threading.Event, logger: logging.Logger) -> None:
    logger.info("Queue worker started")
    while not stop_event.is_set():
        try:
            concurrency = max(1, int_setting("concurrency_limit", 1))
            if count_active_jobs() >= concurrency:
                time.sleep(1.0)
                continue

            item = claim_next_queue_item()
            if not item:
                time.sleep(1.0)
                continue

            thread = threading.Thread(
                target=process_queue_item,
                args=(item, logger),
                daemon=True,
            )
            thread.start()
        except Exception:
            logger.exception("Worker loop error")
            time.sleep(2.0)
    logger.info("Queue worker stopped")


class SettingsUpdate(BaseModel):
    openai_api_key: str | None = None
    openai_model: str = "whisper-1"
    translation_model: str = "gpt-4o-mini"
    translation_chunk_size: int = 40
    price_per_minute_usd: float = 0.006
    concurrency_limit: int = 1
    overwrite_openai_outputs: bool = False
    enable_ffsubsync: bool = True
    show_tv_root: bool = True
    show_movies_root: bool = True
    show_anime_root: bool = True


class QueueItemInput(BaseModel):
    root_key: str
    rel_path: str


class EstimateRequest(BaseModel):
    items: list[QueueItemInput] = Field(default_factory=list)


class AddQueueRequest(BaseModel):
    items: list[QueueItemInput] = Field(default_factory=list)


class AddHrTranslationRequest(BaseModel):
    items: list[QueueItemInput] = Field(default_factory=list)
    confirm_existing_hr: bool = False


class AddFolderRequest(BaseModel):
    root_key: str
    folder_rel_path: str = ""
    confirm: bool = False


init_storage()
logger = init_logging()
init_db()
worker_stop = threading.Event()
worker_thread = threading.Thread(target=worker_loop, args=(worker_stop, logger), daemon=True)
worker_thread.start()

app = FastAPI(title="Subtitle Transcribe UI")


@app.on_event("shutdown")
def shutdown_event() -> None:
    worker_stop.set()
    if worker_thread.is_alive():
        worker_thread.join(timeout=5)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/roots")
def api_roots() -> dict[str, Any]:
    roots = media_roots()
    payload: dict[str, Any] = {}
    for key, info in roots.items():
        payload[key] = {
            "configured": bool(info["path"]),
            "path": str(info["path"]) if info["path"] else "",
            "enabled": bool(info["enabled"]),
        }
    return payload


@app.get("/api/settings")
def api_get_settings() -> dict[str, Any]:
    settings = get_settings()
    key = settings.get("openai_api_key", "")
    masked = ""
    if key:
        masked = key[:7] + "..." + key[-4:]
    return {
        "openai_api_key_masked": masked,
        "openai_api_key_set": bool(key),
        "openai_model": settings.get("openai_model", "whisper-1"),
        "translation_model": settings.get("translation_model", "gpt-4o-mini"),
        "translation_chunk_size": int_setting("translation_chunk_size", 40),
        "price_per_minute_usd": float_setting("price_per_minute_usd", 0.006),
        "concurrency_limit": int_setting("concurrency_limit", 1),
        "overwrite_openai_outputs": bool_setting("overwrite_openai_outputs", False),
        "enable_ffsubsync": bool_setting("enable_ffsubsync", True),
        "show_tv_root": bool_setting("show_tv_root", True),
        "show_movies_root": bool_setting("show_movies_root", True),
        "show_anime_root": bool_setting("show_anime_root", True),
    }


@app.post("/api/settings")
def api_save_settings(payload: SettingsUpdate) -> dict[str, Any]:
    if payload.openai_api_key is not None:
        if payload.openai_api_key.strip():
            save_setting("openai_api_key", payload.openai_api_key.strip())
    save_setting("openai_model", payload.openai_model.strip() or "whisper-1")
    save_setting("translation_model", payload.translation_model.strip() or "gpt-4o-mini")
    save_setting("translation_chunk_size", str(max(10, min(payload.translation_chunk_size, 200))))
    save_setting("price_per_minute_usd", f"{max(payload.price_per_minute_usd, 0.0):.6f}")
    save_setting("concurrency_limit", str(max(payload.concurrency_limit, 1)))
    save_setting("overwrite_openai_outputs", "1" if payload.overwrite_openai_outputs else "0")
    save_setting("enable_ffsubsync", "1" if payload.enable_ffsubsync else "0")
    save_setting("show_tv_root", "1" if payload.show_tv_root else "0")
    save_setting("show_movies_root", "1" if payload.show_movies_root else "0")
    save_setting("show_anime_root", "1" if payload.show_anime_root else "0")
    return {"ok": True}


@app.get("/api/browse")
def api_browse(
    root_key: str = Query(...),
    rel_path: str = Query(""),
    search: str = Query(""),
    no_en: bool = Query(False),
    has_en: bool = Query(False),
    no_hr: bool = Query(False),
    has_hr: bool = Query(False),
) -> dict[str, Any]:
    root = ensure_root_path(root_key)
    current = resolve_video_path(root_key, rel_path)
    if not current.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    if not current.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a folder")

    folders = []
    files = []
    search_l = search.strip().lower()

    entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    for entry in entries:
        if entry.is_dir():
            if search_l and search_l not in entry.name.lower():
                continue
            rel = str(entry.relative_to(root))
            folders.append({"name": entry.name, "rel_path": rel})
            continue

        if entry.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if search_l and search_l not in entry.name.lower():
            continue

        has_en_sub, has_hr_sub, en_files, hr_files = subtitle_presence(entry)
        generated_en = find_best_generated_en_subtitle(entry)
        if no_en and has_en_sub:
            continue
        if has_en and not has_en_sub:
            continue
        if no_hr and has_hr_sub:
            continue
        if has_hr and not has_hr_sub:
            continue

        rel = str(entry.relative_to(root))
        stat = entry.stat()
        files.append(
            {
                "name": entry.name,
                "rel_path": rel,
                "folder": str(entry.parent.relative_to(root)),
                "size_mb": round(stat.st_size / (1024 * 1024), 1),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "has_en_subtitle": has_en_sub,
                "has_hr_subtitle": has_hr_sub,
                "has_generated_en_subtitle": bool(generated_en),
                "generated_en_subtitle_path": str(generated_en) if generated_en else "",
                "en_files": en_files,
                "hr_files": hr_files,
            }
        )

    breadcrumbs = [{"name": root_key, "rel_path": ""}]
    if rel_path:
        parts = Path(rel_path).parts
        acc = []
        for part in parts:
            acc.append(part)
            breadcrumbs.append({"name": part, "rel_path": str(Path(*acc))})

    return {
        "root_key": root_key,
        "root_path": str(root),
        "current_rel_path": rel_path,
        "folders": folders,
        "files": files,
        "breadcrumbs": breadcrumbs,
    }


@app.post("/api/estimate")
def api_estimate(payload: EstimateRequest) -> dict[str, Any]:
    if not payload.items:
        return {"items": [], "total_minutes": 0.0, "total_cost_usd": 0.0}

    price = float_setting("price_per_minute_usd", 0.006)
    output_items = []
    total_minutes = 0.0

    for item in payload.items:
        video = resolve_video_path(item.root_key, item.rel_path)
        if not video.exists() or not video.is_file():
            continue
        if video.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        minutes = parse_duration_minutes(video)
        cost = minutes * price
        total_minutes += minutes
        output_items.append(
            {
                "root_key": item.root_key,
                "rel_path": item.rel_path,
                "filename": video.name,
                "minutes": round(minutes, 2),
                "cost_usd": round(cost, 4),
            }
        )

    return {
        "items": output_items,
        "total_minutes": round(total_minutes, 2),
        "total_cost_usd": round(total_minutes * price, 4),
        "price_per_minute_usd": price,
    }


@app.post("/api/queue/add")
def api_queue_add(payload: AddQueueRequest) -> dict[str, Any]:
    if not payload.items:
        raise HTTPException(status_code=400, detail="No items selected")

    price = float_setting("price_per_minute_usd", 0.006)
    created = 0
    ids: list[int] = []
    for item in payload.items:
        video = resolve_video_path(item.root_key, item.rel_path)
        if not video.exists() or not video.is_file():
            continue
        if video.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        minutes = None
        cost = None
        try:
            minutes = parse_duration_minutes(video)
            cost = minutes * price
        except Exception:
            minutes = None
            cost = None

        new_id = add_queue_item(item.root_key, item.rel_path, video, minutes, cost)
        ids.append(new_id)
        created += 1

    return {"ok": True, "created": created, "queue_ids": ids}


@app.post("/api/queue/add-translate-hr")
def api_queue_add_translate_hr(payload: AddHrTranslationRequest) -> dict[str, Any]:
    if not payload.items:
        raise HTTPException(status_code=400, detail="No items selected")

    created = 0
    ids: list[int] = []
    skipped_missing_en: list[dict[str, Any]] = []
    requires_confirmation: list[dict[str, Any]] = []

    for item in payload.items:
        video = resolve_video_path(item.root_key, item.rel_path)
        if not video.exists() or not video.is_file():
            continue
        if video.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        _, has_hr_sub, _, hr_files = subtitle_presence(video)
        source_en = find_best_generated_en_subtitle(video)
        if not source_en:
            skipped_missing_en.append({"rel_path": item.rel_path, "reason": "missing_generated_en"})
            continue

        if has_hr_sub and not payload.confirm_existing_hr:
            requires_confirmation.append(
                {
                    "rel_path": item.rel_path,
                    "hr_files": hr_files,
                }
            )
            continue

        new_id = add_queue_item(
            item.root_key,
            item.rel_path,
            video,
            None,
            None,
            job_type="translate_hr",
            source_subtitle_path=str(source_en),
            hr_subtitle_existed=1 if has_hr_sub else 0,
            alternate_output_used=1 if has_hr_sub else 0,
        )
        ids.append(new_id)
        created += 1

    return {
        "ok": True,
        "created": created,
        "queue_ids": ids,
        "requires_confirmation": requires_confirmation,
        "skipped_missing_generated_en": skipped_missing_en,
    }


@app.post("/api/queue/add-folder")
def api_queue_add_folder(payload: AddFolderRequest) -> dict[str, Any]:
    if not payload.confirm:
        raise HTTPException(status_code=400, detail="Folder queue requires explicit confirmation")

    folder = resolve_video_path(payload.root_key, payload.folder_rel_path)
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail="Folder does not exist")

    videos = sorted(
        [
            p
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ]
    )
    if not videos:
        return {"ok": True, "created": 0, "queue_ids": []}

    price = float_setting("price_per_minute_usd", 0.006)
    created = 0
    ids: list[int] = []
    root = ensure_root_path(payload.root_key)
    for video in videos:
        rel_path = str(video.relative_to(root))
        minutes = None
        cost = None
        try:
            minutes = parse_duration_minutes(video)
            cost = minutes * price
        except Exception:
            pass
        ids.append(add_queue_item(payload.root_key, rel_path, video, minutes, cost))
        created += 1
    return {"ok": True, "created": created, "queue_ids": ids}


@app.get("/api/queue")
def api_queue() -> dict[str, Any]:
    return {"items": list_queue_items(limit=500)}


@app.get("/api/history")
def api_history() -> dict[str, Any]:
    return {"items": list_history(limit=500)}


@app.post("/api/queue/{queue_id}/retry")
def api_retry(queue_id: int) -> dict[str, Any]:
    conn = db_conn()
    try:
        row = conn.execute("SELECT * FROM queue_items WHERE id=?", (queue_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Queue item not found")
        now = now_utc_iso()
        conn.execute(
            """
            UPDATE queue_items
            SET status='queued',
                current_step='queued',
                started_at=NULL,
                finished_at=NULL,
                elapsed_seconds=NULL,
                eta_seconds=NULL,
                error_message=NULL,
                updated_at=?
            WHERE id=?
            """,
            (now, queue_id),
        )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


@app.post("/api/queue/{queue_id}/remove")
def api_remove(queue_id: int) -> dict[str, Any]:
    conn = db_conn()
    try:
        conn.execute("DELETE FROM queue_items WHERE id=? AND status='queued'", (queue_id,))
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Subtitle Transcribe UI</title>
  <style>
    :root {
      --bg: #0f1724;
      --panel: #182235;
      --panel2: #1f2b42;
      --text: #e8eef9;
      --muted: #9fb0cc;
      --accent: #56c2ff;
      --ok: #2ecc71;
      --warn: #f1c40f;
      --bad: #ff6b6b;
      --border: #30415f;
    }
    body { margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background: linear-gradient(180deg,#101a2a,#0d1320); color:var(--text); }
    .wrap { padding: 14px; display:grid; grid-template-columns: 1.4fr 1fr; gap: 12px; }
    .panel { background: var(--panel); border:1px solid var(--border); border-radius: 10px; padding: 12px; }
    h2 { margin: 0 0 10px; font-size: 18px; }
    h3 { margin: 8px 0; font-size: 15px; color: var(--muted); }
    button { background: var(--panel2); color: var(--text); border:1px solid var(--border); border-radius:8px; padding: 6px 10px; cursor:pointer; }
    button:hover { border-color: var(--accent); }
    input, select { background: #0d1524; border:1px solid var(--border); color: var(--text); border-radius:6px; padding: 6px 8px; }
    .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }
    .muted { color: var(--muted); font-size: 12px; }
    .badge { padding: 2px 6px; border-radius: 999px; font-size: 11px; font-weight: 600; }
    .yes { background: rgba(46,204,113,.15); color: var(--ok); border: 1px solid rgba(46,204,113,.3); }
    .no { background: rgba(255,107,107,.15); color: var(--bad); border: 1px solid rgba(255,107,107,.3); }
    .table-wrap { max-height: 300px; overflow:auto; border:1px solid var(--border); border-radius:8px; }
    table { width:100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-bottom:1px solid var(--border); padding: 6px; text-align:left; vertical-align: top; }
    th { position: sticky; top:0; background: #1a2740; z-index: 1; }
    .queue-wrap { max-height: 340px; overflow:auto; border:1px solid var(--border); border-radius:8px; }
    .tabs { display:flex; gap: 8px; margin-bottom: 10px; }
    .tab { padding: 6px 10px; border:1px solid var(--border); border-radius:999px; cursor:pointer; }
    .tab.active { border-color: var(--accent); color: var(--accent); }
    .hidden { display:none; }
    .root-btn.active { border-color: var(--accent); color: var(--accent); }
    .status { font-weight: 700; text-transform: lowercase; }
    .status.completed { color: var(--ok); }
    .status.failed { color: var(--bad); }
    .status.transcribing { color: var(--accent); }
    .status.translating { color: var(--accent); }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h2>Media Browser</h2>
      <div class="row" id="rootButtons"></div>
      <div class="row">
        <input id="searchInput" placeholder="Search filename/folder in current view..." style="min-width: 320px;">
        <label><input type="checkbox" id="fNoEn"> No EN</label>
        <label><input type="checkbox" id="fHasEn"> Has EN</label>
        <label><input type="checkbox" id="fNoHr"> No HR</label>
        <label><input type="checkbox" id="fHasHr"> Has HR</label>
        <button onclick="refreshBrowse()">Apply</button>
      </div>
      <div class="row muted" id="breadcrumbs"></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th></th><th>Name</th><th>Folder</th><th>EN</th><th>HR</th><th>Size MB</th><th>Actions</th>
            </tr>
          </thead>
          <tbody id="folderRows"></tbody>
        </table>
      </div>
      <div class="row" style="margin-top:10px;">
        <button onclick="estimateSelected()">Estimate Cost</button>
        <button onclick="queueSelected()">Generate EN (Selected)</button>
        <button onclick="translateSelectedToHr()">Translate to HR (Selected)</button>
        <button onclick="queueCurrentFolder()">Queue Current Folder (Confirm)</button>
      </div>
      <div id="estimateBox" class="muted"></div>
    </div>

    <div>
      <div class="panel">
        <div class="tabs">
          <div class="tab active" id="tabQueue" onclick="showTab('queue')">Queue</div>
          <div class="tab" id="tabHistory" onclick="showTab('history')">History</div>
          <div class="tab" id="tabSettings" onclick="showTab('settings')">Settings</div>
        </div>

        <div id="paneQueue">
          <h3>Queue</h3>
          <div class="queue-wrap">
            <table>
              <thead>
                <tr><th>ID</th><th>Type</th><th>File</th><th>Status</th><th>Step</th><th>Elapsed</th><th>ETA</th><th>Output</th><th></th></tr>
              </thead>
              <tbody id="queueRows"></tbody>
            </table>
          </div>
        </div>

        <div id="paneHistory" class="hidden">
          <h3>History</h3>
          <div class="queue-wrap">
            <table>
              <thead>
                <tr><th>When</th><th>Type</th><th>File</th><th>Status</th><th>Elapsed</th><th>Cost</th><th>Output</th><th>Error</th></tr>
              </thead>
              <tbody id="historyRows"></tbody>
            </table>
          </div>
        </div>

        <div id="paneSettings" class="hidden">
          <h3>Settings</h3>
          <div class="row"><label>OpenAI key:</label><input id="sApiKey" type="password" style="min-width:320px;"></div>
          <div class="row muted" id="apiMask"></div>
          <div class="row">
            <label>Model:</label><input id="sModel" value="whisper-1">
            <label>Translate model:</label><input id="sTranslationModel" value="gpt-4o-mini">
            <label>HR chunk size:</label><input id="sTransChunk" type="number" value="40" min="10" max="200" style="width:90px;">
            <label>USD/min:</label><input id="sPrice" type="number" step="0.0001" value="0.006">
            <label>Concurrency:</label><input id="sConc" type="number" value="1" min="1" max="4" style="width:80px;">
          </div>
          <div class="row">
            <label><input type="checkbox" id="sOverwrite"> Overwrite existing .openai output</label>
            <label><input type="checkbox" id="sSync" checked> Run ffsubsync after transcription</label>
          </div>
          <div class="row">
            <label><input type="checkbox" id="sShowTv" checked> Show TV root</label>
            <label><input type="checkbox" id="sShowMovies" checked> Show Movies root</label>
            <label><input type="checkbox" id="sShowAnime" checked> Show Anime root</label>
          </div>
          <div class="row"><button onclick="saveSettings()">Save Settings</button></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let roots = {};
    let currentRoot = "tv";
    let currentRelPath = "";
    let currentFiles = [];

    function esc(v){ return String(v ?? "").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;"); }
    function fmtSec(v){ if(!v && v!==0) return ""; const s=Math.round(v); const m=Math.floor(s/60); const r=s%60; return `${m}m ${r}s`; }

    async function api(path, options={}){
      const res = await fetch(path, { headers: { "Content-Type":"application/json" }, ...options });
      if(!res.ok){ const txt = await res.text(); throw new Error(txt || `HTTP ${res.status}`); }
      return res.json();
    }

    async function loadRoots(){
      roots = await api("/api/roots");
      const rootButtons = document.getElementById("rootButtons");
      rootButtons.innerHTML = "";
      for(const key of ["tv","movies","anime"]){
        const item = roots[key];
        if(!item || !item.enabled) continue;
        const btn = document.createElement("button");
        btn.className = "root-btn" + (key===currentRoot ? " active" : "");
        btn.textContent = `${key.toUpperCase()}`;
        btn.onclick = () => { currentRoot = key; currentRelPath = ""; refreshBrowse(); loadRoots(); };
        rootButtons.appendChild(btn);
      }
    }

    async function refreshBrowse(){
      const search = encodeURIComponent(document.getElementById("searchInput").value || "");
      const noEn = document.getElementById("fNoEn").checked;
      const hasEn = document.getElementById("fHasEn").checked;
      const noHr = document.getElementById("fNoHr").checked;
      const hasHr = document.getElementById("fHasHr").checked;
      const url = `/api/browse?root_key=${encodeURIComponent(currentRoot)}&rel_path=${encodeURIComponent(currentRelPath)}&search=${search}&no_en=${noEn}&has_en=${hasEn}&no_hr=${noHr}&has_hr=${hasHr}`;
      const data = await api(url);
      currentFiles = data.files;

      const bc = document.getElementById("breadcrumbs");
      bc.innerHTML = data.breadcrumbs.map((b,i)=>`<a href="#" onclick="gotoPath('${esc(b.rel_path)}');return false;">${esc(b.name)}</a>${i<data.breadcrumbs.length-1 ? " / " : ""}`).join("");

      const rows = document.getElementById("folderRows");
      rows.innerHTML = "";
      for(const f of data.folders){
        rows.innerHTML += `<tr><td></td><td><a href="#" onclick="gotoPath('${esc(f.rel_path)}');return false;">📁 ${esc(f.name)}</a></td><td>${esc(f.rel_path)}</td><td></td><td></td><td></td><td></td></tr>`;
      }
      for(const f of data.files){
        const relEnc = encodeURIComponent(f.rel_path);
        const enState = f.has_generated_en_subtitle ? "YES (openai)" : (f.has_en_subtitle ? "YES" : "NO");
        rows.innerHTML += `<tr>
          <td><input type="checkbox" class="fileChk" data-rel="${esc(f.rel_path)}"></td>
          <td>🎬 ${esc(f.name)}</td>
          <td>${esc(f.folder)}</td>
          <td><span class="badge ${f.has_en_subtitle ? "yes":"no"}">${enState}</span></td>
          <td><span class="badge ${f.has_hr_subtitle ? "yes":"no"}">${f.has_hr_subtitle ? "YES":"NO"}</span></td>
          <td>${f.size_mb}</td>
          <td>
            <button onclick="queueSingleEn('${relEnc}')">Generate EN</button>
            <button onclick="translateSingleToHr('${relEnc}')" ${f.has_generated_en_subtitle ? "" : "disabled"}>Translate to HR</button>
          </td>
        </tr>`;
      }
    }

    function selectedItems(){
      const boxes = Array.from(document.querySelectorAll(".fileChk:checked"));
      return boxes.map(b => ({ root_key: currentRoot, rel_path: b.getAttribute("data-rel") }));
    }

    function gotoPath(rel){
      currentRelPath = rel || "";
      refreshBrowse();
    }

    async function estimateSelected(){
      const items = selectedItems();
      if(!items.length){ alert("Select at least one file."); return; }
      const data = await api("/api/estimate", { method:"POST", body: JSON.stringify({ items }) });
      document.getElementById("estimateBox").innerText =
        `Selected: ${data.items.length} | Minutes: ${data.total_minutes} | Estimated cost: $${data.total_cost_usd} (at $${data.price_per_minute_usd}/min)`;
      return data;
    }

    async function queueSelected(){
      const items = selectedItems();
      if(!items.length){ alert("Select at least one file."); return; }
      const est = await estimateSelected();
      const ok = confirm(`Queue ${est.items.length} file(s)?\\nEstimated total minutes: ${est.total_minutes}\\nEstimated cost: $${est.total_cost_usd}`);
      if(!ok) return;
      const res = await api("/api/queue/add", { method:"POST", body: JSON.stringify({ items }) });
      alert(`Queued ${res.created} item(s).`);
      await refreshQueue();
    }

    async function queueSingleEn(relEnc){
      const rel = decodeURIComponent(relEnc);
      const res = await api("/api/queue/add", { method:"POST", body: JSON.stringify({ items: [{ root_key: currentRoot, rel_path: rel }] }) });
      alert(`Queued ${res.created} EN transcription job.`);
      await refreshQueue();
    }

    async function queueHrTranslationJobs(items, confirmExistingHr){
      const res = await api("/api/queue/add-translate-hr", {
        method:"POST",
        body: JSON.stringify({ items, confirm_existing_hr: !!confirmExistingHr })
      });
      const missing = (res.skipped_missing_generated_en || []).length;
      const pendingConfirm = (res.requires_confirmation || []).length;
      alert(`Queued ${res.created} HR translation job(s).\\nMissing generated EN: ${missing}\\nNeeds HR confirmation: ${pendingConfirm}`);
      await refreshQueue();
    }

    async function translateSelectedToHr(){
      const items = selectedItems();
      if(!items.length){ alert("Select at least one file."); return; }
      const selectedMap = new Map(currentFiles.map(f => [f.rel_path, f]));
      const hasExistingHr = items.some(it => (selectedMap.get(it.rel_path) || {}).has_hr_subtitle);
      let confirmExistingHr = false;
      if(hasExistingHr){
        confirmExistingHr = confirm("Some selected files already have HR subtitles.\\nExisting HR will NOT be overwritten; alternate .openai.alt.hr.srt files will be created.\\nContinue?");
      }
      await queueHrTranslationJobs(items, confirmExistingHr);
    }

    async function translateSingleToHr(relEnc){
      const rel = decodeURIComponent(relEnc);
      const file = currentFiles.find(f => f.rel_path === rel);
      if(!file || !file.has_generated_en_subtitle){
        alert("Generated EN subtitle (.openai.en.srt/.openai.synced.en.srt) is required first.");
        return;
      }
      let confirmExistingHr = false;
      if(file.has_hr_subtitle){
        confirmExistingHr = confirm("HR subtitle already exists.\\nCreate alternate .openai.alt.hr.srt instead of overwrite?");
      }
      await queueHrTranslationJobs([{ root_key: currentRoot, rel_path: rel }], confirmExistingHr);
    }

    async function queueCurrentFolder(){
      const ok = confirm(`Queue ALL video files in current folder path (${currentRelPath || "/"}) recursively?`);
      if(!ok) return;
      const res = await api("/api/queue/add-folder", {
        method:"POST",
        body: JSON.stringify({ root_key: currentRoot, folder_rel_path: currentRelPath, confirm: true })
      });
      alert(`Queued ${res.created} item(s) from folder.`);
      await refreshQueue();
    }

    async function refreshQueue(){
      const data = await api("/api/queue");
      const rows = document.getElementById("queueRows");
      rows.innerHTML = "";
      for(const q of data.items){
        const jobType = q.job_type === "translate_hr" ? "translate_hr" : "transcribe_en";
        rows.innerHTML += `<tr>
          <td>${q.id}</td>
          <td>${esc(jobType)}</td>
          <td>${esc(q.filename)}<div class="muted">${esc(q.rel_path)}</div></td>
          <td><span class="status ${esc(q.status)}">${esc(q.status)}</span></td>
          <td>${esc(q.current_step || "")}</td>
          <td>${fmtSec(q.elapsed_seconds)}</td>
          <td>${fmtSec(q.eta_seconds)}</td>
          <td>
            ${esc(q.synced_subtitle_path || q.output_subtitle_path || "")}
            ${q.source_subtitle_path ? `<div class="muted">src: ${esc(q.source_subtitle_path)}</div>` : ""}
            ${q.alternate_output_used ? `<div class="muted">alternate HR output used</div>` : ""}
          </td>
          <td>
            ${q.status==="queued" ? `<button onclick="removeQueue(${q.id})">Remove</button>` : ""}
            ${q.status==="failed" ? `<button onclick="retryQueue(${q.id})">Retry</button>` : ""}
          </td>
        </tr>`;
      }
    }

    async function refreshHistory(){
      const data = await api("/api/history");
      const rows = document.getElementById("historyRows");
      rows.innerHTML = "";
      for(const h of data.items){
        const jobType = h.job_type === "translate_hr" ? "translate_hr" : "transcribe_en";
        rows.innerHTML += `<tr>
          <td>${esc(h.finished_at || "")}</td>
          <td>${esc(jobType)}</td>
          <td>${esc(h.filename || "")}<div class="muted">${esc(h.rel_path || "")}</div></td>
          <td>${esc(h.status || "")}</td>
          <td>${fmtSec(h.elapsed_seconds)}</td>
          <td>${h.estimated_cost_usd ? `$${h.estimated_cost_usd.toFixed ? h.estimated_cost_usd.toFixed(4) : h.estimated_cost_usd}` : ""}</td>
          <td>
            ${esc(h.synced_subtitle_path || h.output_subtitle_path || "")}
            ${h.source_subtitle_path ? `<div class="muted">src: ${esc(h.source_subtitle_path)}</div>` : ""}
            ${h.alternate_output_used ? `<div class="muted">alternate HR output used</div>` : ""}
          </td>
          <td class="muted">${esc(h.error_message || "")}</td>
        </tr>`;
      }
    }

    async function retryQueue(id){
      await api(`/api/queue/${id}/retry`, { method:"POST" });
      await refreshQueue();
    }
    async function removeQueue(id){
      await api(`/api/queue/${id}/remove`, { method:"POST" });
      await refreshQueue();
    }

    async function loadSettings(){
      const s = await api("/api/settings");
      document.getElementById("apiMask").innerText = s.openai_api_key_set ? `Stored key: ${s.openai_api_key_masked}` : "No API key saved";
      document.getElementById("sModel").value = s.openai_model || "whisper-1";
      document.getElementById("sTranslationModel").value = s.translation_model || "gpt-4o-mini";
      document.getElementById("sTransChunk").value = s.translation_chunk_size ?? 40;
      document.getElementById("sPrice").value = s.price_per_minute_usd ?? 0.006;
      document.getElementById("sConc").value = s.concurrency_limit ?? 1;
      document.getElementById("sOverwrite").checked = !!s.overwrite_openai_outputs;
      document.getElementById("sSync").checked = !!s.enable_ffsubsync;
      document.getElementById("sShowTv").checked = !!s.show_tv_root;
      document.getElementById("sShowMovies").checked = !!s.show_movies_root;
      document.getElementById("sShowAnime").checked = !!s.show_anime_root;
    }

    async function saveSettings(){
      const payload = {
        openai_api_key: document.getElementById("sApiKey").value || null,
        openai_model: document.getElementById("sModel").value || "whisper-1",
        translation_model: document.getElementById("sTranslationModel").value || "gpt-4o-mini",
        translation_chunk_size: parseInt(document.getElementById("sTransChunk").value || "40", 10),
        price_per_minute_usd: parseFloat(document.getElementById("sPrice").value || "0.006"),
        concurrency_limit: parseInt(document.getElementById("sConc").value || "1", 10),
        overwrite_openai_outputs: document.getElementById("sOverwrite").checked,
        enable_ffsubsync: document.getElementById("sSync").checked,
        show_tv_root: document.getElementById("sShowTv").checked,
        show_movies_root: document.getElementById("sShowMovies").checked,
        show_anime_root: document.getElementById("sShowAnime").checked
      };
      await api("/api/settings", { method:"POST", body: JSON.stringify(payload) });
      document.getElementById("sApiKey").value = "";
      alert("Settings saved.");
      await loadSettings();
      await loadRoots();
    }

    function showTab(name){
      for(const id of ["queue","history","settings"]){
        document.getElementById("pane"+id[0].toUpperCase()+id.slice(1)).classList.toggle("hidden", id!==name);
        document.getElementById("tab"+id[0].toUpperCase()+id.slice(1)).classList.toggle("active", id===name);
      }
    }

    async function boot(){
      await loadSettings();
      await loadRoots();
      await refreshBrowse();
      await refreshQueue();
      await refreshHistory();
      setInterval(refreshQueue, 4000);
      setInterval(refreshHistory, 15000);
    }
    boot().catch(err => alert("UI init error: " + err.message));
  </script>
</body>
</html>
"""
