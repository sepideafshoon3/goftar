"""SQLite cache so the same audio + same settings is never transcribed twice."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from goftar.core.types import Transcript


class TranscriptCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS transcripts ("
            "key TEXT PRIMARY KEY, payload TEXT NOT NULL, created REAL NOT NULL)"
        )
        self._db.commit()

    @staticmethod
    def make_key(audio_path: str | Path, fingerprint: str) -> str:
        digest = hashlib.sha256()
        with open(audio_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"{digest.hexdigest()}:{fingerprint}"

    def get(self, key: str) -> Transcript | None:
        with self._lock:
            row = self._db.execute(
                "SELECT payload FROM transcripts WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return Transcript.from_dict(json.loads(row[0]))

    def put(self, key: str, transcript: Transcript) -> None:
        payload = json.dumps(transcript.to_dict(), ensure_ascii=False)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO transcripts (key, payload, created) VALUES (?, ?, ?)",
                (key, payload, time.time()),
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()
