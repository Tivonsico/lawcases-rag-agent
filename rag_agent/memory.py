"""Conversation memory with per-user storage and atomic persistence."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    from .config import LONG_TERM_DB
except ImportError:  # pragma: no cover - script compatibility
    from config import LONG_TERM_DB

KEEP_LAST_N = 3
MAX_CHARS = 2000
SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not SESSION_ID_RE.fullmatch(session_id):
        raise ValueError("invalid session id")
    return session_id


def validate_user_id(user_id: str) -> str:
    if not isinstance(user_id, str) or not USER_ID_RE.fullmatch(user_id):
        raise ValueError("invalid user id")
    return user_id


class SummaryBufferMemory:
    def __init__(self, session_id: str, user_id: str = "default_user", runtime_dir=None):
        self.session_id = validate_session_id(session_id)
        self.user_id = validate_user_id(user_id)
        self.runtime_dir = Path(runtime_dir or Path(__file__).resolve().parent / "runtime")
        self.buffer: List[Dict] = []
        self.running_summary = ""
        self._load()

    @property
    def storage_path(self) -> Path:
        root = self.runtime_dir.resolve()
        path = (root / "users" / self.user_id / "sessions" / f"{self.session_id}.json").resolve()
        if root != path and root not in path.parents:
            raise ValueError("session path escapes runtime directory")
        return path

    def _load(self):
        path = self.storage_path
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.buffer = data.get("buffer", []) if isinstance(data.get("buffer", []), list) else []
            self.running_summary = data.get("running_summary", "")
        except (OSError, ValueError, json.JSONDecodeError):
            self.buffer, self.running_summary = [], ""

    def save(self):
        path = self.storage_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"user_id": self.user_id, "session_id": self.session_id,
                   "buffer": self.buffer, "running_summary": self.running_summary}
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.session_id}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def add(self, user_query: str, agent_response: str):
        self.buffer.append({"timestamp": datetime.now(timezone.utc).isoformat(),
                            "user": user_query, "agent": agent_response})

    def clear(self): self.buffer.clear()
    def char_count(self): return sum(len(x.get("user", "")) + len(x.get("agent", "")) for x in self.buffer)
    def should_summarize(self): return self.char_count() > MAX_CHARS and len(self.buffer) >= 3
    def get_old_rounds(self): return list(self.buffer[:-KEEP_LAST_N]) if len(self.buffer) > KEEP_LAST_N else []
    def pop_old_rounds(self, n): self.buffer = self.buffer[n:] if n > 0 else self.buffer

    def update_summary(self, new_summary: str):
        self.running_summary = (self.running_summary + "\n" + new_summary).strip()[-1000:]

    def get_messages(self):
        messages = []
        if self.running_summary:
            messages.append({"role": "assistant", "content": f"(对话历史摘要)\n{self.running_summary}"})
        for item in self.buffer:
            if item.get("user"): messages.append({"role": "user", "content": item["user"]})
            if item.get("agent"): messages.append({"role": "assistant", "content": item["agent"]})
        return messages, self.running_summary

    @property
    def summary_prompt(self):
        old = self.get_old_rounds()
        if not old: return ""
        text = "\n".join(f"用户: {x.get('user','')}\n客服: {x.get('agent','')[:150]}" for x in old)
        return f"请把以下旧对话压缩为不超过500字的事实摘要，保留用户原意：\n{text}"

    def parse_summary(self, output): return output.strip()[:600]


class LongTermMemory:
    def __init__(self, db_path: str = None, user_id: str = "default_user"):
        self.db_path = str(db_path or LONG_TERM_DB)
        self.user_id = validate_user_id(user_id)
        self._init_db()

    def _connect(self):
        parent = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(parent, exist_ok=True)
        return sqlite3.connect(self.db_path, timeout=10)

    def _init_db(self):
        with closing(self._connect()) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS user_profile (
                user_id TEXT PRIMARY KEY, personality TEXT DEFAULT '', preferences TEXT DEFAULT '',
                risk_type TEXT DEFAULT '', knowledge_level TEXT DEFAULT '', notes TEXT DEFAULT '',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            conn.execute("INSERT OR IGNORE INTO user_profile (user_id) VALUES (?)", (self.user_id,))
            conn.commit()

    def get_profile(self, user_id: Optional[str] = None):
        uid = validate_user_id(user_id or self.user_id)
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT personality,preferences,risk_type,knowledge_level,notes FROM user_profile WHERE user_id=?", (uid,)).fetchone()
        keys = ("personality", "preferences", "risk_type", "knowledge_level", "notes")
        return dict(zip(keys, row)) if row else {}

    def update_profile(self, profile: Dict, user_id: Optional[str] = None):
        uid = validate_user_id(user_id or self.user_id)
        values = [profile.get(k, "") for k in ("personality", "preferences", "risk_type", "knowledge_level", "notes")]
        with closing(self._connect()) as conn:
            conn.execute("INSERT OR IGNORE INTO user_profile (user_id) VALUES (?)", (uid,))
            conn.execute("""UPDATE user_profile SET personality=COALESCE(NULLIF(?,''),personality),
              preferences=COALESCE(NULLIF(?,''),preferences),risk_type=COALESCE(NULLIF(?,''),risk_type),
              knowledge_level=COALESCE(NULLIF(?,''),knowledge_level),notes=COALESCE(NULLIF(?,''),notes),
              updated_at=CURRENT_TIMESTAMP WHERE user_id=?""", (*values, uid))
            conn.commit()

    def extract_profile_prompt(self, conversation):
        text = "\n".join(f"用户: {x.get('user','')}" for x in conversation[-3:])
        return f"根据提问提取用户画像并返回JSON（personality/preferences/risk_type/knowledge_level）：\n{text}"

    def profile_to_text(self, user_id: Optional[str] = None):
        profile = self.get_profile(user_id)
        labels = {"personality": "性格", "preferences": "偏好", "risk_type": "风险类型", "knowledge_level": "法律知识水平"}
        lines = [f"- {labels[k]}：{profile[k]}" for k in labels if profile.get(k)]
        return "## 用户长期画像\n" + "\n".join(lines) if lines else ""


class MemoryManager:
    def __init__(self, session: SummaryBufferMemory, long_term: LongTermMemory, user_id: Optional[str] = None):
        self.session = session
        self.user_id = validate_user_id(user_id or session.user_id)
        self.long = long_term

    @property
    def session_id(self): return self.session.session_id

    @staticmethod
    def create_session(session_id=None, long_term=None, user_id="default_user", runtime_dir=None):
        sid = validate_session_id(session_id) if session_id else uuid.uuid4().hex
        long = long_term or LongTermMemory(user_id=user_id)
        return MemoryManager(SummaryBufferMemory(sid, user_id, runtime_dir), long, user_id)

    def add_dialogue(self, user_query, agent_response): self.session.add(user_query, agent_response); self.session.save()
    def need_summarize(self): return self.session.should_summarize()
    def get_old_rounds(self): return self.session.get_old_rounds()
    def pop_old_rounds(self, n): self.session.pop_old_rounds(n)
    def update_summary(self, value): self.session.update_summary(value); self.session.save()
    def get_messages(self): return self.session.get_messages()[0]
    def get_long_text(self): return self.long.profile_to_text(self.user_id)
    def save(self): self.session.save()
