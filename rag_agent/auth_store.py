"""Local-only account storage with plaintext passwords and ephemeral login tokens."""
from __future__ import annotations

import json
import os
import secrets
import tempfile
import threading
from pathlib import Path

try:
    from .memory import validate_user_id
except ImportError:  # pragma: no cover - direct script compatibility
    from memory import validate_user_id


class AuthInputError(ValueError):
    pass


class UserExistsError(ValueError):
    pass


class InvalidCredentialsError(ValueError):
    pass


class LocalAuthStore:
    """Persist local credentials and keep issued login tokens in process memory."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self._lock = threading.RLock()
        self._tokens: dict[str, str] = {}
        self._users = self._load()

    @staticmethod
    def _validate(username, password):
        try:
            username = validate_user_id(username.strip() if isinstance(username, str) else username)
        except ValueError as exc:
            raise AuthInputError("invalid username") from exc
        if not isinstance(password, str) or not password or len(password) > 128:
            raise AuthInputError("password must contain 1-128 characters")
        return username, password

    def _load(self):
        if not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid local user file: {self.path}") from exc
        users = data.get("users") if isinstance(data, dict) else None
        if not isinstance(users, dict) or any(
            not isinstance(name, str)
            or not isinstance(record, dict)
            or not isinstance(record.get("password"), str)
            for name, record in users.items()
        ):
            raise RuntimeError(f"invalid local user file: {self.path}")
        return users

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"users": self._users}, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)

    def _issue(self, username):
        token = secrets.token_urlsafe(32)
        self._tokens[token] = username
        return token

    def register(self, username, password):
        username, password = self._validate(username, password)
        with self._lock:
            if username in self._users:
                raise UserExistsError("username already exists")
            self._users[username] = {"password": password}
            try:
                self._save()
            except Exception:
                self._users.pop(username, None)
                raise
            return self._issue(username)

    def login(self, username, password):
        username, password = self._validate(username, password)
        with self._lock:
            record = self._users.get(username)
            if not record or not secrets.compare_digest(record["password"], password):
                raise InvalidCredentialsError("invalid username or password")
            return self._issue(username)

    def authenticate(self, token):
        if not isinstance(token, str) or not token:
            return None
        with self._lock:
            return self._tokens.get(token)

    def logout(self, token):
        with self._lock:
            return self._tokens.pop(token, None) is not None
