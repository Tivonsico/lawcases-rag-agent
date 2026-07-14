"""Flask application factory for the legal RAG service.

Importing this module never initializes embeddings, indexes, databases or LLM clients.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

from flask import Flask, Response, g, jsonify, request, send_from_directory, stream_with_context

try:
    from .auth_store import (AuthInputError, InvalidCredentialsError,
                             LocalAuthStore, UserExistsError)
    from .config import LOG_LEVEL, BM25_INDEX_PATH, RUNTIME_DIR, LONG_TERM_DB
    from .init_db import load_bm25_index
    from .memory import LongTermMemory, MemoryManager, validate_session_id
except ImportError:  # pragma: no cover - direct script compatibility
    from auth_store import (AuthInputError, InvalidCredentialsError,
                            LocalAuthStore, UserExistsError)
    from config import LOG_LEVEL, BM25_INDEX_PATH, RUNTIME_DIR, LONG_TERM_DB
    from init_db import load_bm25_index
    from memory import LongTermMemory, MemoryManager, validate_session_id

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


class SlidingWindowLimiter:
    def __init__(self, limit=60, window_seconds=60):
        self.limit, self.window = int(limit), float(window_seconds)
        self._events, self._lock = defaultdict(deque), threading.Lock()

    def allow(self, key):
        now = time.monotonic()
        with self._lock:
            events = self._events[key]
            while events and events[0] <= now - self.window: events.popleft()
            if len(events) >= self.limit: return False
            events.append(now)
            return True


class SessionEntry:
    def __init__(self, memory, agent):
        self.memory, self.agent = memory, agent
        self.lock, self.touched = threading.Lock(), time.monotonic()


class SessionCache:
    def __init__(self, max_size=128, ttl=1800):
        self.max_size, self.ttl = int(max_size), float(ttl)
        self._items, self._lock = OrderedDict(), threading.Lock()

    def get_or_create(self, key, factory):
        now = time.monotonic()
        with self._lock:
            for stale in [k for k, v in self._items.items() if now - v.touched > self.ttl]:
                self._items.pop(stale, None)
            entry = self._items.pop(key, None)
            if entry is None: entry = factory()
            entry.touched = now
            self._items[key] = entry
            while len(self._items) > self.max_size: self._items.popitem(last=False)
            return entry


def _default_services(app):
    """Build expensive services only when create_app is called without injected fakes."""
    try:
        from .agent import LegalAgent
        from .embedding import EmbeddingService
        from .retriever import BM25Index, HybridRetriever
        from .vector_store import VectorStore
    except ImportError:  # pragma: no cover
        from agent import LegalAgent
        from embedding import EmbeddingService
        from retriever import BM25Index, HybridRetriever
        from vector_store import VectorStore

    embed = EmbeddingService(mock=False)
    vector_store = VectorStore(embed_service=embed)
    # Pickle is executable input: load only the administrator-configured fixed path.
    bm25, _chunks, _manifest = load_bm25_index(BM25_INDEX_PATH, vector_store)
    retriever = HybridRetriever(vector_store=vector_store, embed_service=embed, bm25_index=bm25)
    return {"agent_factory": lambda memory: LegalAgent(retriever=retriever, memory=memory),
            "vector_count": vector_store.count}


def create_app(config=None, services=None):
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=256 * 1024,
        MAX_MESSAGE_CHARS=8000,
        RUNTIME_DIR=RUNTIME_DIR,
        LONG_TERM_DB=LONG_TERM_DB,
        RATE_LIMIT=60, RATE_WINDOW_SECONDS=60,
        SESSION_CACHE_SIZE=128, SESSION_CACHE_TTL=1800,
        GLOBAL_CONCURRENCY=8, USER_CONCURRENCY=2,
    )
    if config: app.config.update(config)
    service_map = dict(services) if services is not None else _default_services(app)
    if not callable(service_map.get("agent_factory")):
        raise ValueError("services.agent_factory is required")

    runtime = Path(app.config["RUNTIME_DIR"]).resolve()
    auth_store = service_map.get("auth_store") or LocalAuthStore(runtime / "auth" / "users.json")
    cache = SessionCache(app.config["SESSION_CACHE_SIZE"], app.config["SESSION_CACHE_TTL"])
    limiter = SlidingWindowLimiter(app.config["RATE_LIMIT"], app.config["RATE_WINDOW_SECONDS"])
    global_sem = threading.BoundedSemaphore(int(app.config["GLOBAL_CONCURRENCY"]))
    user_sems, sem_lock = {}, threading.Lock()

    def error(message, status): return jsonify({"error": message, "status": status}), status

    @app.errorhandler(413)
    def too_large(_exc): return error("request too large", 413)

    @app.errorhandler(404)
    def not_found(_exc): return error("not found", 404)

    @app.errorhandler(Exception)
    def unexpected(exc):
        logger.exception("unhandled API error", exc_info=exc)
        return error("internal server error", 500)

    @app.before_request
    def authenticate_and_limit():
        if not request.path.startswith("/api/") or request.path == "/api/health": return None
        client_ip = request.remote_addr or "unknown"
        if request.path in {"/api/auth/register", "/api/auth/login"}:
            if not limiter.allow(("auth", client_ip)): return error("rate limit exceeded", 429)
            return None
        header = request.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else ""
        user_id = auth_store.authenticate(token)
        if user_id is None:
            return error("unauthorized", 401)
        g.user_id, g.auth_token = user_id, token
        if not limiter.allow((g.user_id, client_ip)): return error("rate limit exceeded", 429)

    def session_path(user_id, session_id):
        validate_session_id(session_id)
        path = (runtime / "users" / user_id / "sessions" / f"{session_id}.json").resolve()
        if runtime != path and runtime not in path.parents: raise ValueError("invalid session path")
        return path

    def make_entry(user_id, session_id):
        long_mem = LongTermMemory(app.config["LONG_TERM_DB"], user_id=user_id)
        memory = MemoryManager.create_session(session_id, long_mem, user_id, runtime)
        return SessionEntry(memory, service_map["agent_factory"](memory))

    def owned_entry(user_id, session_id, must_exist=True):
        path = session_path(user_id, session_id)
        if must_exist and not path.is_file(): return None
        return cache.get_or_create((user_id, session_id), lambda: make_entry(user_id, session_id))

    def json_body():
        if not request.is_json: return None, error("application/json required", 400)
        data = request.get_json(silent=True)
        if not isinstance(data, dict): return None, error("JSON object required", 400)
        return data, None

    def auth_fields():
        data, failure = json_body()
        if failure: return None, None, failure
        return data.get("username"), data.get("password"), None

    @app.post("/api/auth/register")
    def register():
        username, password, failure = auth_fields()
        if failure: return failure
        try:
            token = auth_store.register(username, password)
        except AuthInputError as exc:
            return error(str(exc), 400)
        except UserExistsError:
            return error("username already exists", 409)
        return jsonify({"username": username.strip(), "token": token}), 201

    @app.post("/api/auth/login")
    def login():
        username, password, failure = auth_fields()
        if failure: return failure
        try:
            token = auth_store.login(username, password)
        except AuthInputError as exc:
            return error(str(exc), 400)
        except InvalidCredentialsError:
            return error("invalid username or password", 401)
        return jsonify({"username": username.strip(), "token": token})

    @app.get("/api/auth/me")
    def current_user():
        return jsonify({"username": g.user_id})

    @app.post("/api/auth/logout")
    def logout():
        auth_store.logout(g.auth_token)
        return jsonify({"status": "ok"})

    def parse_chat():
        data, failure = json_body()
        if failure: return None, None, failure
        message, sid = data.get("message"), data.get("session_id")
        if not isinstance(message, str) or not message.strip(): return None, None, error("message must be a non-empty string", 400)
        if len(message) > int(app.config["MAX_MESSAGE_CHARS"]): return None, None, error("message too long", 413)
        if not isinstance(sid, str): return None, None, error("session_id must be a string", 400)
        try: validate_session_id(sid)
        except ValueError: return None, None, error("invalid session id", 400)
        entry = owned_entry(g.user_id, sid)
        if entry is None: return None, None, error("session not found", 404)
        return message.strip(), entry, None

    def acquire_capacity(user_id):
        with sem_lock:
            user_sem = user_sems.setdefault(user_id, threading.BoundedSemaphore(int(app.config["USER_CONCURRENCY"])))
        if not global_sem.acquire(blocking=False): return None
        if not user_sem.acquire(blocking=False): global_sem.release(); return None
        return user_sem

    @app.get("/")
    def index(): return send_from_directory(str(STATIC_DIR), "index.html")

    @app.get("/api/health")
    def health():
        count = service_map.get("vector_count", 0)
        return jsonify({"status": "ok", "vector_count": count() if callable(count) else count})

    @app.get("/api/sessions")
    def list_sessions():
        folder = runtime / "users" / g.user_id / "sessions"
        result = []
        for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if folder.is_dir() else []:
            try:
                sid = validate_session_id(path.stem); data = json.loads(path.read_text(encoding="utf-8"))
                buffer = data.get("buffer", [])
                preview = next((x.get("user", "")[:80] for x in buffer if x.get("user")), "")
                result.append({"id": sid, "preview": preview or data.get("running_summary", "")[:60] or "(empty session)", "message_count": len(buffer)})
            except (OSError, ValueError, json.JSONDecodeError):
                logger.warning("ignored invalid session file: %s", path.name)
        return jsonify({"sessions": result})

    @app.post("/api/sessions")
    def create_session_route():
        sid = __import__("uuid").uuid4().hex
        entry = owned_entry(g.user_id, sid, must_exist=False)
        entry.memory.save()
        return jsonify({"session_id": sid, "preview": "(new session)"}), 201

    @app.get("/api/sessions/<session_id>/messages")
    def messages(session_id):
        try: entry = owned_entry(g.user_id, session_id)
        except ValueError: return error("invalid session id", 400)
        if entry is None: return error("session not found", 404)
        output = []
        for item in entry.memory.session.buffer:
            if item.get("user"): output.append({"role": "user", "content": item["user"]})
            if item.get("agent"): output.append({"role": "assistant", "content": item["agent"]})
        return jsonify({"messages": output})

    @app.post("/api/chat")
    def chat():
        message, entry, failure = parse_chat()
        if failure: return failure
        user_sem = acquire_capacity(g.user_id)
        if user_sem is None: return error("service busy", 503)
        try:
            if not entry.lock.acquire(blocking=False): return error("session busy", 429)
            try: response = entry.agent.answer(message)
            finally: entry.lock.release()
            return jsonify({"response": response, "session_id": entry.memory.session_id})
        finally: user_sem.release(); global_sem.release()

    @app.post("/api/chat/stream")
    def chat_stream():
        message, entry, failure = parse_chat()
        if failure: return failure
        user_sem = acquire_capacity(g.user_id)
        if user_sem is None: return error("service busy", 503)
        if not entry.lock.acquire(blocking=False):
            user_sem.release(); global_sem.release()
            return error("session busy", 429)

        def generate():
            try:
                yield f"data: {json.dumps({'type':'meta','session_id':entry.memory.session_id})}\n\n"
                for chunk in entry.agent.answer_stream(message):
                    payload = {"type": "done"} if chunk == "[DONE]" else {"type": "chunk", "content": chunk}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except Exception:
                logger.exception("stream failed")
                yield f"data: {json.dumps({'type':'error','content':'stream failed'})}\n\n"
            finally:
                entry.lock.release(); user_sem.release(); global_sem.release()
        return Response(stream_with_context(generate()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    app.extensions["session_cache"] = cache
    app.extensions["auth_store"] = auth_store
    return app


if __name__ == "__main__":
    import os
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")), debug=False)
