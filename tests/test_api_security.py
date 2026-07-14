import importlib
import json
import sys
from pathlib import Path


def create(client, headers):
    response = client.post("/api/sessions", headers=headers)
    assert response.status_code == 201
    return response.get_json()["session_id"]


def register(client, username, password="password"):
    response = client.post(
        "/api/auth/register", json={"username": username, "password": password}
    )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.get_json()['token']}"}


def test_import_has_no_real_service_initialization():
    sys.modules.pop("rag_agent.api_server", None)
    module = importlib.import_module("rag_agent.api_server")
    assert not hasattr(module, "app")


def test_authentication_is_required_and_client_user_id_is_ignored(client, alice_headers):
    assert client.get("/api/sessions").status_code == 401
    sid = create(client, alice_headers)
    response = client.post("/api/chat", headers=alice_headers,
                           json={"session_id": sid, "message": "hello", "user_id": "bob"})
    assert response.status_code == 200
    # The forged body identity must not move the session into Bob's namespace.
    bob = register(client, "bob")
    assert client.get(f"/api/sessions/{sid}/messages", headers=bob).status_code == 404
    messages = client.get(f"/api/sessions/{sid}/messages", headers=alice_headers).get_json()["messages"]
    assert messages[0]["content"] == "hello"


def test_user_cannot_list_or_read_another_users_session(client, alice_headers):
    sid = create(client, alice_headers)
    client.post("/api/chat", headers=alice_headers, json={"session_id": sid, "message": "secret"})
    bob = register(client, "bob")
    assert client.get("/api/sessions", headers=bob).get_json()["sessions"] == []
    assert client.get(f"/api/sessions/{sid}/messages", headers=bob).status_code == 404
    assert client.post("/api/chat", headers=bob, json={"session_id": sid, "message": "steal"}).status_code == 404


def test_session_id_is_strict_and_traversal_is_rejected(client, alice_headers):
    for sid in ("abc", "../secrets", "A" * 32, "0" * 31):
        response = client.post("/api/chat", headers=alice_headers, json={"session_id": sid, "message": "x"})
        assert response.status_code == 400


def test_json_types_and_message_size_are_bounded(client, alice_headers):
    sid = create(client, alice_headers)
    assert client.post("/api/chat", headers=alice_headers, data="plain").status_code == 400
    assert client.post("/api/chat", headers=alice_headers, json=[]).status_code == 400
    assert client.post("/api/chat", headers=alice_headers, json={"session_id": sid, "message": 1}).status_code == 400
    assert client.post("/api/chat", headers=alice_headers, json={"session_id": sid, "message": "x" * 21}).status_code == 413


def test_request_body_limit_returns_413_before_json_processing(tmp_path):
    from rag_agent.api_server import create_app
    from conftest import FakeAgent
    app = create_app({"TESTING": True,
                      "RUNTIME_DIR": str(tmp_path / "r"), "LONG_TERM_DB": str(tmp_path / "r" / "m.db"),
                      "MAX_CONTENT_LENGTH": 64, "RATE_LIMIT": 10}, {"agent_factory": FakeAgent})
    client = app.test_client()
    headers = register(client, "user")
    response = client.post("/api/chat", headers=headers, data=b"x" * 65,
                           content_type="application/json")
    assert response.status_code == 413


def test_session_cache_evicts_least_recently_used_entry():
    from rag_agent.api_server import SessionCache
    class Entry:
        pass
    cache = SessionCache(max_size=2, ttl=3600)
    cache.get_or_create("a", Entry)
    cache.get_or_create("b", Entry)
    a = cache.get_or_create("a", Entry)
    cache.get_or_create("c", Entry)
    replacement = Entry()
    assert cache.get_or_create("a", Entry) is a
    assert cache.get_or_create("b", lambda: replacement) is replacement


def test_rate_limit_returns_429(tmp_path):
    from rag_agent.api_server import create_app
    from conftest import FakeAgent
    app = create_app({"TESTING": True,
                      "RUNTIME_DIR": str(tmp_path / "r"), "LONG_TERM_DB": str(tmp_path / "r" / "m.db"),
                      "RATE_LIMIT": 1}, {"agent_factory": FakeAgent})
    client = app.test_client(); headers = register(client, "user")
    assert client.get("/api/sessions", headers=headers).status_code == 200
    assert client.get("/api/sessions", headers=headers).status_code == 429


def test_stream_and_persistence_are_user_scoped(client, alice_headers):
    sid = create(client, alice_headers)
    response = client.post("/api/chat/stream", headers=alice_headers,
                           json={"session_id": sid, "message": "stream"})
    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    assert response.data.count(b"reply:stream") == 1
    assert b'"type": "done"' in response.data
    assert b'"type": "done", "content"' not in response.data
    messages = client.get(f"/api/sessions/{sid}/messages", headers=alice_headers).get_json()["messages"]
    assert messages[0]["content"] == "stream"


def test_register_persists_plaintext_and_login_logout_lifecycle(app, client):
    response = client.post(
        "/api/auth/register", json={"username": "carol", "password": "plain-secret"}
    )
    assert response.status_code == 201
    first_token = response.get_json()["token"]
    users_path = app.config["RUNTIME_DIR"] + "/auth/users.json"
    assert json.loads(Path(users_path).read_text(encoding="utf-8"))["users"]["carol"] == {
        "password": "plain-secret"
    }
    assert client.post(
        "/api/auth/login", json={"username": "carol", "password": "wrong"}
    ).status_code == 401
    login = client.post(
        "/api/auth/login", json={"username": "carol", "password": "plain-secret"}
    )
    assert login.status_code == 200
    headers = {"Authorization": f"Bearer {login.get_json()['token']}"}
    assert client.get("/api/auth/me", headers=headers).get_json()["username"] == "carol"
    assert client.post("/api/auth/logout", headers=headers).status_code == 200
    assert client.get("/api/auth/me", headers=headers).status_code == 401
    assert first_token != login.get_json()["token"]


def test_registration_validates_input_and_rejects_duplicate(client):
    assert client.post(
        "/api/auth/register", json={"username": "../bad", "password": "x"}
    ).status_code == 400
    assert client.post(
        "/api/auth/register", json={"username": "valid", "password": ""}
    ).status_code == 400
    assert client.post(
        "/api/auth/register", json={"username": "valid", "password": "x"}
    ).status_code == 201
    assert client.post(
        "/api/auth/register", json={"username": "valid", "password": "x"}
    ).status_code == 409
