import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeAgent:
    def __init__(self, memory):
        self.memory = memory

    def answer(self, message):
        reply = f"reply:{message}"
        self.memory.add_dialogue(message, reply)
        return reply

    def answer_stream(self, message):
        reply = f"reply:{message}"
        yield reply
        self.memory.add_dialogue(message, reply)
        yield "[DONE]"


@pytest.fixture
def app(tmp_path):
    from rag_agent.api_server import create_app
    return create_app({
        "TESTING": True,
        "RUNTIME_DIR": str(tmp_path / "runtime"),
        "LONG_TERM_DB": str(tmp_path / "runtime" / "memory.db"),
        "RATE_LIMIT": 100,
        "MAX_MESSAGE_CHARS": 20,
    }, {"agent_factory": FakeAgent, "vector_count": 7})


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def alice_headers(client):
    response = client.post(
        "/api/auth/register", json={"username": "alice", "password": "alice-pass"}
    )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.get_json()['token']}"}
