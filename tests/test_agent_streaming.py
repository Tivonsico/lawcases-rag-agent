from types import SimpleNamespace

from rag_agent.agent import LegalAgent


class FakeMemory:
    def __init__(self):
        self.dialogues = []
        self.session = SimpleNamespace(buffer=[])

    def get_messages(self):
        return []

    def get_long_text(self):
        return ""

    def add_dialogue(self, user, agent):
        self.dialogues.append((user, agent))

    def need_summarize(self):
        return False


class FakeIntent:
    def process(self, _query):
        return {}


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def search(self, **_kwargs):
        return SimpleNamespace(documents=self.documents)


def make_agent(doc_count=3):
    docs = [
        {"doc_name": f"案例{i}", "display_name": f"案例{i}", "text": f"内容{i}"}
        for i in range(1, doc_count + 1)
    ]
    memory = FakeMemory()
    agent = LegalAgent(FakeRetriever(docs), memory, intent_processor=FakeIntent())
    captured = {}

    def stream(messages):
        captured["messages"] = messages
        yield "第一段"
        yield "第二段"

    agent._call_messages_stream = stream
    return agent, memory, captured


def test_answer_stream_is_incremental_and_done_does_not_repeat_body():
    agent, memory, _captured = make_agent()

    chunks = list(agent.answer_stream("如何处理"))

    assert chunks[:2] == ["第一段", "第二段"]
    assert chunks[-1] == "[DONE]"
    assert "第一段第二段" not in chunks[2:]
    assert memory.dialogues[0][1].startswith("第一段第二段")


def test_prompt_requires_three_relevant_sources_without_padding_irrelevant_ones():
    agent, _memory, captured = make_agent(doc_count=5)

    list(agent.answer_stream("如何处理"))

    prompt = captured["messages"][-1]["content"]
    assert "本次提供 5 篇候选文档" in prompt
    assert "相关文档不少于三篇" in prompt
    assert "至少三篇不同的相关文档" in prompt
    assert "不得为了达到数量要求引用低相关文档" in prompt


def test_non_streaming_compatibility_uses_same_stream_pipeline():
    agent, memory, _captured = make_agent()

    response = agent.answer("如何处理")

    assert response.startswith("第一段第二段")
    assert "[DONE]" not in response
    assert len(memory.dialogues) == 1
