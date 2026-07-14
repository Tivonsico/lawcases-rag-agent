"""
Agent 主模块
- 每次对话先检索
- 前三轮对话按 HUMAN: / AI: 格式传入
- 检索到 → RAG 回答 + 出处
- 没检索到 → 明确标注为模型生成的一般信息（不伪称网络来源）
"""
import json
import re
from typing import Dict, List, Optional

try:
    from .config import LLM_API_URL, LLM_API_KEY, LLM_MODEL
    from .retriever import HybridRetriever
    from .memory import MemoryManager
    from .intent import IntentProcessor
except ImportError:  # pragma: no cover - direct script compatibility
    from config import LLM_API_URL, LLM_API_KEY, LLM_MODEL
    from retriever import HybridRetriever
    from memory import MemoryManager
    from intent import IntentProcessor


SYSTEM_PROMPT = """你是一个中国法律案例客服 Agent（法律咨询助手）。
你必须用专业、严肃、准确的口吻回答用户关于法律案例的问题。

## 回答规范
1. 如果下面提供了"检索到的知识库内容"，优先基于它回答，并带出处：
   来源文档《指导性案例XX号》
2. 如果检索结果与问题不相关或没有检索到内容，可以给出一般性法律信息，
   但不得声称该内容来源于网络、具体法条或案例，除非上下文确实提供了该来源。
3. 即使检索到了相关案例，当你的分析超出了检索内容本身的范围（如推断结论、
   建议具体行动等），回答末尾也请加上这句免责声明：
   *除明确标注的知识库案例引文外，其余内容由模型生成，未联网核验。*
4. 严禁代替司法机关下定论。不得说"构成正当防卫"、"构成犯罪"等确定性判断，
   应表述为"可能符合"、"参考类似案例"、"存在被认定为的可能性"等建议性口吻。
   最终结论应由公安机关、法院等有权机关作出。
5. 保持专业严肃的语气，引用法条和案例编号时要准确
6. 如果检索到多篇相关案例，请进行综合分析：
   - 对比各案例的裁判要点，指出一致之处和差异
   - 优先引用最直接相关的案例进行回答
   - 多篇案例可互为补充，共同支撑你的分析
   - 有三篇或以上相关文档时，至少引用三篇不同文档；只有一至两篇相关时引用全部相关文档
   - 不得为了凑足三篇而引用与问题关系不大的文档"""


class LegalAgent:
    """
    法律案例客服 Agent
    每次检索 + 携带对话历史 → LLM 回答
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        memory: MemoryManager,
        llm_call_func=None,
        llm_api_url: str = None,
        llm_api_key: str = None,
        llm_model: str = None,
        intent_processor: IntentProcessor = None,
    ):
        self.retriever = retriever
        self.memory = memory
        self.llm_call = llm_call_func or self._default_llm_call
        self.llm_api_url = llm_api_url or LLM_API_URL
        self.llm_api_key = llm_api_key or LLM_API_KEY
        self.llm_model = llm_model or LLM_MODEL
        self._prev_docs = []
        self.intent_processor = intent_processor or IntentProcessor(llm_call_func=self.llm_call)

    def _default_llm_call(self, prompt: str) -> str:
        return self._call_messages([{"role": "user", "content": prompt}])

    def _call_messages_stream(self, messages: list):
        """流式调用 LLM，逐 chunk yield"""
        if not self.llm_api_url:
            yield "[LLM 未配置：请设置 .env 中的 LLM_API_URL 和 LLM_API_KEY]"
            return

        import requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_api_key}",
        }
        payload = {
            "model": self.llm_model or "deepseek-chat",
            "messages": messages,
            "temperature": 0.3,
            "stream": True,
        }

        try:
            resp = requests.post(
                self.llm_api_url, headers=headers, json=payload, timeout=120, stream=True
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    line = line.decode("utf-8", errors="ignore").strip()
                    if not line or line.startswith(":") or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"[LLM 调用失败: {e}]"

    def _call_messages(self, messages: list) -> str:
        if not self.llm_api_url:
            return "[LLM 未配置：请设置 .env 中的 LLM_API_URL 和 LLM_API_KEY]"

        import requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_api_key}",
        }
        payload = {
            "model": self.llm_model or "deepseek-chat",
            "messages": messages,
            "temperature": 0.3,
        }

        try:
            resp = requests.post(
                self.llm_api_url, headers=headers, json=payload, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[LLM 调用失败: {e}]"

    def _build_history_messages(self) -> List[Dict]:
        """
        构建对话历史消息列表：
        委托 memory.get_messages() 返回 summary + buffer 格式
        """
        return self.memory.get_messages()

    def _build_content(self, user_query: str, docs: List[Dict]) -> str:
        """
        构建本轮 user 消息内容：按文档分组展示检索结果 + 用户问题
        docs: [
            {
                "doc_name": "指导性案例192号 李开祥...",
                "display_name": "指导性案例192号",
                "text": "该文档所有匹配 chunk 的合并文本",
                ...
            }
        ]
        """
        parts = []

        parts.append("## 检索到的知识库内容")
        if docs:
            for i, doc in enumerate(docs, 1):
                doc_name = doc.get("display_name") or doc.get("doc_name", "")
                text = doc.get("text", "")
                parts.append(f"\n--- 结果{i} 来源《{doc_name}》---")
                parts.append(text)
            parts.append(
                f"\n## 引用覆盖要求\n"
                f"本次提供 {len(docs)} 篇候选文档。请先判断每篇文档与当前问题是否实质相关："
                "若相关文档不少于三篇，回答必须引用至少三篇不同的相关文档并综合比较；"
                "若只有一至两篇相关文档，则引用全部相关文档；若均不相关，则不要引用。"
                "不得为了达到数量要求引用低相关文档，也不得引用候选列表之外的文档。"
            )
        else:
            parts.append("（未检索到相关内容，你可以用自己的法律知识回答）")

        # 长期画像
        long_text = self.memory.get_long_text()
        if long_text:
            parts.append(f"\n## 用户长期画像\n{long_text}")

        parts.append(f"\n## 用户当前问题\n{user_query}")
        return "\n".join(parts)

    def _verify_source(self, response: str, docs: List[Dict]) -> str:
        """Remove unsupported citations instead of rewriting them to another case."""
        if "来源文档《" not in response:
            return response

        # 从 doc dict 中提取文档名
        actual_names = []
        for d in docs:
            name = d.get("display_name") or d.get("doc_name", "")
            if name:
                actual_names.append(name)

        pattern = re.compile(r"来源文档《(.+?)》")
        matches = pattern.findall(response)
        for cited in matches:
            found = False
            for actual in actual_names:
                if cited in actual or actual in cited:
                    found = True
                    break
            if not found:
                response = response.replace(
                    f"来源文档《{cited}》",
                    "（该出处未在本次检索结果中得到验证）"
                )
        return response

    @staticmethod
    def _append_disclosure(response: str, docs: List[Dict]) -> str:
        marker = "未联网核验"
        if marker in response:
            return response
        if docs:
            note = "*案例引文仅来自本次知识库检索；其余内容由模型生成，未联网核验。*"
        else:
            note = "*知识库未检索到可引用案例；以下为模型生成的一般信息，未联网核验。*"
        return f"{response.rstrip()}\n\n{note}"

    def answer(self, user_query: str) -> str:
        """兼容非流式调用；实际生成统一走流式管线。"""
        return "".join(
            chunk for chunk in self.answer_stream(user_query) if chunk != "[DONE]"
        )

    def answer_stream(self, user_query: str):
        """
        流式回答生成器
        先检索+构建 prompt → 流式调 LLM → 追加披露说明并记忆。
        持续 yield 增量文本，最后单独 yield ``[DONE]`` 作为完成标记。
        """
        # 1. 意图分析 + 关键词扩展
        intent_result = self.intent_processor.process(user_query)
        expanded_keywords = intent_result.get("expanded_keywords", [])
        if not expanded_keywords:
            expanded_keywords = intent_result.get("keywords", [])

        # 2. 检索
        retrieval = self.retriever.search(
            query=user_query,
            normalized_query=intent_result.get("normalized_query", ""),
            exact_terms=intent_result.get("exact_terms", ()),
        )
        docs = retrieval.documents

        all_docs = list(docs)
        if self._prev_docs and len(user_query) <= 15:
            prev_names = {d.get("doc_name", "") for d in all_docs}
            for pd in self._prev_docs:
                pn = pd.get("doc_name", "")
                if pn and pn not in prev_names:
                    all_docs.insert(0, pd)
                    prev_names.add(pn)
        self._prev_docs = docs

        # 3. 构建 messages
        content = self._build_content(user_query, all_docs)
        history = self._build_history_messages()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": content})

        # 4. 流式调 LLM，收集完整响应
        full_response = ""
        for chunk in self._call_messages_stream(messages):
            if chunk.startswith("[LLM"):
                full_response = self._fallback_answer(user_query, docs)
                yield full_response
                break
            full_response += chunk
            yield chunk  # 实时推给前端

        # 5. 已发送的增量文本不能在结尾被整篇替换。流式链路通过 prompt
        # 限制可引用来源，并把披露说明作为最后一个增量块发送。
        final_response = self._append_disclosure(full_response, all_docs)
        disclosure = final_response[len(full_response):]
        if disclosure:
            yield disclosure
        full_response = final_response

        # 6. 记忆
        self.memory.add_dialogue(user_query, full_response)
        self._compress_memory()

        # 完成事件只传控制信号，不重复发送整篇正文。
        yield "[DONE]"

    def _compress_memory(self):
        """buffer 字符数超阈值 → 旧轮次被 LLM 压缩为 running_summary"""
        if self.memory.need_summarize() and self.llm_call:
            try:
                summary_prompt = self.memory.session.summary_prompt
                if not summary_prompt:
                    return
                summary_raw = self.llm_call(summary_prompt)
                if summary_raw and not summary_raw.startswith("[LLM"):
                    new_summary = self.memory.session.parse_summary(summary_raw)
                    old_n = len(self.memory.get_old_rounds())
                    self.memory.update_summary(new_summary)
                    self.memory.pop_old_rounds(old_n)
            except Exception as e:
                print(f"[WARN] 记忆压缩失败: {e}")

        # 每 5 轮提取一次用户画像
        buffer_len = len(self.memory.session.buffer)
        if buffer_len > 0 and buffer_len % 5 == 0 and self.llm_call:
            try:
                profile_prompt = self.memory.long.extract_profile_prompt(
                    self.memory.session.buffer
                )
                profile_raw = self.llm_call(profile_prompt)
                raw = profile_raw.strip().replace("```json", "").replace("```", "").strip()
                profile = json.loads(raw)
                self.memory.long.update_profile(profile)
            except Exception as e:
                print(f"[WARN] 用户画像提取失败: {e}")

    def _fallback_answer(self, query: str, docs: List[Dict]) -> str:
        """LLM 不可用时的回退"""
        if not docs:
            return "抱歉，我未在知识库中查到相关法律案例。"
        names = [d.get("display_name", "未知") for d in docs[:2]]
        return (
            f"客户您好，您查询的「{query}」相关案例：\n"
            f"{docs[0].get('text', '')[:200]}...\n"
            f"来源文档：{'、'.join(names)}\n"
            f"（系统提示：LLM 未配置，此为回退回答）"
        )
