"""
RAG Agent 入口
交互式命令行界面
"""
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LOG_LEVEL, DOC_DIR, BM25_INDEX_PATH
from init_db import load_bm25_index
from test_evaluator import run_test_mode
from chunker import chunk_all_documents
from embedding import EmbeddingService
from vector_store import VectorStore
from retriever import HybridRetriever, BM25Index
from memory import LongTermMemory, MemoryManager
from agent import LegalAgent

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_bm25_or_rebuild(vector_store=None):
    """加载 BM25 索引，如不存在则重建"""
    if os.path.exists(BM25_INDEX_PATH):
        try:
            bm25, chunks, _manifest = load_bm25_index(BM25_INDEX_PATH, vector_store)
            logger.debug("BM25 索引加载完成")
            return bm25, chunks
        except Exception as e:
            logger.warning(f"BM25 加载失败，将重建: {e}")

    # 重建
    logger.info("重建 BM25 索引...")
    chunks = chunk_all_documents(DOC_DIR)
    bm25\
        =BM25Index()
    bm25.build(chunks)
    return bm25, chunks


def _read_query(session) -> str:
    """读取用户输入，粘贴多行时合并展示并让用户确认后再发送"""
    prompt = f"\n🙋 [{session[:8]}] 您: "
    first = input(prompt).strip()
    if not first:
        return first

    # 尝试检测 stdin 中是否还有更多数据（粘贴多行场景）
    more = _drain_stdin(timeout=0.25)
    if not more:
        return first

    combined = "\n".join([first] + more)
    print(f"\n📋 检测到 {len(more)+1} 行粘贴内容：")
    print("─" * 50)
    print(combined)
    print("─" * 50)
    print("⏎ 回车 → 发送  |  输入新内容 → 替换  |  q → 取消")
    edit = input().strip()
    if edit.lower() == "q":
        return ""
    if edit:
        return edit
    combined = combined[:20000]
    return combined


def _drain_stdin(timeout=0.25):
    """检测 stdin 缓冲区中是否还有等待读取的行（粘贴残留）。

    使用平台原生的非阻塞 API，避免在 Windows 上因
    ``BufferedReader.peek()`` 阻塞导致死锁（详见下文）。

    死锁背景：
    ``BufferedReader.peek()`` 在持有内部锁的情况下调用底层
    ``RawIO.read()``；而 Windows 控制台的 ``ReadConsoleW`` 会阻塞
    直到有输入，且不释放锁。此时主线程若调用 ``input()``（即
    ``readline()``），会试图获取同一把锁，造成永久的线程死锁。
    """
    import sys, time, os

    result = []
    deadline = time.monotonic() + timeout

    if os.name == "nt":
        try:
            import msvcrt
        except ImportError:
            return result
        while time.monotonic() < deadline:
            if msvcrt.kbhit():
                try:
                    line = sys.stdin.readline().strip()
                except (OSError, ValueError):
                    break
                if not line:
                    break
                result.append(line)
            time.sleep(0.02)
        return result

    # ── Unix：select 非阻塞轮询 ──
    import select as _select
    while time.monotonic() < deadline:
        r, _, _ = _select.select([sys.stdin], [], [], 0)
        if r:
            try:
                line = sys.stdin.readline().strip()
            except (OSError, ValueError):
                break
            if not line:
                break
            result.append(line)
        else:
            time.sleep(0.02)
    return result


def main():
    print("\n" + "=" * 60)
    print("     ⚖️  法律案例 · 智能知识库 Agent")
    print("=" * 60)

    # 1. 初始化各模块
    print("\n[初始化中...]\n")

    # Embedding（使用千问3真实 embedding）
    embed = EmbeddingService(mock=False)
    print("  ✓ Embedding 服务")

    # 向量库
    vs = VectorStore(embed_service=embed)
    print(f"  ✓ 向量数据库 (已有 {vs.count()} 条)")

    # BM25 + chunks
    bm25, chunks = load_bm25_or_rebuild(vs)
    print(f"  ✓ BM25 索引 (基于 {len(chunks) if chunks else 0} chunks)")

    # 混合检索器
    retriever = HybridRetriever(
        vector_store=vs,
        embed_service=embed,
        bm25_index=bm25,
    )
    print("  ✓ 混合检索器")

    # 记忆管理（默认会话）
    long_mem = LongTermMemory()
    memory = MemoryManager.create_session(long_term=long_mem)
    current_session = memory.session_id
    print(f"  ✓ 记忆管理器 (会话: {current_session})")

    # Agent
    agent = LegalAgent(
        retriever=retriever,
        memory=memory,
        llm_call_func=None,  # 使用默认 HTTP 调用，需配置 LLM_API_URL
    )
    print("  ✓ Agent 引擎")

    print("\n" + "=" * 60)
    print("输入您的问题（输入 q 退出，输入 clear 清空记忆）")
    print("可用命令: /session 查看会话, /session new 新建会话, /test 检索评测")
    print("=" * 60)

    # 交互循环
    while True:
        try:
            query = _read_query(current_session)
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not query:
            continue
        if query.lower() in ("q", "quit", "exit"):
            print("再见！")
            break
        if query.lower() == "clear":
            memory.session.clear()
            print("短期记忆已清空")
            continue

        if query.lower().startswith("/session"):
            parts = query.split()
            if len(parts) >= 2 and parts[1] == "new":
                memory = MemoryManager.create_session(long_term=long_mem)
                agent.memory = memory
                current_session = memory.session_id
                print(f"🆕 新建会话: {current_session}")
            else:
                print(f"当前会话: {current_session}")
            continue

        if query.lower() in ("/test", "/测试", "test"):
            run_test_mode(retriever, None)
            continue

        # 回答
        print("\n🤖 Agent: ", end="", flush=True)
        response = agent.answer(query)
        print(response)
        print()


if __name__ == "__main__":
    main()
