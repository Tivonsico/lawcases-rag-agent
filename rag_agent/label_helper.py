"""
辅助工具：搜索文档（文件名 + 语义向量搜索）
用法:
  python label_helper.py <关键词>               # 向量搜索（语义相似）
  python label_helper.py <关键词> --filename     # 仅按文件名搜
  python label_helper.py <关键词> --both         # 两者都搜
  python label_helper.py                         # 交互模式
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DOC_DIR
from embedding import EmbeddingService
from vector_store import VectorStore


# ── 文件列表 ──
files = sorted([f.replace('.txt', '') for f in os.listdir(DOC_DIR) if f.endswith('.txt')])


def search_filename(keyword: str):
    """按文件名搜索"""
    matches = []
    for f in files:
        display = f.split('(')[0].strip()
        if keyword.lower() in f.lower() or keyword in display:
            matches.append((f, display))
    return matches


def search_vector(embed: EmbeddingService, vs: VectorStore, keyword: str):
    """
    向量语义搜索：
    找出向量库中与关键词最相似的 chunks，按文档聚合返回。
    返回 [(完整名, 显示名, 匹配chunks列表)]
    """
    query_vec = embed.embed_one(keyword)
    results = vs.search(query_text=keyword, query_embedding=query_vec, top_k=5)

    # 按 doc_name 聚合
    doc_map = {}
    for r in results:
        doc_name = r["metadata"].get("doc_name", "")
        if not doc_name:
            continue
        if doc_name not in doc_map:
            display = doc_name.split('(')[0].strip()
            doc_map[doc_name] = {"display": display, "chunks": []}
        doc_map[doc_name]["chunks"].append({
            "text": r["text"][:120].replace('\n', ' '),
            "score": r["score"],
        })

    matches = []
    for doc_name, info in doc_map.items():
        matches.append((
            doc_name,
            info["display"],
            info["chunks"],
        ))
    # 按最高分排序
    matches.sort(key=lambda x: max(c["score"] for c in x[2]), reverse=True)
    return matches


# ── 打印 ──

def print_filename_results(matches, keyword):
    if not matches:
        print(f"  未找到文件名包含「{keyword}」的文档\n")
    else:
        print(f"  找到 {len(matches)} 个匹配文档:\n")
        for i, (full, display) in enumerate(matches, 1):
            print(f"    {i:3d}. {display}")
            print(f"        完整名: {full}")
        print()


def print_vector_results(matches, keyword):
    if not matches:
        print(f"  未找到与「{keyword}」语义相关的文档\n")
    else:
        print(f"  找到 {len(matches)} 个语义相关文档:\n")
        for i, (full, display, chunks) in enumerate(matches, 1):
            best_score = max(c["score"] for c in chunks)
            print(f"    {i:3d}. {display}  (相似度: {best_score:.3f})")
            print(f"        完整名: {full}")
            for c in chunks:
                print(f"        [{c['score']:.3f}] …{c['text']}…")
            print()


# ── 交互循环 ──

def interactive_loop():
    embed, vs = _init_vector_engine()

    print("=" * 55)
    print("  文档搜索工具（输入 q 退出）")
    print("=" * 55)
    print("  直接输入关键词 → 向量语义搜索")
    print("  #关键词         → 仅搜文件名（如 #合同）")
    print("  ##关键词        → 文件名 + 向量都搜")
    print("=" * 55)
    print()

    while True:
        try:
            raw = input("🔍 关键词: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见！")
            break

        if not raw:
            continue
        if raw.lower() in ("q", "quit", "exit"):
            print("再见！")
            break

        # 判断搜索模式
        if raw.startswith("##"):
            keyword = raw[2:].strip()
            mode = "both"
        elif raw.startswith("#"):
            keyword = raw[1:].strip()
            mode = "filename"
        else:
            keyword = raw
            mode = "vector"

        if not keyword:
            print("  请输入关键词\n")
            continue

        if mode in ("filename", "both"):
            matches_fn = search_filename(keyword)
            print_filename_results(matches_fn, keyword)

        if mode in ("vector", "both"):
            matches_v = search_vector(embed, vs, keyword)
            print_vector_results(matches_v, keyword)


# ── 初始化向量引擎 ──

def _init_vector_engine():
    embed = EmbeddingService(mock=False)
    vs = VectorStore(embed_service=embed)
    return embed, vs


# ── 主入口 ──

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        keyword = sys.argv[1]
        flags = {a.lower() for a in sys.argv[2:]}

        use_fn = flags & {"--filename", "--both", "-b", "-f"}
        use_vs = not flags or flags & {"--vector", "--both", "-b", "-v"}

        if use_fn:
            matches = search_filename(keyword)
            print_filename_results(matches, keyword)
        if use_vs:
            embed, vs = _init_vector_engine()
            matches = search_vector(embed, vs, keyword)
            print_vector_results(matches, keyword)
        if not use_fn and not use_vs:
            print("用法: python label_helper.py <关键词> [--vector | --filename | --both]")
    else:
        interactive_loop()
