"""
文档切分模块
- 先按段落拆分（双换行/标题）
- 每段落按句拆分（。！？）
- 以 ~200 字为基准，遇段落边界停
- 每个 chunk 带前一句作为上下文
- 打标签：文档名 / chunk序号 / 核心关键词
"""
import os
import re
import json
import hashlib
import logging
from typing import List, Dict, Optional

from config import DOC_DIR, CHUNK_SIZE, CHUNK_MIN

SCHEMA_VERSION = "2"
logger = logging.getLogger(__name__)


def _first_match(pattern: str, text: str, default: str = "unknown") -> str:
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1).strip() if match else default


def extract_case_metadata(filename: str, content: str) -> Dict[str, object]:
    """规则抽取可验证元数据；无法确认的字段不猜测。"""
    stem = os.path.splitext(os.path.basename(filename))[0]
    source_id = _first_match(r"\((FBM-[^)]+)\)", stem)
    match = re.search(r"((?:指导性?案例\s*\d+\s*号)|(?:检例第\s*\d+\s*号))", stem)
    case_number = re.sub(r"\s+", "", match.group(1)) if match else "unknown"
    case_type = ("最高法指导案例" if case_number.startswith(("指导案例", "指导性案例"))
                 else "最高检指导案例" if case_number.startswith("检例") else "unknown")
    source = source_id if source_id != "unknown" else hashlib.sha256(stem.encode()).hexdigest()[:20]
    laws = sorted(set(re.findall(r"《[^》]{2,40}》(?:第[一二三四五六七八九十百千万零〇\d]+条)?", content)))
    years = re.findall(r"(?:19|20)\d{2}", content)
    category = next((value for needle, value in (("公益诉讼", "公益诉讼"), ("执行", "执行"),
                    ("行政", "行政"), ("刑事", "刑事"), ("民事", "民事"))
                    if needle in stem or needle in content[:2000]), "unknown")
    return {
        "case_id": f"case:{source.lower()}", "case_type": case_type,
        "case_number": case_number, "source_id": source_id,
        "cause": _first_match(r"案由[:：]\s*([^\n]+)", content), "category": category,
        "authority": _first_match(r"([\u4e00-\u9fff]{2,30}(?:人民法院|人民检察院))", content),
        "trial_level": _first_match(r"审级[:：]\s*([^\n]+)", content),
        "publication_year": years[0] if years else "unknown",
        "judgment_year": years[-1] if years else "unknown",
        "legal_provisions": laws[:20], "charge": _first_match(r"罪名[:：]\s*([^\n]+)", content),
        "entities": [], "behaviors": [], "dispute_focus": "unknown",
        "validity_status": "invalid" if "【失效】" in stem else "unknown",
        "schema_version": SCHEMA_VERSION,
    }


def _section_type(text: str) -> str:
    for heading, value in {"裁判要点": "decision_summary", "相关法条": "legal_basis",
                           "基本案情": "facts", "裁判结果": "result", "裁判理由": "reasoning"}.items():
        if heading in text[:40]:
            return value
    return "body"


def load_documents(doc_dir: str = None) -> List[Dict[str, str]]:
    """加载所有文档，返回 [{filepath, filename, content}]"""
    docs = []
    target = doc_dir or DOC_DIR
    if not os.path.isdir(target):
        logger.warning("文档目录不存在: %s", target)
        return docs
    for fname in sorted(os.listdir(target)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(target, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.warning("读取失败 %s: %s", fname, e)
            continue
        docs.append({"filepath": fpath, "filename": fname, "content": content})
    logger.debug("共加载 %s 篇文档", len(docs))
    return docs


def _parse_paragraphs(text: str) -> List[str]:
    """将文本拆为段落，保留段落间结构"""
    # 去掉标题行和来源行
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            cleaned.append("")  # 保留空行作为段落分隔
        else:
            cleaned.append(line)
    # 按双换行拆段
    raw_paras = re.split(r"\n\s*\n", "\n".join(cleaned))
    paras = [p.strip() for p in raw_paras if p.strip()]
    return paras


def _parse_sentences(paragraph: str) -> List[str]:
    """将段落拆为句子，保留  。！？； 作为边界"""
    # 先按标点分割，保留分隔符
    parts = re.split(r"(?<=[。！？；])", paragraph)
    sents = [p.strip() for p in parts if p.strip()]
    # 对于长句无标点或标点很少的，按逗号进一步拆分
    final = []
    for s in sents:
        if len(s) > 150 and re.search(r"[，、]", s):
            sub = re.split(r"(?<=[，、])", s)
            final.extend([x.strip() for x in sub if x.strip()])
        else:
            final.append(s)
    return final


def _extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """简单关键词提取：按词频统计（中文2-gram + 高频词）"""
    # 简单的基于字频的关键词——实际项目中可替换为 jieba + tfidf
    from collections import Counter
    # 去除非中文/字母/数字
    clean = re.sub(r"[^一-龥a-zA-Z0-9]", "", text)
    if len(clean) < 4:
        return list(clean)[:top_n]
    # 提取含2字以上的词(简单的字频滑动)
    words = []
    for i in range(len(clean) - 1):
        bigram = clean[i:i+2]
        if all('一' <= c <= '龥' for c in bigram):
            words.append(bigram)
    if len(words) < 3:
        words = [c for c in clean if '一' <= c <= '龥']
    counter = Counter(words)
    return [w for w, _ in counter.most_common(top_n)]


def chunk_document(
    filename: str,
    content: str,
    chunk_size: int = None,
    chunk_min: int = None
) -> List[Dict]:
    """
    对一篇文档进行语义切分
    返回: [{tag, text, doc_name, chunk_index, keywords, prev_context}]
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if chunk_min is None:
        chunk_min = CHUNK_MIN
    case_metadata = extract_case_metadata(filename, content)

    # 提取文档标题（第一行或文件名）
    doc_name = filename.replace(".txt", "")
    # 尝试从内容中提取真实标题
    first_line = content.strip().split("\n")[0] if content.strip() else ""
    if first_line.startswith("标题:"):
        title = first_line.replace("标题:", "").strip()
        if title:
            doc_name = title

    # 拆段落 -> 句子
    paragraphs = _parse_paragraphs(content)
    all_sentences = []
    for para in paragraphs:
        sents = _parse_sentences(para)
        if not sents:
            continue
        all_sentences.append(sents)

    # 将句子展平，同时记录段落边界索引
    flat_sents = []      # 所有句子列表
    para_boundary = []   # 每个段落结束的索引(不含)
    for sents in all_sentences:
        for s in sents:
            flat_sents.append(s)
        para_boundary.append(len(flat_sents))

    # ── 核心分块逻辑 ──
    chunks = []
    i = 0
    sent_count = len(flat_sents)
    chunk_idx = 0
    prev_sentence = ""  # 前一句(跨chunk的上下文)

    while i < sent_count:
        current_chunk = []
        char_count = 0
        # 记录当前chunk的第一句索引，用于判断段落边界
        start_i = i

        while i < sent_count:
            sent = flat_sents[i]
            sent_len = len(sent)

            # 如果加入这句会超过chunk_size，检查是否已在段落边界
            if char_count + sent_len > chunk_size and char_count >= chunk_min:
                # 检查 i 是否是某段落的开始
                is_para_start = any(i == b for b in para_boundary)
                if is_para_start:
                    break  # 段落边界，可以切
                # 也检查当前句是否是独立的短句（属于自然停顿）
                if sent_len > chunk_size * 0.6:
                    # 句子本身就很大，强行切
                    if char_count >= chunk_min:
                        break

            current_chunk.append(sent)
            char_count += sent_len
            i += 1

            # 如果到了段落边界，强制停
            if any(i == b for b in para_boundary):
                break

            # 如果已经远超chunk_size，停
            if char_count >= chunk_size * 1.5:
                break

        if not current_chunk:
            i += 1
            continue

        chunk_text = "".join(current_chunk)
        if len(chunk_text) < chunk_min and i < sent_count:
            # 太短了且后面还有内容，合并到下一轮
            # 但如果前一句有上下文，仍然保留
            if chunks and len(chunk_text) < chunk_min // 2:
                # 并入上一个chunk
                last = chunks[-1]
                last["text"] += chunk_text
                last["keywords"] = _extract_keywords(last["text"])
                continue

        # 构建带上下文的文本
        context_text = chunk_text
        if prev_sentence:
            context_text = prev_sentence + "\n" + chunk_text

        # 如果chunk以引文或列表结尾，适当处理
        keywords = _extract_keywords(chunk_text)

        tag = f"{doc_name} | chunk{chunk_idx+1} | {' '.join(keywords[:5])}"

        chunks.append({
            "tag": tag,
            "text": context_text,
            "doc_name": doc_name,
            "chunk_index": chunk_idx,
            "keywords": keywords,
            "prev_context": prev_sentence,
            "chunk_id": f"{case_metadata['case_id']}:chunk:{chunk_idx}",
            "section_type": _section_type(chunk_text),
            **case_metadata,
        })

        # 更新前一句为当前chunk的最后一句，作为下一chunk的上下文
        prev_sentence = current_chunk[-1] if current_chunk else ""
        chunk_idx += 1

    return chunks


def chunk_all_documents(doc_dir: str = None) -> List[Dict]:
    """对所有文档切分，返回所有chunks"""
    docs = load_documents(doc_dir)
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc["filename"], doc["content"])
        all_chunks.extend(chunks)
        logger.debug("%s: %s chunks", doc["filename"], len(chunks))
    logger.debug("切分完成: %s 篇文档, %s 个 chunks", len(docs), len(all_chunks))
    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all_documents()
    if chunks:
        # 打印前3个作为预览
        for c in chunks[:3]:
            print(f"\n{'='*60}")
            print(f"标签: {c['tag']}")
            print(f"文本({len(c['text'])}字): {c['text'][:100]}...")
        print(f"\n总计 {len(chunks)} chunks")
