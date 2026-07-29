"""Railway WSGI entry point — auto-initializes index on first deploy."""
import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_agent"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 检测索引是否存在，不存在则从文档重建
from config import DOC_DIR, BM25_INDEX_PATH, CHROMA_PERSIST_DIR  # noqa: E402

_index_missing = not os.path.isfile(BM25_INDEX_PATH) or not os.path.isdir(CHROMA_PERSIST_DIR)

if _index_missing:
    logger.info("知识库索引未找到，开始从文档重建（首次部署较慢，约 5-15 分钟）...")
    from init_db import build_indexes
    try:
        result = build_indexes()
        logger.info("知识库重建完成：%s chunks, %s 向量", len(result["chunks"]), result["vector_store"].count())
    except Exception as exc:
        logger.error("知识库重建失败：%s", exc)
        logger.error("请检查 .env 中 EMBEDDING_API_KEY 是否正确配置")
        raise
else:
    logger.info("知识库索引已就绪")

# 创建 Flask 应用
from api_server import create_app  # noqa: E402
app = create_app()

if __name__ == "__main__":
    import os
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "5000")))