"""Railway WSGI entry point — auto-initializes index on first deploy."""
import os, sys, logging, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_agent"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 检测索引是否存在，不存在则从文档重建
from config import DOC_DIR, BM25_INDEX_PATH, CHROMA_PERSIST_DIR  # noqa: E402

# ── 文件锁，防止多 worker 同时重建索引 ──
LOCK_PATH = "/tmp/legal-rag-rebuild.lock"

def acquire_lock(timeout=300):
    """尝试获取重建锁，避免多 worker 竞争"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(3)
    logger.warning("等待重建锁超时（其他 worker 可能已卡死），强制覆盖")
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except:
        return False

def release_lock():
    try:
        os.unlink(LOCK_PATH)
    except:
        pass

_index_missing = not os.path.isfile(BM25_INDEX_PATH)

if _index_missing:
    if acquire_lock():
        try:
            logger.info("知识库索引未找到，开始从文档重建（首次部署较慢，约 5-15 分钟）...")
            from vector_store import VectorStore
            from embedding import EmbeddingService
            from init_db import build_indexes

            # 清除可能的部分数据（上次崩溃遗留的）
            temp_vs = VectorStore(embed_service=EmbeddingService(mock=False))
            if temp_vs.count() > 0:
                logger.info("检测到不完整的向量数据，先清空再重建...")
                temp_vs.clear()
            del temp_vs

            result = build_indexes()
            logger.info("知识库重建完成：%s chunks, %s 向量",
                        len(result["chunks"]), result["vector_store"].count())
        except Exception as exc:
            logger.error("知识库重建失败：%s", exc)
            logger.error("请检查 .env 中 EMBEDDING_API_KEY 是否正确配置")
            release_lock()
            raise
        release_lock()
    else:
        # 无法获取锁，等持有锁的 worker 完成后自动 recovery 重启
        logger.info("索引正在由其他 worker 重建，等待后重试...")
        time.sleep(5)
else:
    logger.info("知识库索引已就绪")

# 创建 Flask 应用
from api_server import create_app  # noqa: E402
app = create_app()

if __name__ == "__main__":
    import os
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "5000")))