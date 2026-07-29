"""WSGI entry point for Gunicorn.
Index MUST be pre-built before starting Gunicorn (see deploy.py).
This module assumes BM25 index + Chroma DB already exist.
"""
import os, sys, logging

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_agent"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from config import BM25_INDEX_PATH, CHROMA_PERSIST_DIR

# Fail fast: if index is missing, tell operator to rebuild first
bm25_exists = os.path.isfile(BM25_INDEX_PATH)
chroma_ok = os.path.isdir(CHROMA_PERSIST_DIR) and (
    os.path.isfile(os.path.join(CHROMA_PERSIST_DIR, "index_manifest.json"))
)

if not bm25_exists or not chroma_ok:
    msg = (
        f"知识库索引不完整！\n"
        f"  BM25: {BM25_INDEX_PATH} → {'存在' if bm25_exists else '缺失'}\n"
        f"  Chroma: {CHROMA_PERSIST_DIR} → {'正常' if chroma_ok else '缺失/不完整'}\n"
        f"请先运行重建脚本：cd /opt/legal-rag && .venv/bin/python rag_agent/init_db.py"
    )
    logger.error(msg)
    # Create a minimal app that reports the error
    from flask import Flask, jsonify
    app = Flask(__name__)

    @app.route("/")
    def index():
        return jsonify({"status": "rebuilding", "message": "知识库索引正在重建，请稍后刷新"}), 503

    @app.route("/api/health")
    def health():
        return jsonify({"status": "error", "message": msg}), 503
else:
    logger.info("知识库索引已就绪，启动应用...")
    from api_server import create_app
    app = create_app()

if __name__ == "__main__":
    import os
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "5000")))
