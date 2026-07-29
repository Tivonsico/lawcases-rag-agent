"""全自动部署脚本 — SSH 到阿里云 ECS 并完成部署（部署到 /rag-agent 子路径）"""
import paramiko
import time

HOST = "182.92.152.243"
USER = "root"
PASSWORD = "Hello2023"

# 内网仓库路径（推送到 GitHub 后在服务器上 clone）
GIT_REPO = "https://github.com/Tivonsico/lawcases-rag-agent.git"
# 如果仓库是私有的，可以在这里设置 token：https://<token>@github.com/...
# GIT_REPO = "https://<token>@github.com/Tivonsico/lawcases-rag-agent.git"

SYSTEMD_SERVICE = """[Unit]
Description=Legal RAG Agent (Flask + Gunicorn)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/legal-rag
Environment=PATH=/opt/legal-rag/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin
ExecStart=/opt/legal-rag/.venv/bin/gunicorn -w 2 -b 0.0.0.0:5000 wsgi:app --access-logfile /var/log/legal-rag-access.log --error-logfile /var/log/legal-rag-error.log --log-level info --timeout 120
Restart=always
RestartSec=5
StandardOutput=append:/var/log/legal-rag-access.log
StandardError=append:/var/log/legal-rag-error.log

[Install]
WantedBy=multi-user.target
"""

COMMANDS = [
    # 1. Stop old service
    "systemctl stop legal-rag 2>/dev/null; pkill -f gunicorn 2>/dev/null; sleep 1",

    # 2. Backup old directory
    '[ -d /opt/legal-rag ] && mv /opt/legal-rag /opt/legal-rag.bak.$(date +%m%d-%H%M) || true',

    # 3. Clone latest code
    f"git clone {GIT_REPO} /opt/legal-rag",

    # 4. Create venv + install deps from requirements.txt
    "cd /opt/legal-rag && python3 -m venv .venv",
    "cd /opt/legal-rag && .venv/bin/pip install --upgrade pip -q",
    "cd /opt/legal-rag && .venv/bin/pip install --no-cache-dir -r rag_agent/requirements.txt",

    # 5. Create .env (API 密钥)
    """cat > /opt/legal-rag/.env << 'EOF'
LLM_API_KEY="sk-4ca1dc8294534c0f9be96659123ec387"
LLM_API_URL="https://api.deepseek.com/v1/chat/completions"
LLM_MODEL="deepseek-chat"
EMBEDDING_API_KEY="sk-ws-H.RXPRHLX.50Ha.MEYCIQDUVo1zdoRJlcflbbM8axrm2v7CTpYvB9CcwHqZnXWYmAIhAPH5s1Q-qQMkcvGRpKImeoyHSBQbXUlhciC-x_K5qJ9x"
EMBEDDING_API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
EMBEDDING_MODEL="text-embedding-v4"
LEGAL_RAG_RUNTIME_DIR=/opt/legal-rag/runtime
LEGAL_RAG_INDEX_DIR=/opt/legal-rag/runtime/indexes
LEGAL_RAG_DOC_DIR=/opt/legal-rag/rag_agent/data/legal_cases
EOF""",

    # 6. Pre-build index (slow, ~5-15 min). MUST complete before gunicorn starts.
    """echo "═════════════════════════════════════════════════════════════"
echo "   🔨 开始重建知识库索引（首次部署，约 5-15 分钟）"
echo "   生成向量需要调用阿里云 DashScope Embedding API..."
echo "═════════════════════════════════════════════════════════════"
cd /opt/legal-rag && .venv/bin/python -c "
import sys, logging
sys.path.insert(0, 'rag_agent')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
from init_db import build_indexes
result = build_indexes()
print(f'✅ 知识库重建完成：{len(result[\"chunks\"])} chunks, {result[\"vector_store\"].count()} 向量')
" 2>&1""",

    # 7. Create systemd service
    f"cat > /etc/systemd/system/legal-rag.service << 'SVCEOF'\n{SYSTEMD_SERVICE}\nSVCEOF",

    # 8. Create log directory
    "mkdir -p /var/log /opt/legal-rag/runtime/indexes /opt/legal-rag/runtime/reports",

    # 8. Configure Nginx for /rag-agent sub-path (不动 Java 主站)
    """NGINX_CONF=/etc/nginx/sites-available/legal-rag
cat > $NGINX_CONF << 'NEOF'
# === legal-rag agent — 部署在 /rag-agent 子路径下 ===
# Java 商城主站的 Nginx 配置在其他文件中，此文件不影响主站
server {
    listen 80;
    server_name tianfangsanjiaozhou.top tianfangdianjing.top;
    client_max_body_size 1m;

    location /rag-agent {
        return 301 /rag-agent/;
    }

    location /rag-agent/ {
        proxy_pass http://127.0.0.1:5000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 600s;
        proxy_buffering off;
        proxy_redirect / /rag-agent/;
    }
}
NEOF
# 仅启用 legal-rag 配置，不删除 default（确保不覆盖 Java 主站）
ln -sf $NGINX_CONF /etc/nginx/sites-enabled/legal-rag
nginx -t && systemctl reload nginx""",

    # 9. Enable + start service
    "systemctl daemon-reload && systemctl enable legal-rag && systemctl restart legal-rag",

    # 10. Wait for startup + health check
    "sleep 5 && curl -s http://127.0.0.1:5000/api/health",
]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

print(f"🔗 连接到 {HOST}...")
ssh.connect(HOST, username=USER, password=PASSWORD, timeout=15)
print("✅ SSH 已连接\n")

for i, cmd in enumerate(COMMANDS):
    label = f"[{i+1}/{len(COMMANDS)}]"
    print(f"{label} {cmd[:70]}{'...' if len(cmd)>70 else ''}")

    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=600)
    exit_code = stdout.channel.recv_exit_status()

    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()

    if exit_code != 0:
        # Step 1-2 failing is OK (nothing to stop)
        if i <= 1:
            print(f"  ⚠️  跳过（非致命）: {err[:200]}")
        else:
            print(f"  ❌ 失败 (exit={exit_code})")
            if err: print(f"  {err[:500]}")
            ssh.close()
            exit(1)
    else:
        if out:
            print(f"  → {out[:300]}")
        else:
            print("  ✓ 完成")

ssh.close()
print("\n🎉 部署完成！访问 http://tianfangsanjiaozhou.top/rag-agent")
