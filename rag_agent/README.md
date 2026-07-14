# 法律案例 RAG

本项目是本地法律案例检索与问答原型，包含结构化切分、向量/BM25 多路召回、标准加权 RRF、文档聚合、评测、带用户隔离的 Flask API 和前端。它不是法律意见，也不应直接作为公开生产服务部署。

## 环境与安装

支持 Python 3.10～3.13。以下命令均从仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

`pyproject.toml` 是依赖真源；`rag_agent/requirements.txt` 仅为旧工具兼容镜像。更新依赖时须同步该文件。

## 配置

当前原型仍兼容 `rag_agent/config.py` 中的本地模型密钥（按本轮需求未迁移或轮换）。不要提交新密钥。网站账号由用户在页面自行注册，不需要配置访问 Token。

本地账号保存在 `runtime/auth/users.json`。按当前原型要求，密码暂时以**明文**保存；只能在可信本机使用，严禁部署到公网。上服务器前必须迁移到数据库并使用成熟的密码哈希方案。

可选路径变量：

- `LEGAL_RAG_DATA_DIR`、`LEGAL_RAG_DOC_DIR`
- `LEGAL_RAG_RUNTIME_DIR`、`LEGAL_RAG_INDEX_DIR`
- `LEGAL_RAG_CHROMA_DIR`、`LEGAL_RAG_BM25_PATH`
- `LEGAL_RAG_REPORT_DIR`、`LEGAL_RAG_SESSION_DIR`
- `LEGAL_RAG_LONG_TERM_DB`

默认新产物写入仓库根的 `runtime/`，不会迁移或删除已有的 `rag_agent/chroma_db`、BM25、SQLite、会话和报告。若要继续使用旧产物，请用上述变量明确指向旧路径。

## 数据与建索引

案例文本默认位于 `rag_agent/data/legal_cases/`。建库命令：

```powershell
python -m rag_agent.init_db
```

构建流程使用 upsert，并为 Chroma 与 BM25 写入同一份 manifest。manifest 记录语料、切分和 embedding 配置；加载时不兼容会明确失败，此时应备份旧索引后重新构建。BM25 使用 pickle，只能加载管理员配置的本地固定文件，绝不能使用上传文件或不可信 pickle。

## CLI、API 与前端

交互式 CLI：

```powershell
python -m rag_agent.main
```

启动 API（默认仅监听 `127.0.0.1:5000`）：

```powershell
python -m rag_agent.api_server
```

浏览器访问 `http://127.0.0.1:5000/`，在右上角输入账号和密码后点击“注册”，以后使用同一账号密码登录。服务端会签发临时登录令牌供浏览器内部使用，用户无需查看或配置 Token；服务重启后需要重新登录。客户端不能自行指定 `user_id`。API 包含会话所有权校验、请求体/消息长度、限流、并发和缓存上限；401、404、413、429 会分别提示。

健康检查无需认证：

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/health
```

## 评测

调参与留出评测严格分开：

```powershell
python rag_agent/main.py
```

启动后输入 `/test`。两次直接回车会评测 `evaluation_100.jsonl` 的 100 条数据，并使用与正常提问相同的完整混合检索（BM25 + 向量）。报告以中文显示召回率、准确率、命中率、平均准确率、NDCG、MRR 和 P50/P95 延迟；任一通道失败时会中止本次完整混合评测，不输出误导性的全 0 结果。需要离线排查时才在第二个提示输入 `bm25`。

候选数应根据 dev 集 Recall 曲线的平台点选择：依次比较 K=20、50、100（需要时继续扩大）带来的召回增量与 P50/P95 延迟。当增大 K 几乎不再提高 Recall、但延迟明显增加时，选平台附近更小的 K，而不是固定选择最大值。

### 30 条深标 + 70 条已知目标评测

仓库提供一个可审计的四阶段流程；`drafts/` 下现有 30+70 条记录仍是机器草稿，不是正式基准，也不会替换原来的 11 条 sanity set。

1. 从 535 篇语料按来源配额和固定 seed 生成互不重叠的候选：

   ```powershell
   python -m rag_agent.evaluation_data prepare --doc-dir rag_agent/data/legal_cases --output-dir rag_agent/data/evaluation/drafts --seed 20260713 --exclude-jsonl rag_agent/data/evaluation/dev.jsonl --exclude-jsonl rag_agent/data/evaluation/test.jsonl
   ```

2. 可选地调用兼容 OpenAI Chat Completions 的服务生成自然问题。先用 `--max-records 2` 做小样本检查；密钥只放环境变量，不写入仓库。此步骤不会把记录标成人工通过：

   ```powershell
   $env:DEEPSEEK_API_KEY="<rotated-key>"
   python -m rag_agent.evaluation_generator --input rag_agent/data/evaluation/drafts/target_review.jsonl --manifest rag_agent/data/evaluation/drafts/corpus_manifest.json --output rag_agent/data/evaluation/drafts/target_generated.jsonl --api-url https://api.deepseek.com/v1/chat/completions --model deepseek-chat --api-key-env DEEPSEEK_API_KEY --max-records 2
   ```

3. 人工复核。Target 每条确认问题自然、没有案号/完整标题/姓名泄漏，填写 `reviewer`、`reviewed_at`、`target_leakage_checked=true`，最后改为 `human_verified_target`。Core 需要先合并多组检索结果形成候选池，再逐条给 0/1/2 相关性，核对 `relevant_case_ids`，最后改为 `human_verified_core`。可随时运行：

   ```powershell
   python -m rag_agent.evaluation_data validate rag_agent/data/evaluation/drafts/target_review.jsonl
   ```

4. 只有全部记录满足协议、人工字段和期望数量时才能原子发布；否则命令失败且不产生正式文件：

   ```powershell
   python -m rag_agent.evaluation_data publish --input rag_agent/data/evaluation/drafts/core_review.jsonl --output rag_agent/data/evaluation/core.jsonl --manifest rag_agent/data/evaluation/drafts/corpus_manifest.json --protocol core --expected-count 30
   python -m rag_agent.evaluation_data publish --input rag_agent/data/evaluation/drafts/target_review.jsonl --output rag_agent/data/evaluation/target.jsonl --manifest rag_agent/data/evaluation/drafts/corpus_manifest.json --protocol target --expected-count 70
   ```

Core 可报告 pooled Recall、Precision/MAP、graded NDCG、MRR 和 judgment coverage，但它们只对标注候选池有效；Target 只报告已知目标的 Recall/HitRate/MRR，禁止报告 Precision/MAP/NDCG。两类协议不做混合总分。

`rag_agent/data/evaluation/ablation_variants.json` 定义 9 组实验：纯向量、纯 BM25、两路融合、依次加入 normalization/HyDE/同义词/exact、六路等权对照，以及带 reranker 的最终组。通过 `load_variants` 与 `evaluate_variants` 让每组继续走同一个 `HybridRetriever.search` 入口。只在 dev 上选权重和 RRF k，冻结配置后再跑 test；汇总 Recall@20、NDCG@5、MRR、P50/P95 和单请求成本。reranker 组必须由调用方提供真实 reranker，否则评测会拒绝运行。

## 测试

测试不需要真实网络或生产密钥：

```powershell
pytest -q
python -m compileall rag_agent tests
```

可分别运行 `tests/test_retrieval.py`、`test_evaluation.py`、`test_indexing.py` 与 `test_api_security.py` 定位问题。

## 数据生命周期与部署限制

- `runtime/indexes/`：Chroma、BM25 与 manifest；配置变化后整体重建，不混用版本。
- `runtime/users/`、`runtime/long_term_memory.db`：敏感会话与画像；应限制文件权限，并制定保留/删除策略。
- `runtime/reports/`、日志：实验产物；不得包含 token 或完整敏感咨询内容。
- `.gitignore` 只阻止新产物入库，不会删除已有文件；已被 Git 跟踪的秘密或数据仍需单独处理历史。
- 当前鉴权、限流、并发计数和 LRU/TTL 缓存是单进程内存实现。多 worker 或多实例部署必须改用 Redis/数据库等共享后端，并在反向代理层配置 TLS、请求大小、超时和限流。
- 外部 embedding/LLM 请求需在生产环境补充统一的重试退避、审计、成本监控和隐私合规策略。
- 模型自身知识不等于网络来源。回答只把实际检索到的文档标为案例来源，其他内容明确标为模型生成且未联网核验；当前出处检查不能替代逐结论的证据一致性验证。
