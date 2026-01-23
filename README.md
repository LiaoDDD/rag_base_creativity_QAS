# 創造力傾向 RAG 問答系統（語意偵測 + TruLens 評估）

- **介面**：Streamlit（預設 port 8501）同時呈現 RAG / Base LLM 回應
- **語意偵測**：CreativityDetector（向量 + 關鍵字）標註最可能的創造力特質
- **RAG 管線**：LangChain + Chroma（預設寫入 `./chroma_db/`，若不可寫會自動改用暫存或 `CHROMA_FALLBACK_DIR`），每題同時輸出 RAG 與 Base LLM
- **評估**：TruLens（Groundedness / Context Relevance / Answer Relevance）匯出 `eval/results.csv`

## 1) 本機快速執行（無 Docker）
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
# 打開 http://localhost:8501
```

## 2) 容器化部署（Docker + Compose）
```bash
docker compose up --build
# 打開 http://localhost:8501
```

## 3) 準備知識庫
- 把 `txt/pdf/docx` 放進 `./data/`
- UI 側欄勾選「重新索引前清空舊資料」並按「索引 / 重新索引」
- 若要完全重建，可按「清空向量資料庫」後再索引（避免資料重複）

## 4) TruLens 評估（RAG vs Base）
- **評分需求**：回答仍由 Ollama 模型生成，但三項 TruLens 指標（Answer / Context / Groundedness）目前僅支援 **Azure OpenAI**。請在執行評估前設定：
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_KEY`
  - `OPENAI_API_VERSION`（例：`2024-06-01`）
  - `AZURE_OPENAI_DEPLOYMENT`（部署名稱）
- 若沙箱無法寫入 `/tmp`，請事先將 `TMPDIR`/`TEMP`/`TMP` 指到可寫路徑，或暫時設定 `RAG_QA_SKIP_TEMP_CHECK=1`（僅建議於受限環境除錯）。
- `run_eval` 會直接呼叫 TruLens `AzureOpenAIProvider` 進行評分，不再依賴 `TruChain` Recorder，避免指標卡在 NaN 或固定值。輸出包含：
  - `eval/results.csv`：題目 `trait`、RAG/Base 逐題回答、實際檢索片段、`rag_doc_ids`、`rag_cited_doc_ids`、`Citations_*` 等欄位。
  - `eval/metrics_summary.csv`：仍保留平均值、標準差與樣本數，並統計 `Citation Hit Rate`（回答真正引用量表題項的比例）。
  - `eval/fig5_1.png`：以最新統計重新產製的對照圖（Base 仍無情境/根據性分數，RAG 三項皆期望接近 1）。
- UI：頁面底部按「執行評估」即可觸發上述流程。
- CLI：
```bash
docker compose run --rm eval  # 預設使用 eval/questions_zh.json
# 或
python -m backend.evaluate_trulens
```
- 預設題庫 `eval/questions_zh.json` 為 JSON Array，每題包含 `id` / `trait` / `question` 三欄；可自訂更多情境以擴大評估樣本。
- 若環境無法寫入 `/tmp` 等暫存路徑，請先將 `TMPDIR`/`TEMP`/`TMP` 指到可寫入的資料夾，或在必要時設定 `RAG_QA_SKIP_TEMP_CHECK=1`（僅建議沙箱內開發使用）。

## 5) 可調參數（環境變數或側欄）
- `OLLAMA_BASE_URL`（預設 `http://localhost:11434`，或改成你的 Ollama/LLM 服務位址）
- `OLLAMA_MODEL`（例：`llama3.3:70b` 或你部署的模型名稱）
- `HF_EMBEDDING_MODEL`（例：`intfloat/multilingual-e5-base`）
- `CHROMA_DIR`（預設 `./chroma_db`。若該路徑不可寫，系統會依序嘗試 `CHROMA_FALLBACK_DIR`、系統暫存 `/tmp/rag_qa_chroma_db`、以及 `~/.cache/rag_qa_chroma_db`）
- `CHROMA_FALLBACK_DIR`（可選。提供具寫入權限的備用路徑給向量資料庫使用）
- `CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K` / `TEMPERATURE`
- `DETECTION_THRESHOLD`（預設 0.25，決定是否啟用 RAG）

## 6) 部署提示
- 若使用 Docker，請依環境調整 `docker-compose.yml` 中的 `OLLAMA_BASE_URL`、`OLLAMA_MODEL` 與對外 port。
- 若要使用本機 Ollama，請先啟動 Ollama 服務並確保模型已拉取完成。
- 若要在無 Azure OpenAI 的環境下先測試 UI，可略過 TruLens 評估流程。

---

### 系統架構重點
1. **創造力語意偵測模組**：結合 HuggingFace 向量與關鍵字規則，輸出特質標籤、信心值與是否啟用 RAG 的建議。
2. **檢索協調器**：依偵測結果啟用 RAG 或回退 Base LLM，並對檢索片段加上來源資訊。
3. **提示詞層級化**：System Prompt（角色/倫理）、Context Block（檢索結果）、Response Schema（結論→理由→限制）。
4. **TruLens 評估**：冷啟動時重置資料庫，確保每次比較僅包含本回合紀錄；結果表列出每題的 RAG/Base 回應與指標。
