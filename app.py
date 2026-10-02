import os
import streamlit as st
import pandas as pd

from backend.config import cfg
from backend.ingest import ingest_directory, reset_vector_store
from backend.rag_pipeline import VectorStoreManager, CreativityQA, summarize_detection
from backend.detector import CreativityDetector

st.set_page_config(page_title="rag_qa (RAG + TruLens)", page_icon="🧠", layout="wide")

st.sidebar.title("⚙️ 設定")
ollama_url = st.sidebar.text_input("Ollama Base URL", value=cfg.OLLAMA_BASE_URL)
model_name = st.sidebar.text_input("Ollama 模型名稱", value=cfg.OLLAMA_MODEL)
rag_on = st.sidebar.checkbox("🔍 啟用 RAG 檢索", value=True)
top_k = st.sidebar.slider("Top-K 文件數", 1, 10, cfg.TOP_K)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, cfg.TEMPERATURE, 0.05)
show_ctx = st.sidebar.checkbox("顯示檢索片段", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 TruLens 評分設定（Azure）")
azure_endpoint_default = st.session_state.get("trulens_azure_endpoint", os.getenv("AZURE_OPENAI_ENDPOINT", ""))
azure_key_default = st.session_state.get("trulens_azure_api_key", os.getenv("AZURE_OPENAI_API_KEY", ""))
azure_version_default = st.session_state.get("trulens_azure_api_version", os.getenv("OPENAI_API_VERSION", ""))
azure_deploy_default = st.session_state.get("trulens_azure_deployment", os.getenv("AZURE_OPENAI_DEPLOYMENT", ""))

azure_endpoint = st.sidebar.text_input("Azure OpenAI Endpoint", value=azure_endpoint_default)
azure_api_key = st.sidebar.text_input("Azure OpenAI API Key", value=azure_key_default, type="password")
azure_api_version = st.sidebar.text_input(
    "OPENAI_API_VERSION（例：2024-06-01）", value=azure_version_default or "2024-06-01"
)
azure_deployment = st.sidebar.text_input("Azure OpenAI 部署名稱", value=azure_deploy_default)

if st.sidebar.button("套用 TruLens 設定"):
    missing = []
    azure_fields = {
        "AZURE_OPENAI_ENDPOINT": azure_endpoint,
        "AZURE_OPENAI_API_KEY": azure_api_key,
        "OPENAI_API_VERSION": azure_api_version,
        "AZURE_OPENAI_DEPLOYMENT": azure_deployment,
    }
    for label, env_name in [
        ("Azure OpenAI Endpoint", "AZURE_OPENAI_ENDPOINT"),
        ("Azure OpenAI API Key", "AZURE_OPENAI_API_KEY"),
        ("OPENAI_API_VERSION", "OPENAI_API_VERSION"),
        ("Azure 部署名稱", "AZURE_OPENAI_DEPLOYMENT"),
    ]:
        if not azure_fields[env_name].strip():
            missing.append(label)

    if missing:
        st.sidebar.error(f"請完整輸入：{'、'.join(missing)}")
    else:
        for env_name, value in azure_fields.items():
            os.environ[env_name] = value.strip()
        st.session_state["trulens_azure_endpoint"] = azure_endpoint
        st.session_state["trulens_azure_api_key"] = azure_api_key
        st.session_state["trulens_azure_api_version"] = azure_api_version
        st.session_state["trulens_azure_deployment"] = azure_deployment
        st.sidebar.success("已更新 TruLens 評分設定（Azure）。")

st.sidebar.markdown("---")
st.sidebar.subheader("📚 知識庫")
data_dir = st.sidebar.text_input("資料夾路徑", value="./data")
reset_before_ingest = st.sidebar.checkbox("重新索引前清空舊資料", value=True)
if st.sidebar.button("索引 / 重新索引"):
    with st.spinner("正在建立向量索引 ..."):
        ingest_directory(
            data_dir=data_dir,
            persist_dir=cfg.CHROMA_DIR,
            collection_name=cfg.COLLECTION_NAME,
            reset=reset_before_ingest,
        )
    st.sidebar.success("索引完成！")

if st.sidebar.button("🔁 清空向量資料庫"):
    reset_vector_store(cfg.CHROMA_DIR)
    st.sidebar.success("已清空向量資料庫。請重新索引。")

uploaded = st.sidebar.file_uploader("上傳檔案（txt/pdf/docx）", accept_multiple_files=True, type=["txt", "pdf", "docx"])
if uploaded:
    os.makedirs(data_dir, exist_ok=True)
    for f in uploaded:
        save_path = os.path.join(data_dir, f.name)
        with open(save_path, "wb") as out:
            out.write(f.read())
    st.sidebar.success(f"已儲存 {len(uploaded)} 個檔案到 {data_dir}，請按『索引 / 重新索引』。")

st.title("🧠 rag_qa：RAG 問答與 TruLens 評估原型")
st.caption("以創造力語意偵測 + RAG 與 Base LLM 並行輸出，觀察差異。")

vs = VectorStoreManager(persist_dir=cfg.CHROMA_DIR, collection_name=cfg.COLLECTION_NAME)
detector = CreativityDetector()
qa = CreativityQA(vs, detector=detector)

if "hist" not in st.session_state:
    st.session_state.hist = []

question = st.text_input("輸入你的問題（例：『學生常對課本內容提出質疑是否代表好奇心強？』）")
ask = st.button("送出")

if ask and question.strip():
    with st.spinner("生成中 ..."):
        res = qa.answer_pair(
            question=question.strip(),
            enable_rag=rag_on,
            k=top_k,
            temperature=temperature,
            base_url=ollama_url,
            model=model_name,
        )
        st.session_state.hist.append({"q": question, **res})

for turn in st.session_state.hist[::-1]:
    st.markdown(f"**你：** {turn['q']}")
    detection = turn.get("detection")
    det_summary = summarize_detection(detection)
    st.info(f"語意偵測：{det_summary}")
    cols = st.columns(3)

    with cols[0]:
        st.subheader("✳️ Base LLM（無 Prompt）")
        st.markdown(turn.get("base", {}).get("answer", "_尚無資料_"))

    with cols[1]:
        st.subheader("📝 Base + Prompt 回應")
        st.markdown(turn.get("base_prompt", {}).get("answer", "_尚無資料_"))

    with cols[2]:
        st.subheader("🔍 RAG + Prompt 回應")
        if turn.get("rag_used") and turn.get("rag", {}).get("answer"):
            st.markdown(turn["rag"]["answer"])
            if show_ctx and turn["rag"]["docs"]:
                with st.expander("檢索片段（Context）"):
                    for i, d in enumerate(turn["rag"]["docs"], 1):
                        meta = d.metadata or {}
                        st.markdown(f"- **片段#{i}** | 來源: `{meta.get('source','unknown')}`")
                        st.write(d.page_content)
        else:
            st.warning("本次未使用 RAG（偵測信心不足或使用者停用）。")

st.markdown("---")
st.subheader("📊 TruLens 快速評估")
st.write("按下按鈕將用預設題庫對 RAG 與 Base 三指標（RAG Triad）做一次比較，結果會輸出到 `./eval/results.csv`。")
if st.button("執行評估（RAG vs. Base）"):
    from backend.evaluate_trulens import run_eval
    with st.spinner("TruLens 評估執行中，請稍候 ..."):
        df = run_eval(questions=None, questions_path="./eval/questions_zh.json", base_url=ollama_url, model=model_name)
        os.makedirs("./eval", exist_ok=True)
        df.to_csv("./eval/results.csv", index=False, encoding="utf-8-sig")
    st.success("TruLens 評估完成，已輸出 ./eval/results.csv")
    st.dataframe(df)
