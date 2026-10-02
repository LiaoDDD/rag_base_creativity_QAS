import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from chromadb import errors as chroma_errors

from backend.config import cfg
from backend.embeddings import build_embeddings_with_fallback
from backend.prompts import SYSTEM_PROMPT, RAG_USER_PROMPT, BASE_USER_PROMPT
from backend.detector import DetectionResult, CreativityDetector

def build_embeddings():
    return build_embeddings_with_fallback(
        model_name=cfg.HF_EMBEDDING_MODEL,
        device=cfg.HF_EMBEDDING_DEVICE,
        batch_size=cfg.HF_EMBEDDING_BATCH,
        max_length=cfg.HF_EMBEDDING_MAX_LENGTH,
        normalize=cfg.HF_EMBEDDING_NORMALIZE,
        query_prefix=cfg.HF_EMBEDDING_QUERY_PREFIX,
        document_prefix=cfg.HF_EMBEDDING_DOC_PREFIX,
    )

class VectorStoreManager:
    def __init__(self, persist_dir: Optional[str] = None, collection_name: Optional[str] = None):
        self.persist_dir = persist_dir or cfg.CHROMA_DIR
        self.collection_name = collection_name or cfg.COLLECTION_NAME
        self._emb = build_embeddings()
        self.vs = self._build_store()

    def _build_store(self):
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self._emb,
            persist_directory=self.persist_dir,
        )

    def _reset_persist_dir(self):
        if os.path.isdir(self.persist_dir):
            shutil.rmtree(self.persist_dir)
        os.makedirs(self.persist_dir, exist_ok=True)

    def _switch_to_temp_dir(self):
        """Fallback to a guaranteed-writable temporary directory when persistence fails."""
        sandbox_base = Path(tempfile.gettempdir()) / "rag_qa_chroma_fallbacks"
        sandbox_base.mkdir(parents=True, exist_ok=True)
        new_dir = tempfile.mkdtemp(prefix="chroma_", dir=str(sandbox_base))
        warnings.warn(
            f"Chroma 向量庫路徑 {self.persist_dir} 無法寫入，"
            f"已自動改用暫存目錄 {new_dir}。"
        )
        self.persist_dir = new_dir
        self.vs = self._build_store()

    def _handle_chroma_error(self, exc: Exception, retry_callable):
        message = str(exc)
        if isinstance(exc, chroma_errors.InternalError) and "no such table: embeddings_queue" in message:
            warnings.warn(
                "偵測到舊版 Chroma SQLite schema（缺少 embeddings_queue）；將自動清空持久化資料重新建立。"
            )
            self._reset_persist_dir()
            self.vs = self._build_store()
            return retry_callable()
        if isinstance(exc, chroma_errors.InternalError) and "readonly" in message.lower():
            self._switch_to_temp_dir()
            return retry_callable()
        raise

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        try:
            return self.vs.add_texts(texts=texts, metadatas=metadatas)  # Chroma 0.4+ auto-persist
        except Exception as exc:  # pragma: no cover - defensive path
            return self._handle_chroma_error(exc, lambda: self.vs.add_texts(texts=texts, metadatas=metadatas))

    def similarity_search(self, query: str, k: int = None):
        k = k or cfg.TOP_K
        try:
            docs = self.vs.similarity_search(query, k=k)
        except Exception as exc:  # pragma: no cover - defensive path
            docs = self._handle_chroma_error(exc, lambda: self.vs.similarity_search(query, k=k))
        return docs

def get_llm(temperature: float = None, base_url: Optional[str] = None, model: Optional[str] = None):
    return ChatOllama(
        base_url=base_url or cfg.OLLAMA_BASE_URL,
        model=model or cfg.OLLAMA_MODEL,
        temperature=cfg.TEMPERATURE if temperature is None else temperature,
        num_ctx=cfg.OLLAMA_NUM_CTX
    )

def summarize_detection(detection: Optional[DetectionResult]) -> str:
    if detection is None or detection.trait == "未知":
        return "尚未偵測到明確的創造力特質，請以通用知識提供謹慎回應。"

    keywords = ", ".join(detection.keyword_hits) if detection.keyword_hits else "（無直接關鍵詞）"
    facets = ", ".join(detection.subfacets) if getattr(detection, "subfacets", None) else "（未命中子指標）"
    return (
        f"最可能的創造力特質：{detection.trait}。"
        f"關鍵詞命中：{keywords}。"
        f"子指標：{facets}。"
    )


class CreativityQA:
    def __init__(self, vector_store: VectorStoreManager, detector: Optional[CreativityDetector] = None):
        self.vs = vector_store
        self.detector = detector or CreativityDetector()
        self.prompt_rag = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", RAG_USER_PROMPT)]
        )
        self.prompt_base = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", BASE_USER_PROMPT)]
        )
        self.parser = StrOutputParser()

    def format_context(self, docs) -> str:
        if not docs:
            return "（尚未建立或找到符合的創造力文本）"
        parts = []
        for _, d in enumerate(docs, 1):
            meta = d.metadata or {}
            src = meta.get("source", "unknown")
            labels = [f"來源: {src}"]
            if meta.get("scale"):
                labels.append(meta["scale"])
            if meta.get("item_id"):
                labels.append(f"題項: {meta['item_id']}")
            if meta.get("trait"):
                labels.append(f"特質: {meta['trait']}")
            facets = meta.get("facets") or meta.get("subfacets")
            if facets:
                if isinstance(facets, str):
                    labels.append(f"指標: {facets}")
                else:
                    labels.append(f"指標: {', '.join(facets)}")
            parts.append(f"[{' | '.join(labels)}]\n{d.page_content.strip()}")
        return "\n\n".join(parts)

    def _invoke_llm(self, prompt_template, question, context, detection_summary, llm):
        messages = prompt_template.format_messages(
            question=question, context=context, detection_summary=detection_summary
        )
        return self.parser.invoke(llm.invoke(messages))

    def _invoke_direct_llm(self, question: str, llm):
        """Bypass system prompts and detectors so the base model answers with its raw response."""
        return self.parser.invoke(llm.invoke([HumanMessage(content=question)]))

    def answer_pair(
        self,
        question: str,
        enable_rag: bool = True,
        k: Optional[int] = None,
        temperature: Optional[float] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        detection: Optional[DetectionResult] = None,
    ):
        llm = get_llm(temperature=temperature, base_url=base_url, model=model)

        base_answer = self._invoke_direct_llm(question, llm)

        detection = detection or self.detector.detect(question)
        detection_summary = summarize_detection(detection)
        prompt_context = "（此模式未使用外部檢索脈絡）"
        base_prompt_answer = self._invoke_llm(
            self.prompt_base,
            question,
            prompt_context,
            detection_summary,
            llm,
        )
        rag_docs: List[Any] = []
        rag_answer = None
        rag_used = enable_rag and detection.should_use_rag(cfg.DETECTION_THRESHOLD)
        if rag_used:
            rag_docs = self.vs.similarity_search(question, k=k or cfg.TOP_K)
            context_text = self.format_context(rag_docs)
            rag_answer = self._invoke_llm(self.prompt_rag, question, context_text, detection_summary, llm)

        return {
            "detection": detection,
            "rag_used": rag_used,
            "rag": {"answer": rag_answer, "docs": rag_docs},
            "base_prompt": {"answer": base_prompt_answer, "docs": []},
            "base": {"answer": base_answer, "docs": []},
        }
