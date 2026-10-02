from __future__ import annotations

import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.config import cfg
from backend.rag_pipeline import VectorStoreManager, CreativityQA

from trulens.providers.openai import AzureOpenAI as AzureOpenAIProvider


MAX_TRULENS_SCORE = 3.0
DOC_ID_PATTERN = re.compile(r"Q\d{2}", re.IGNORECASE)


@dataclass
class EvalQuestion:
    id: int | None
    trait: str
    text: str


def load_questions(questions: List[Any] | None = None, path: str | None = None) -> List[EvalQuestion]:
    """Normalize the question payload into EvalQuestion objects."""

    normalized: List[EvalQuestion] = []

    def _normalize_entry(entry: Any) -> EvalQuestion | None:
        if isinstance(entry, EvalQuestion):
            return entry
        if isinstance(entry, dict):
            text = entry.get("question") or entry.get("text")
            if not text:
                return None
            trait = entry.get("trait") or ""
            return EvalQuestion(id=entry.get("id"), trait=trait, text=text.strip())
        if isinstance(entry, str):
            return EvalQuestion(id=None, trait="", text=entry.strip())
        return None

    if questions:
        for item in questions:
            normalized_entry = _normalize_entry(item)
            if normalized_entry and normalized_entry.text:
                normalized.append(normalized_entry)

    if not normalized and path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            normalized_entry = _normalize_entry(entry)
            if normalized_entry and normalized_entry.text:
                normalized.append(normalized_entry)

    if not normalized:
        raise RuntimeError("找不到可用的評估題目，請確認 questions 或 questions_path。")

    return normalized


def _series_stats(series: pd.Series) -> tuple[float, float, int]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan, np.nan, 0
    return float(values.mean()), float(values.std(ddof=0)), int(len(values))


def _hit_rate(series: pd.Series) -> tuple[float, int]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan, 0
    hits = int((values > 0).sum())
    return float(hits / len(values)), int(len(values))


def _normalize_score(value: Any, max_score: float = MAX_TRULENS_SCORE) -> float:
    if value is None:
        return np.nan
    try:
        score = float(value)
    except (TypeError, ValueError):
        return np.nan
    score = max(0.0, min(score, max_score))
    return score / max_score


def _build_azure_provider_for_eval() -> AzureOpenAIProvider:
    missing = [
        name
        for name in (
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "OPENAI_API_VERSION",
            "AZURE_OPENAI_DEPLOYMENT",
        )
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(f"Azure 評分模式缺少環境變數：{', '.join(missing)}")

    provider = AzureOpenAIProvider(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("OPENAI_API_VERSION"),
    )

    try:
        provider.clear_capabilities_cache()
        provider._set_capabilities({"cfg": False, "structured_outputs": False})
    except Exception as exc:  # pragma: no cover - defensive logging
        warnings.warn(f"無法調整 AzureOpenAIProvider 能力標記：{exc}")

    return provider


def _call_with_retry(func, *args, max_attempts: int = 3, retry_wait: float = 5.0, **kwargs):
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - network / API errors
            last_exc = exc
            if attempt >= max_attempts:
                break
            time.sleep(retry_wait)
    raise RuntimeError(f"TruLens 指標計算失敗：{last_exc}")


def _score_answer(
    provider,
    question: str,
    answer: str | None,
    *,
    max_attempts: int,
    retry_wait: float,
) -> float:
    if not answer or not answer.strip():
        return np.nan
    score, _ = _call_with_retry(
        provider.relevance_with_cot_reasons,
        question,
        answer,
        max_attempts=max_attempts,
        retry_wait=retry_wait,
    )
    return _normalize_score(score)


def _score_context(
    provider,
    question: str,
    context: str,
    *,
    max_attempts: int,
    retry_wait: float,
) -> float:
    if not context or "尚未" in context:
        return np.nan
    score, _ = _call_with_retry(
        provider.context_relevance_with_cot_reasons,
        question,
        context,
        max_attempts=max_attempts,
        retry_wait=retry_wait,
    )
    return _normalize_score(score)


def _score_groundedness(
    provider,
    context: str,
    answer: str,
    *,
    max_attempts: int,
    retry_wait: float,
) -> float:
    if not context or not answer:
        return np.nan
    score, _ = _call_with_retry(
        provider.groundedness_measure_with_cot_reasons,
        context,
        answer,
        max_attempts=max_attempts,
        retry_wait=retry_wait,
    )
    return _normalize_score(score)


def _answer_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        ans = value.get("answer")
        if isinstance(ans, str):
            return ans
        if ans is not None:
            return str(ans)
    return "" if value is None else str(value)


def _doc_metadata(docs) -> tuple[List[str], List[str]]:
    doc_ids: List[str] = []
    doc_labels: List[str] = []
    for doc in docs or []:
        meta = doc.metadata or {}
        doc_id = str(meta.get("item_id") or meta.get("source") or "").strip()
        if doc_id:
            doc_ids.append(doc_id)
        parts: List[str] = []
        for key in ("source", "scale", "trait", "item_id"):
            val = meta.get(key)
            if val:
                parts.append(str(val))
        if parts:
            doc_labels.append(" | ".join(parts))
    return doc_ids, doc_labels


def _match_doc_ids(answer: str, candidates: Sequence[str]) -> List[str]:
    if not answer:
        return []
    lower = answer.lower()
    matched: List[str] = []
    seen: set[str] = set()
    for doc_id in candidates:
        key = doc_id.lower()
        if key in seen:
            continue
        if key and key in lower:
            matched.append(doc_id)
            seen.add(key)
    return matched


def _regex_citations(answer: str) -> int:
    if not answer:
        return 0
    return len(DOC_ID_PATTERN.findall(answer))


def _plot_and_save_bar(summary_df: pd.DataFrame, out_path: str):
    metrics = ["Context Relevance", "Groundedness", "Answer Relevance"]
    modes = ["Base", "RAG"]
    x = np.arange(len(metrics))
    width = 0.36

    def _values_for(mode: str) -> List[float]:
        values: List[float] = []
        for metric in metrics:
            subset = summary_df.loc[summary_df["mode"] == mode, metric]
            val = subset.values[0] if not subset.empty else np.nan
            values.append(val)
        return values

    base_vals = _values_for("Base")
    rag_vals = _values_for("RAG")

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, base_vals, width, label="未使用 RAG（Base）")
    plt.bar(x + width / 2, rag_vals, width, label="使用 RAG")
    plt.xticks(x, metrics)
    plt.ylim(0, 1.05)
    plt.ylabel("平均得分（0–1）")
    plt.title("圖5-1：未使用 RAG 與使用 RAG 之系統效能指標比較")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def run_eval(
    questions: List[Any] | None = None,
    questions_path: str = "./eval/questions_zh.json",
    persist_dir: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    *,
    max_feedback_attempts: int = 3,
    feedback_retry_wait: float = 5.0,
) -> pd.DataFrame:
    base_url = base_url or cfg.OLLAMA_BASE_URL
    model = model or cfg.OLLAMA_MODEL

    eval_questions = load_questions(questions, questions_path)
    provider = _build_azure_provider_for_eval()

    vs = VectorStoreManager(persist_dir=persist_dir or cfg.CHROMA_DIR, collection_name=cfg.COLLECTION_NAME)
    qa = CreativityQA(vs)

    rows: List[Dict[str, Any]] = []
    os.makedirs("./eval", exist_ok=True)

    for item in eval_questions:
        question = item.text
        result = qa.answer_pair(
            question=question,
            enable_rag=True,
            k=cfg.TOP_K,
            temperature=cfg.TEMPERATURE,
            base_url=base_url,
            model=model,
        )

        rag_payload = result.get("rag") or {}
        rag_answer = _answer_text(rag_payload.get("answer"))
        rag_docs = rag_payload.get("docs") or []
        rag_context = qa.format_context(rag_docs)
        rag_doc_ids, rag_doc_labels = _doc_metadata(rag_docs)
        rag_cited_doc_ids = _match_doc_ids(rag_answer, rag_doc_ids)
        rag_used = bool(result.get("rag_used") and rag_answer)

        base_prompt_payload = result.get("base_prompt") or {}
        base_prompt_answer = _answer_text(base_prompt_payload.get("answer"))

        base_payload = result.get("base") or {}
        base_answer = _answer_text(base_payload.get("answer"))

        answer_rel_rag = _score_answer(
            provider,
            question,
            rag_answer,
            max_attempts=max_feedback_attempts,
            retry_wait=feedback_retry_wait,
        )
        context_rel_rag = _score_context(
            provider,
            question,
            rag_context,
            max_attempts=max_feedback_attempts,
            retry_wait=feedback_retry_wait,
        )
        groundedness_rag = _score_groundedness(
            provider,
            rag_context,
            rag_answer,
            max_attempts=max_feedback_attempts,
            retry_wait=feedback_retry_wait,
        )

        answer_rel_base = _score_answer(
            provider,
            question,
            base_answer,
            max_attempts=max_feedback_attempts,
            retry_wait=feedback_retry_wait,
        )
        citations_base = _regex_citations(base_answer)
        answer_rel_base_prompt = _score_answer(
            provider,
            question,
            base_prompt_answer,
            max_attempts=max_feedback_attempts,
            retry_wait=feedback_retry_wait,
        )
        citations_base_prompt = _regex_citations(base_prompt_answer)

        rows.append(
            {
                "id": item.id,
                "trait": item.trait,
                "question": question,
                "rag_used": rag_used,
                "answer_rag": rag_answer,
                "answer_base": base_answer,
                "answer_base_prompt": base_prompt_answer,
                "context_rag": rag_context,
                "rag_sources": "; ".join(rag_doc_labels),
                "rag_doc_ids": "; ".join(rag_doc_ids),
                "rag_cited_doc_ids": "; ".join(rag_cited_doc_ids),
                "AnswerRel_RAG": answer_rel_rag,
                "ContextRel_RAG": context_rel_rag,
                "Groundedness_RAG": groundedness_rag,
                "AnswerRel_Base": answer_rel_base,
                "AnswerRel_BasePrompt": answer_rel_base_prompt,
                "Citations_RAG": len(rag_cited_doc_ids),
                "Citations_Base": citations_base,
                "Citations_BasePrompt": citations_base_prompt,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv("./eval/results.csv", index=False, encoding="utf-8-sig")

    base_answer_rel, base_answer_rel_std, base_answer_rel_count = _series_stats(df["AnswerRel_Base"])
    base_prompt_answer_rel, base_prompt_answer_rel_std, base_prompt_answer_rel_count = _series_stats(
        df["AnswerRel_BasePrompt"]
    )
    rag_answer_rel, rag_answer_rel_std, rag_answer_rel_count = _series_stats(df["AnswerRel_RAG"])
    rag_context_rel, rag_context_rel_std, rag_context_rel_count = _series_stats(df["ContextRel_RAG"])
    rag_groundedness, rag_groundedness_std, rag_groundedness_count = _series_stats(df["Groundedness_RAG"])
    rag_citation_rate, rag_citation_samples = _hit_rate(df["Citations_RAG"])
    base_citation_rate, base_citation_samples = _hit_rate(df["Citations_Base"])
    base_prompt_citation_rate, base_prompt_citation_samples = _hit_rate(df["Citations_BasePrompt"])

    summary = pd.DataFrame(
        [
            {
                "mode": "Base",
                "Answer Relevance": base_answer_rel,
                "Answer Relevance Std": base_answer_rel_std,
                "Answer Relevance Count": base_answer_rel_count,
                "Context Relevance": np.nan,
                "Context Relevance Std": np.nan,
                "Context Relevance Count": 0,
                "Groundedness": np.nan,
                "Groundedness Std": np.nan,
                "Groundedness Count": 0,
                "Citation Hit Rate": base_citation_rate,
                "Citation Samples": base_citation_samples,
            },
            {
                "mode": "Base+Prompt",
                "Answer Relevance": base_prompt_answer_rel,
                "Answer Relevance Std": base_prompt_answer_rel_std,
                "Answer Relevance Count": base_prompt_answer_rel_count,
                "Context Relevance": np.nan,
                "Context Relevance Std": np.nan,
                "Context Relevance Count": 0,
                "Groundedness": np.nan,
                "Groundedness Std": np.nan,
                "Groundedness Count": 0,
                "Citation Hit Rate": base_prompt_citation_rate,
                "Citation Samples": base_prompt_citation_samples,
            },
            {
                "mode": "RAG",
                "Answer Relevance": rag_answer_rel,
                "Answer Relevance Std": rag_answer_rel_std,
                "Answer Relevance Count": rag_answer_rel_count,
                "Context Relevance": rag_context_rel,
                "Context Relevance Std": rag_context_rel_std,
                "Context Relevance Count": rag_context_rel_count,
                "Groundedness": rag_groundedness,
                "Groundedness Std": rag_groundedness_std,
                "Groundedness Count": rag_groundedness_count,
                "Citation Hit Rate": rag_citation_rate,
                "Citation Samples": rag_citation_samples,
            },
        ]
    )
    summary.to_csv("./eval/metrics_summary.csv", index=False, encoding="utf-8-sig")

    _plot_and_save_bar(summary, "./eval/fig5_1.png")
    return df


if __name__ == "__main__":
    df = run_eval()
    print("[OK] Evaluation finished. See ./eval/results.csv, ./eval/metrics_summary.csv, ./eval/fig5_1.png")
