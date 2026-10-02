import os
import re
import shutil
from typing import List, Dict, Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from docx import Document

from backend.config import cfg
from backend.rag_pipeline import VectorStoreManager


def load_williams_markdown(path: str, data_dir: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    rel_source = os.path.relpath(path, start=data_dir)
    current_header = None
    buffer: List[str] = []

    def flush_block(header: Optional[str], lines: List[str]):
        if not header:
            return None
        entry = build_williams_entry(header, lines, rel_source, path)
        if entry:
            entries.append(entry)

    with open(path, "r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.rstrip("\n")
            if line.startswith("## "):
                flush_block(current_header, buffer)
                current_header = line.strip()
                buffer = []
                continue
            if current_header:
                buffer.append(line)
    flush_block(current_header, buffer)
    return entries


def build_williams_entry(header: str, lines: List[str], rel_source: str, abs_source: str) -> Optional[Dict[str, Any]]:
    clean = header.lstrip("# ").strip()
    parts = [p.strip() for p in clean.split("｜") if p.strip()]
    if not parts or not parts[0].startswith("題項"):
        return None
    number = re.sub(r"\D", "", parts[0])
    item_id = f"Q{number.zfill(2)}" if number else parts[0]
    trait = parts[1] if len(parts) > 1 else "未註記"
    polarity = parts[2] if len(parts) > 2 else "unspecified"

    def extract_value(label: str) -> str:
        prefix = f"- **{label}**："
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(prefix):
                return stripped.split("：", 1)[1].strip()
        return ""

    anchor = extract_value("錨點")
    source_detail = extract_value("來源")
    statement = extract_value("文本")
    if not statement:
        statement = "\n".join([ln.strip() for ln in lines if ln.strip()]).strip()
    if not statement:
        return None

    text_parts = [
        f"威廉斯創造力傾向量表｜{trait}｜{item_id}（{polarity}）",
    ]
    if anchor:
        text_parts.append(f"錨點：{anchor}")
    if source_detail:
        text_parts.append(f"來源：{source_detail}")
    text_parts.append(f"題項內容：{statement}")
    metadata = {
        "source": rel_source,
        "source_type": "williams_markdown",
        "scale": "威廉斯創造力傾向量表",
        "trait": trait,
        "item_id": item_id,
        "polarity": polarity,
    }
    return {"text": "\n".join(text_parts), "metadata": metadata, "structured": True, "source": abs_source}

def read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def load_files_from_dir(data_dir: str) -> List[Dict[str, Any]]:
    texts: List[Dict[str, Any]] = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            ext = os.path.splitext(fn)[1].lower()
            try:
                if ext == ".md" and os.path.basename(fp) == "williams_rag_corpus.md":
                    entries = load_williams_markdown(fp, data_dir)
                    if entries:
                        texts.extend(entries)
                        continue
                if ext in [".txt", ".md"]:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                elif ext in [".pdf"]:
                    loader = PyPDFLoader(fp)
                    pages = loader.load()
                    content = "\n".join([p.page_content for p in pages])
                elif ext in [".docx"]:
                    content = read_docx(fp)
                else:
                    continue
                texts.append({"text": content, "source": fp})
            except Exception as e:
                print(f"[WARN] 無法載入 {fp}: {e}")
    return texts

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or cfg.CHUNK_SIZE,
        chunk_overlap=overlap or cfg.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
    )
    return [c.page_content for c in splitter.create_documents([text])]

def reset_vector_store(persist_dir: str):
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)

def ingest_directory(
    data_dir: str,
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
    reset: bool = False,
) -> VectorStoreManager:
    persist_dir = persist_dir or cfg.CHROMA_DIR
    if reset:
        reset_vector_store(persist_dir)
    vs = VectorStoreManager(persist_dir=persist_dir, collection_name=collection_name)
    raw = load_files_from_dir(data_dir)
    for item in raw:
        if item.get("structured"):
            meta = item.get("metadata") or {}
            if "source" not in meta:
                rel = os.path.relpath(item.get("source", ""), start=data_dir)
                meta["source"] = rel
            vs.add_texts([item["text"]], [meta])
            continue
        chunks = chunk_text(item["text"])
        rel = os.path.relpath(item["source"], start=data_dir)
        source_type = "docx" if rel.endswith(".docx") else "pdf" if rel.endswith(".pdf") else "text"
        metas = [{"source": rel, "source_type": source_type} for _ in chunks]
        vs.add_texts(chunks, metas)
    return vs
