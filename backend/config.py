from dataclasses import dataclass
import os
import tempfile
from pathlib import Path
from typing import List
import warnings

def _ensure_tempdir() -> str:
    """
    Guarantee that tempfile has at least one writable directory.
    This prevents third-party libraries (dill / TruLens) from crashing
    during import when /tmp 不可寫。
    """
    if os.getenv("RAG_QA_SKIP_TEMP_CHECK"):
        fallback = Path(os.getenv("TMPDIR") or "/tmp")
        tempfile.tempdir = str(fallback)
        for env_name in ("TMPDIR", "TEMP", "TMP"):
            os.environ.setdefault(env_name, str(fallback))
        warnings.warn(
            "RAG_QA_SKIP_TEMP_CHECK=1 已啟用：略過暫存路徑寫入檢查，"
            "若環境依然不可寫，TruLens 相關功能可能失敗。"
        )
        return str(fallback)

    def _add_candidate(value, bucket, seen):
        if not value:
            return
        path = Path(value).expanduser()
        if path in seen:
            return
        seen.add(path)
        bucket.append(path)

    candidates: List[Path] = []
    seen: set[Path] = set()

    for env_name in ("TMPDIR", "TEMP", "TMP"):
        _add_candidate(os.getenv(env_name), candidates, seen)

    try:
        _add_candidate(Path(tempfile.gettempdir()), candidates, seen)
    except FileNotFoundError:
        pass

    for path in [
        Path("/tmp"),
        Path("/var/tmp"),
        Path("/usr/tmp"),
        Path.cwd() / ".tmp",
        Path.cwd() / "chroma_db",
        Path.cwd() / "eval",
        Path.home() / ".cache" / "rag_qa_tmp",
    ]:
        _add_candidate(path, candidates, seen)

    errors: List[str] = []
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".rag_qa_tmp"
            with open(probe, "w", encoding="utf-8") as fp:
                fp.write("ok")
            probe.unlink()
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
            continue

        for env_name in ("TMPDIR", "TEMP", "TMP"):
            os.environ.setdefault(env_name, str(candidate))
        tempfile.tempdir = str(candidate)
        return str(candidate)

    detail = "; ".join(errors) if errors else "找不到可寫入的暫存資料夾"
    raise RuntimeError(
        "無法建立可寫入的臨時資料夾，請設定 TMPDIR/TEMP/TMP 至可寫路徑。"
        f"（嘗試的路徑：{detail}）"
    )


_TEMP_DIR = _ensure_tempdir()


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _try_prepare_dir(candidate: Path) -> bool:
    """
    Ensure the candidate directory is writable by creating it (if needed)
    and writing a short test file.
    """
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        test_file = candidate / ".write_test"
        with open(test_file, "w", encoding="utf-8") as fp:
            fp.write("ok")
        if test_file.exists():
            test_file.unlink()
        return True
    except OSError as exc:
        warnings.warn(f"無法使用向量儲存路徑 {candidate}: {exc}")
        return False


def _resolve_chroma_dir(preferred: str) -> str:
    """
    Try to locate a writable directory for Chroma persistence.
    Priority:
      1. User provided CHROMA_DIR
      2. CHROMA_FALLBACK_DIR (optional)
      3. $TMPDIR / system temp
      4. ~/.cache/rag_qa_chroma_db
    """
    candidates: List[Path] = []
    if preferred:
        candidates.append(Path(preferred).expanduser())
    fallback_env = os.getenv("CHROMA_FALLBACK_DIR")
    if fallback_env:
        candidates.append(Path(fallback_env).expanduser())
    candidates.append(Path(tempfile.gettempdir()) / "rag_qa_chroma_db")
    candidates.append(Path.home() / ".cache" / "rag_qa_chroma_db")

    for candidate in candidates:
        if _try_prepare_dir(candidate):
            return str(candidate)

    raise RuntimeError(
        "無法建立可寫入的 Chroma 儲存資料夾，"
        "請設定環境變數 CHROMA_DIR 或 CHROMA_FALLBACK_DIR 至可寫入的位置。"
    )


@dataclass
class AppConfig:
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "https://iard.liontravel.com/ollama")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.3:70b")
    OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    HF_EMBEDDING_MODEL: str = os.getenv("HF_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    HF_EMBEDDING_DEVICE: str = os.getenv("HF_EMBEDDING_DEVICE", "cpu")
    HF_EMBEDDING_BATCH: int = int(os.getenv("HF_EMBEDDING_BATCH", "16"))
    HF_EMBEDDING_MAX_LENGTH: int = int(os.getenv("HF_EMBEDDING_MAX_LENGTH", "512"))
    HF_EMBEDDING_NORMALIZE: bool = _env_bool("HF_EMBEDDING_NORMALIZE", "true")
    HF_EMBEDDING_QUERY_PREFIX: str = os.getenv("HF_EMBEDDING_QUERY_PREFIX", "")
    HF_EMBEDDING_DOC_PREFIX: str = os.getenv("HF_EMBEDDING_DOC_PREFIX", "")
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_qa_creativity")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "750"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
    DETECTION_THRESHOLD: float = float(os.getenv("DETECTION_THRESHOLD", "0.25"))

    def __post_init__(self):
        self.CHROMA_DIR = _resolve_chroma_dir(self.CHROMA_DIR)
        if "e5" in (self.HF_EMBEDDING_MODEL or "").lower():
            if not os.getenv("HF_EMBEDDING_QUERY_PREFIX"):
                self.HF_EMBEDDING_QUERY_PREFIX = "query: "
            if not os.getenv("HF_EMBEDDING_DOC_PREFIX"):
                self.HF_EMBEDDING_DOC_PREFIX = "passage: "


cfg = AppConfig()
