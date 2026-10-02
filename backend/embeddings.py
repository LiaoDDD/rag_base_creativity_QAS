from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence
import warnings

import torch
import torch.nn.functional as F
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModel, AutoTokenizer


def _chunk_iterable(items: Sequence[str], chunk_size: int) -> Iterable[List[str]]:
    for idx in range(0, len(items), chunk_size):
        yield list(items[idx : idx + chunk_size])


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@dataclass
class SafeHFTransformersEmbeddings(Embeddings):
    """
    Lightweight embedding backend that uses `transformers.AutoModel`
    directly so it does not rely on `SentenceTransformer.to()`, which
    currently crashes on PyTorch 2.9 with meta tensors.
    """

    model_name: str
    device: str = "cpu"
    batch_size: int = 8
    max_length: int = 512
    normalize: bool = True
    query_prefix: str = ""
    document_prefix: str = ""

    def __post_init__(self) -> None:
        self._device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        if self._device.type != "cpu":
            self._model.to(self._device)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        cleaned = [t.replace("\n", " ").strip() for t in texts]
        outputs: List[torch.Tensor] = []
        for batch in _chunk_iterable(cleaned, self.batch_size):
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                last_hidden_state = self._model(**enc).last_hidden_state
            pooled = _mean_pool(last_hidden_state, enc["attention_mask"])
            if self.normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
            outputs.append(pooled.cpu())
        if not outputs:
            return []
        return torch.cat(outputs, dim=0).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"{self.document_prefix}{t}" if self.document_prefix else t for t in texts]
        return self._encode(prefixed)

    def embed_query(self, text: str) -> List[float]:
        value = f"{self.query_prefix}{text}" if self.query_prefix else text
        result = self._encode([value])
        return result[0] if result else []


def build_embeddings_with_fallback(
    *,
    model_name: str,
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 512,
    normalize: bool = True,
    query_prefix: str = "",
    document_prefix: str = "",
) -> Embeddings:
    """
    Try HuggingFaceEmbeddings first (fast path). If torch hits the meta tensor bug,
    fall back to the lightweight SafeHFTransformersEmbeddings.
    """
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": normalize,
                "batch_size": batch_size,
            },
        )
    except NotImplementedError as exc:
        message = str(exc).lower()
        if "meta tensor" not in message:
            raise
        warnings.warn(
            "HuggingFaceEmbeddings failed due to meta tensors; "
            "falling back to SafeHFTransformersEmbeddings."
        )
    return SafeHFTransformersEmbeddings(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        normalize=normalize,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
    )
