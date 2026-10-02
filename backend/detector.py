from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from backend.config import cfg
from backend.embeddings import build_embeddings_with_fallback


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


TRAIT_LIBRARY: Dict[str, Dict[str, Any]] = {
    "好奇性": {
        "description": (
            "指個人對新事物和未知領域的探索慾望，喜歡觀察和了解事物背後的真相。"
            "富有追根究柢與多元求知精神，樂於投入曖昧或開放的情境，"
            "願意深入思索事物奧妙並主動觀察特殊現象以尋找答案，課後仍鍥而不捨地追問\"為什麼\"。"
        ),
        "subfacets": ["追根究柢", "探索慾望", "了解真相", "觀察"],
        "keywords": [
            {"term": "好奇性", "weight": 0.06, "facet": "探索慾望"},
            {"term": "好奇心", "weight": 0.05, "facet": "追根究柢"},
            {"term": "追根究柢", "weight": 0.06, "facet": "追根究柢"},
            {"term": "問", "weight": 0.05, "facet": "追根究柢"},
            {"term": "為什麼", "weight": 0.05, "facet": "追根究柢"},
            {"term": "提問", "weight": 0.05, "facet": "追根究柢"},
            {"term": "思考", "weight": 0.05, "facet": "了解真相"},
            {"term": "觀察", "weight": 0.05, "facet": "觀察"},
            {"term": "探究", "weight": 0.05, "facet": "探索慾望"},
            {"term": "求知", "weight": 0.04, "facet": "探索慾望"},
            {"term": "找答案", "weight": 0.04, "facet": "了解真相"},
            {"term": "新事物", "weight": 0.04, "facet": "探索慾望"},
            {"term": "未知", "weight": 0.04, "facet": "探索慾望"},
        ],
    },
    "想像力": {
        "description": (
            "指個人能夠構思、幻想以及將想法視覺化的能力。"
            "擅長視覺化與建立心像，能幻想尚未發生的情節，"
            "運用直覺推測並超越現實限制來生成新構想，細節生動如描寫飄浮城市與會說話的建築。"
        ),
        "subfacets": ["視覺化", "幻想", "構思", "超現實"],
        "keywords": [
            {"term": "想像力", "weight": 0.06, "facet": "幻想"},
            {"term": "豐富", "weight": 0.05, "facet": "幻想"},
            {"term": "天馬行空", "weight": 0.05, "facet": "幻想"},
            {"term": "生動", "weight": 0.04, "facet": "視覺化"},
            {"term": "推測", "weight": 0.04, "facet": "構思"},
            {"term": "超現實", "weight": 0.04, "facet": "超現實"},
            {"term": "故事", "weight": 0.04, "facet": "幻想"},
        ],
    },
    "挑戰性": {
        "description": (
            "指個人願意接受挑戰，不墨守成規，並喜歡解決複雜問題的特質。 "
            "主動尋找多種可能性並比較現實與理想差距，"
            "能在雜亂線索中理出秩序，願意鑽研複雜問題並持續調整，完成基本要求後仍樂於挑戰更難的任務。"
        ),
        "subfacets": ["不墨守成規", "樂於挑戰",  "探究複雜問題"],
        "keywords": [
            {"term": "挑戰性", "weight": 0.06, "facet": "樂於挑戰"},
            {"term": "樂於挑戰", "weight": 0.05, "facet": "樂於挑戰"},
            {"term": "自我挑戰", "weight": 0.05, "facet": "樂於挑戰"},
            {"term": "困難", "weight": 0.05, "facet": "探究複雜問題"},
            {"term": "任務", "weight": 0.04, "facet": "探究複雜問題"},
            {"term": "可能性", "weight": 0.04, "facet": "不墨守成規"},
            {"term": "突破", "weight": 0.04, "facet": "不墨守成規"},
            {"term": "複雜", "weight": 0.04, "facet": "探究複雜問題"},
        ],
    },
    "冒險性": {
        "description": (
            "指個人願意嘗試新事物、不怕失敗的特質。"
            "勇於承擔失敗與批評，敢於猜測並在不確定與混亂中完成任務。"
            "會為自己的觀點據理力爭並樂於嘗試新挑戰，常自告奮勇成為第一個嘗試高風險實驗的人。"
        ),
        "subfacets": ["面對失敗", "勇於嘗試", "辯護觀點"],
        "keywords": [
            {"term": "冒險性", "weight": 0.06, "facet": "面對失敗"},
            {"term": "冒險精神", "weight": 0.05, "facet": "面對失敗"},
            {"term": "不怕失敗", "weight": 0.05, "facet": "面對失敗"},
            {"term": "挫折", "weight": 0.04, "facet": "面對失敗"},
            {"term": "嘗試", "weight": 0.05, "facet": "勇於嘗試"},
            {"term": "新事物", "weight": 0.04, "facet": "勇於嘗試"},
            {"term": "自告奮勇", "weight": 0.05, "facet": "勇於嘗試"},
            {"term": "風險", "weight": 0.04, "facet": "面對失敗"},
            {"term": "面對", "weight": 0.04, "facet": "面對失敗"},
            {"term": "爭取", "weight": 0.04, "facet": "辯護觀點"},
        ],
    },
}

TRAIT_DESCRIPTIONS: Dict[str, str] = {
    trait: data["description"] for trait, data in TRAIT_LIBRARY.items()
}


@dataclass
class DetectionResult:
    trait: str
    confidence: float
    keyword_hits: List[str]
    trait_scores: Dict[str, float]
    subfacets: List[str]

    def should_use_rag(self, threshold: float = 0.25) -> bool:
        return self.confidence >= threshold


class CreativityDetector:
    """
    粗略的創造力語意偵測器，結合嵌入餘弦相似度與關鍵字計分。
    """

    def __init__(
        self,
        embedding_model=None,
        similarity_threshold: float = 0.25,
        keyword_boost: float = 0.04,
    ):
        self.similarity_threshold = similarity_threshold
        self.keyword_boost = keyword_boost
        self.embeddings = embedding_model or build_embeddings()
        self.prototype_vectors = {
            trait: self.embeddings.embed_query(desc)
            for trait, desc in TRAIT_DESCRIPTIONS.items()
        }
        self.keyword_index: Dict[str, List[Dict[str, Any]]] = {
            trait: specs.get("keywords", [])
            for trait, specs in TRAIT_LIBRARY.items()
        }

    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-8
        return float(np.dot(va, vb) / denom)

    def _count_keywords(self, text: str) -> Dict[str, Dict[str, Any]]:
        hits: Dict[str, Dict[str, Any]] = {}
        lower = text.lower()
        for trait, specs in self.keyword_index.items():
            matched_terms: List[str] = []
            facets = set()
            score = 0.0
            for spec in specs:
                term = spec.get("term", "")
                if not term:
                    continue
                if term.lower() in lower:
                    matched_terms.append(term)
                    weight = spec.get("weight") or self.keyword_boost
                    score += weight
                    facet = spec.get("facet")
                    if facet:
                        facets.add(facet)
            if matched_terms:
                hits[trait] = {
                    "keywords": matched_terms,
                    "score": round(score, 4),
                    "facets": sorted(facets),
                }
        return hits

    def detect(self, text: str) -> DetectionResult:
        if not text.strip():
            return DetectionResult(
                trait="未知",
                confidence=0.0,
                keyword_hits=[],
                trait_scores={},
                subfacets=[],
            )

        query_vec = self.embeddings.embed_query(text)
        keyword_hits = self._count_keywords(text)

        trait_scores: Dict[str, float] = {}
        for trait, proto_vec in self.prototype_vectors.items():
            score = self._cosine_sim(query_vec, proto_vec)
            if trait in keyword_hits:
                score += keyword_hits[trait]["score"]
            trait_scores[trait] = round(score, 4)

        trait, confidence = max(trait_scores.items(), key=lambda x: x[1])
        final_trait = trait
        if confidence < self.similarity_threshold:
            final_trait = "未知"

        hits: List[str] = []
        subfacets: List[str] = []
        if final_trait != "未知" and final_trait in keyword_hits:
            hits = keyword_hits[final_trait]["keywords"]
            subfacets = keyword_hits[final_trait]["facets"]

        return DetectionResult(
            trait=final_trait,
            confidence=confidence,
            keyword_hits=hits,
            trait_scores=trait_scores,
            subfacets=subfacets,
        )
