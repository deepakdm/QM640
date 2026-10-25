"""Typed experiment configuration loaded from configs/*.yaml."""
from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional
import yaml
from pydantic import BaseModel


class ChunkingCfg(BaseModel):
    strategy: Literal["fixed", "section"] = "fixed"
    chunk_size: int = 512
    overlap: int = 64


class RetrievalCfg(BaseModel):
    # "none" is the experimental CONTROL: the LLM answers with no retrieved context at all.
    # It isolates the effect of retrieval, which is what the primary hypothesis tests.
    mode: Literal["none", "dense", "bm25", "hybrid"] = "dense"
    top_k: int = 5
    rerank: bool = False
    rerank_model: Optional[str] = None
    hybrid_alpha: float = 0.5


class GenerationCfg(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 800
    require_citations: bool = True


class EvaluationCfg(BaseModel):
    runs_per_case: int = 3
    judge_sample_size: int = 40


class ExperimentConfig(BaseModel):
    name: str
    corpus_dir: Path
    gold_path: Path
    chunking: ChunkingCfg = ChunkingCfg()
    retrieval: RetrievalCfg = RetrievalCfg()
    generation: GenerationCfg = GenerationCfg()
    evaluation: EvaluationCfg = EvaluationCfg()

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))
