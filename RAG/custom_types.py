from typing import Any, List, Optional

import pydantic


class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[dict[str, Any]]
    source_id: Optional[str] = None


class RAGUpsertResult(pydantic.BaseModel):
    ingested: int


class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str] = []
    patient_ids: list[str] = []
    doc_ids: list[str] = []
    canonical_symptoms: list[str] = []


class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str] = []
    num_contexts: int = 0
    contexts: list[str] = []
    patient_ids: list[str] = []
    doc_ids: list[str] = []