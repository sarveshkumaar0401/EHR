import os
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from data_loader import embed_texts

_CLIENT_CACHE = {}
DEFAULT_STORAGE_PATH = Path(__file__).resolve().parent / "qdrant_aci_db_v2"


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="aci_bench_clinical_notes", dim=384, path=None):
        self.collection = collection
        self.dim = dim
        self.storage_path = self._normalize_storage_path(path)
        os.makedirs(self.storage_path, exist_ok=True)

        cache_key = (self.storage_path, url)
        if cache_key in _CLIENT_CACHE:
            self.client = _CLIENT_CACHE[cache_key]
        else:
            if path and not str(path).startswith("http"):
                self.client = QdrantClient(path=self.storage_path)
            else:
                self.client = QdrantClient(url=url, timeout=30)
            _CLIENT_CACHE[cache_key] = self.client

        try:
            self.client.get_collection(collection_name=collection)
        except Exception:
            self.create_collection()

    @staticmethod
    def _normalize_storage_path(path):
        if path is None:
            return str(DEFAULT_STORAGE_PATH.resolve())

        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = (Path(__file__).resolve().parent / path_obj).resolve()
        return str(path_obj)

    def create_collection(self):
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="patient_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="canonical_symptoms",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="split",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    def reset_collection(self):
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.create_collection()

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def index_documents(self, documents, batch_size=128):
        print("Vectorizing and indexing records into Qdrant...")
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            texts = [b["text"] for b in batch]
            embeddings = embed_texts(texts)
            ids = [str(uuid.uuid4()) for _ in batch]
            payloads = [{"text": b["text"], **b["metadata"]} for b in batch]
            self.upsert(ids, embeddings, payloads)
            print(f"Indexed {min(i + batch_size, len(documents))}/{len(documents)} chunks.")

    def build_query_filter(self, patient_id=None, symptom_filter=None):
        must_filters = []
        if patient_id:
            must_filters.append(FieldCondition(key="patient_id", match=MatchValue(value=patient_id)))
        if symptom_filter:
            must_filters.append(FieldCondition(key="canonical_symptoms", match=MatchAny(any=symptom_filter)))
        return Filter(must=must_filters) if must_filters else None

    def search(self, query_vector, patient_id=None, symptom_filter=None, top_k=4):
        query_filter = self.build_query_filter(patient_id=patient_id, symptom_filter=symptom_filter)
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        contexts = []
        patient_ids = []
        doc_ids = []
        symptoms = set()

        for r in result.points:
            payload = r.payload or {}
            text = payload.get("text", "")
            if text:
                contexts.append(text)
            patient_id_value = payload.get("patient_id")
            if patient_id_value:
                patient_ids.append(patient_id_value)
            doc_id = payload.get("doc_id")
            if doc_id:
                doc_ids.append(doc_id)
            for symptom in payload.get("canonical_symptoms", []) or []:
                symptoms.add(symptom)

        return {
            "contexts": contexts,
            "patient_ids": patient_ids,
            "doc_ids": doc_ids,
            "canonical_symptoms": list(symptoms),
        }