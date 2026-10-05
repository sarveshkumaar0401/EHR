# Graph-ready ACI-Bench ingestion

Run from the `RAG` directory to index the ACI-Bench `virtassist` training split
into the separate `aci_bench_graph_ready` collection:

```powershell
uv run python ingest_graph_ready.py
```

The collection uses `RAG/qdrant_aci_db_v2` and preserves the
`aci_bench_clinical_notes` collection. Stable point IDs make repeated runs
upsert the same chunks instead of creating duplicates. To rebuild only the
graph-ready collection from scratch:

```powershell
uv run python ingest_graph_ready.py --replace
```

The Qdrant payload stores each chunk's `patient_id`, `doc_id`,
`canonical_symptoms`, and `pertinent_negatives` for graph handoff and filtering.

## Local Qdrant storage recovery

The application now uses `RAG/qdrant_aci_db_v2` because the legacy
`RAG/qdrant_aci_db` metadata is incompatible with the installed Qdrant client.
The legacy directory is left untouched. Re-ingest the collections you need
before querying them:

```powershell
uv run python main.py
uv run python ingest_graph_ready.py
```
