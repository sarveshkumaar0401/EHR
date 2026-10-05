import os

from google import genai
from google.genai import types
from google.genai.errors import ClientError

from data_loader import embed_texts, load_and_chunk_dataset
from vector_db import DEFAULT_STORAGE_PATH, QdrantStorage

QDRANT_STORAGE_PATH = str(DEFAULT_STORAGE_PATH)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
gemini_client = genai.Client() if GEMINI_API_KEY else None


def retrieve(
    query: str,
    patient_id: str = None,
    symptom_filter: list = None,
    top_k: int = 4,
    collection: str = "aci_bench_clinical_notes",
):
    """Pure RAG retrieval against the Qdrant vector DB."""
    store = QdrantStorage(collection=collection, dim=384, path=QDRANT_STORAGE_PATH)
    query_vector = embed_texts([query])[0]
    return store.search(query_vector, patient_id=patient_id, symptom_filter=symptom_filter, top_k=top_k)


def answer_query(
    user_query: str,
    patient_id: str = None,
    symptom_filter: list = None,
    top_k: int = 4,
    show_context: bool = True,
    collection: str = "aci_bench_clinical_notes",
):
    """Return grounded answer from retrieved clinical-note chunks only."""
    results = retrieve(
        user_query,
        patient_id=patient_id,
        symptom_filter=symptom_filter,
        top_k=top_k,
        collection=collection,
    )

    if not results["contexts"]:
        return "No relevant clinical records matching the query and filter constraints."

    context_chunks = []
    if show_context:
        print("\n--- Retrieved Context Chunks ---")
    for i, context in enumerate(results["contexts"]):
        if show_context:
            print(f"[{i + 1}] Patient: {results['patient_ids'][i] if i < len(results['patient_ids']) else 'N/A'} | Doc: {results['doc_ids'][i] if i < len(results['doc_ids']) else 'N/A'}")
            print(f"    Snippet: {context[:120]}...\n")
        context_chunks.append(
            f"[Patient: {results['patient_ids'][i] if i < len(results['patient_ids']) else 'N/A'} | "
            f"Doc: {results['doc_ids'][i] if i < len(results['doc_ids']) else 'N/A'}]\n{context}"
        )

    context_str = "\n\n".join(context_chunks)
    system_instruction = (
        "You are an expert clinical reasoning assistant. Formulate your findings "
        "strictly based on the structured clinical note excerpts provided. "
        "If the excerpts do not contain the answer, say so and do not infer it. "
        "Cite the Doc ID and Patient ID for every clinical claim made."
    )
    prompt = f"Retrieved Context:\n{context_str}\n\nClinical Question: {user_query}"

    if gemini_client:
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0,
                ),
            )
            return response.text
        except ClientError as exc:
            if getattr(exc, "status_code", None) == 429:
                print("Gemini quota exhausted. Falling back to retrieved context only.")
                return "GEMINI quota exhausted. Retrieved context:\n\n" + context_str
            print(f"Gemini request failed: {exc}")
            return "Gemini request failed. Retrieved context:\n\n" + context_str
        except Exception as exc:
            print(f"Gemini request failed: {exc}")
            return "Gemini request failed. Retrieved context:\n\n" + context_str

    return "GEMINI_API_KEY missing. Extracted context block:\n\n" + context_str


def run_pipeline():
    """Create the vector DB from the ACI-Bench notes and run a basic retrieval demo."""
    store = QdrantStorage(collection="aci_bench_clinical_notes", dim=384, path=QDRANT_STORAGE_PATH)
    store.reset_collection()

    chunks = load_and_chunk_dataset()
    store.index_documents(chunks)

    query = "Assessment and plan for patient presenting with dyspnea and cough"
    print(f"\n[RAG demo] Query: '{query}'")
    answer = answer_query(query, top_k=2)
    print("\n--- Response ---")
    print(answer)


if __name__ == "__main__":
    run_pipeline()

