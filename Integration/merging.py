import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings


# 1. Setup Environment & Gemini LLMs
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in Integration/.env")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# We use Gemini 2.5 Pro for the complex task of writing Cypher queries
cypher_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0
)

# We use Gemini 2.5 Flash for rapid final answer synthesis
qa_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
# 2. Connect to Neo4j (Knowledge Graph)
# This automatically reads your schema from your Docker instance!
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    username=os.getenv("NEO4J_USER", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "")
)

# 3. Setup GraphCypherQAChain for Graph Retrieval
# Use a strict custom Cypher prompt so the model follows the exact medical graph schema.
cypher_template = """You are an expert Neo4j Cypher translator for a medical audio and disease database.
Translate the user's question into a valid Cypher query.
Use ONLY the provided schema. Do not hallucinate node names, labels, or relationships.

Schema:
{schema}

Graph Traversal Rules:
- To find symptoms associated with a specific disease: (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
- To find diseases when given a specific symptom: (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
- To find recommended treatments for a disease: (d:Disease)-[:TREATED_BY]->(t:Treatment)

IMPORTANT STRICT RULES:
- The valid node labels are exactly: Disease, Symptom, Treatment, Anatomy, Trigger, Category. Do not invent others (e.g., never use Medication, Patient, or Condition).
- The valid relationship types for medical conditions are exactly: [:HAS_SYMPTOM] and [:TREATED_BY]. Never invent edges.
- Never use complex OR statements for labels. For example, use `t:Treatment`, not `t:Treatment OR t:Medication`.
- Always use `toLower()` when matching string names (e.g., `WHERE toLower(d.name) CONTAINS 'encephalitis'`), as the database properties are stored in lowercase.
- Always use directional arrows based on the traversal rules above.database stores them in lowercase.

Question: {question}
Cypher query:"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template=cypher_template,
)

graph_chain = GraphCypherQAChain.from_llm(
    cypher_llm=cypher_llm,
    qa_llm=qa_llm,
    graph=graph,
    verbose=True,
    return_direct=True,
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True
)

# 4. Connect to Qdrant (Vector Database)
# Use the already-created local RAG vector DB under RAG/qdrant_aci_db_v2 rather than
# assuming the remote default localhost Qdrant instance has the right collections loaded.
RAG_DIR = Path(__file__).resolve().parents[1] / "RAG"
QDRANT_STORAGE_PATH = os.getenv("QDRANT_STORAGE_PATH", str(RAG_DIR / "qdrant_aci_db_v2"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
PATIENT_COLLECTION = os.getenv("PATIENT_COLLECTION", "aci_bench_clinical_notes")
MEDICAL_COLLECTION = os.getenv("MEDICAL_COLLECTION", "aci_bench_graph_ready")
LEGACY_COLLECTION = os.getenv("LEGACY_COLLECTION", "docs")

use_local_qdrant = os.getenv("USE_LOCAL_QDRANT", "true").lower() == "true"
if use_local_qdrant:
    qdrant_client = QdrantClient(path=QDRANT_STORAGE_PATH, timeout=30)
else:
    qdrant_client = QdrantClient(url=QDRANT_URL, check_compatibility=False)


def _ensure_qdrant_collection(collection_name: str):
    """Create the collection if it does not exist, which prevents import-time crashes on empty Qdrant instances."""
    if qdrant_client.collection_exists(collection_name=collection_name):
        return

    print(f"[Info] Creating missing Qdrant collection: {collection_name}")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        ),
    )


def _make_qdrant_store(collection_name: str):
    _ensure_qdrant_collection(collection_name)
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
        content_payload_key="text"
    )


qdrant_stores = {
    PATIENT_COLLECTION: _make_qdrant_store(PATIENT_COLLECTION),
    MEDICAL_COLLECTION: _make_qdrant_store(MEDICAL_COLLECTION),
    LEGACY_COLLECTION: _make_qdrant_store(LEGACY_COLLECTION),
}

# Create a standard vector retriever to pull the top 3 most relevant text chunks
qdrant_retrievers = {
    name: store.as_retriever(search_kwargs={"k": 3}) for name, store in qdrant_stores.items()
}

# 5. Define Context Formatting Functions
def _extract_search_terms(question: str):
    """Extract meaningful medical keywords from the clinician query."""
    terms = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", question)
    stop_words = {
        "what", "where", "when", "which", "whose", "their", "there", "then",
        "about", "patient", "patients", "speaker", "speakers", "transcript",
        "transcripts", "audio", "observations", "clinical", "query", "doctor",
        "clinician", "doctors", "said", "says", "tell", "find", "found",
        "related", "linked", "association", "associated", "between", "using",
        "are", "use", "used", "first", "return", "only", "entities",
        "relationships", "graph-backed", "backed",
        "disease", "symptom", "symptoms", "treatment", "treatments", "therapy",
        "medical", "medication", "medications", "diagnosis", "diagnose",
        "graph", "database", "retrieval", "context", "answer", "information",
        "and", "or", "the", "a", "an", "is", "in", "on", "of", "for",
        "with", "have", "has", "into", "from", "this", "these", "those",
        "question", "questions", "report", "reported", "story", "history"
    }
    cleaned = []
    for term in terms:
        term_l = term.lower().strip("'\".,!?()[]{}")
        if len(term_l) > 3 and term_l not in stop_words:
            cleaned.append(term_l)
    return cleaned


def get_graph_context(question: str, medical_lookup: bool = False) -> str:
    """Use exact symptom matches, with an optional strict medical lookup mode."""
    search_terms = _extract_search_terms(question)

    try:
        disease_query = """
        MATCH (d:Disease)
        WHERE any(term IN $terms WHERE toLower(d.name) = term OR toLower(d.name) CONTAINS term)
        OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom)
        OPTIONAL MATCH (d)-[:TREATED_BY]->(t:Treatment)
        RETURN d.name AS disease,
               collect(DISTINCT s.name) AS symptoms,
               collect(DISTINCT t.name) AS treatments
        ORDER BY disease
        LIMIT 50
        """
        rows = graph.query(disease_query, params={"terms": search_terms})
        if rows:
            return str(rows)

        symptom_query = """
        MATCH (s:Symptom)
        WHERE any(term IN $terms WHERE toLower(s.name) = term OR toLower(s.name) CONTAINS term)
        OPTIONAL MATCH (d:Disease)-[:HAS_SYMPTOM]->(s)
        RETURN s.name AS symptom,
               collect(DISTINCT d.name) AS diseases
        ORDER BY symptom
        LIMIT 50
        """
        rows = graph.query(symptom_query, params={"terms": search_terms})
        if rows:
            return str(rows)
    except Exception as e:
        print(f"[Warning] Focused medical graph lookup failed: {e}")

    try:
        result = graph_chain.invoke({"query": question})
        graph_result = result.get("result", "") if isinstance(result, dict) else result
        if graph_result:
            return str(graph_result)
    except Exception as e:
        print(f"[Warning] Graph retrieval failed: {e}")

    if medical_lookup:
        return "No disease or treatment relationships found for the extracted symptoms."

    try:
        speaker_match = re.search(r"Speaker\s*[:#]?\s*['\"]?([A-Za-z0-9]+)['\"]?", question, flags=re.IGNORECASE)
        if speaker_match:
            speaker_id = speaker_match.group(1)
            fallback_query = """
            MATCH (s:Speaker {id: $speaker_id})
            OPTIONAL MATCH (s)-[:SPOKE]->(a:AudioSample)
            OPTIONAL MATCH (a)-[:MENTIONS_SYMPTOM]->(sym:Symptom)
            RETURN s.id AS speaker_id,
                   collect(DISTINCT a.sentence) AS transcripts,
                   collect(DISTINCT sym.name) AS symptoms
            """
            rows = graph.query(fallback_query, params={"speaker_id": speaker_id})
            if rows:
                return str(rows)

        if search_terms:
            fallback_query = """
            MATCH (s:Speaker)-[:SPOKE]->(a:AudioSample)
            WHERE any(term IN $terms WHERE toLower(a.sentence) CONTAINS term)
            OPTIONAL MATCH (a)-[:MENTIONS_SYMPTOM]->(sym:Symptom)
            RETURN s.id AS speaker_id,
                   a.sentence AS transcript,
                   collect(DISTINCT sym.name) AS symptoms
            LIMIT 20
            """
            rows = graph.query(fallback_query, params={"terms": search_terms})
            if rows:
                return str(rows)
    except Exception as e:
        print(f"[Warning] Direct graph fallback failed: {e}")

    return "No structured data found."


def format_vector_docs(docs) -> str:
    """Formats the unstructured text chunks from Qdrant into a single string."""
    if not docs:
        return ""
    return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)


def classify_question(question: str) -> str:
    """Classify the question so the retrieval layer chooses the right source first."""
    q = question.lower()
    graph_terms = [
        "disease", "symptom", "diagnosis", "treatment", "medication", "therapy",
        "drug", "condition", "diagnosed", "diagnose", "medicines", "plan",
        "treatments", "symptoms", "what disease", "what symptom", "what treatment"
    ]
    vector_terms = [
        "history", "patient history", "story", "timeline", "transcript", "speaker",
        "what happened", "when did", "how long", "describe", "narrative", "past",
        "previous", "called", "said", "reported", "observation", "clinical note"
    ]

    if any(term in q for term in graph_terms):
        return "graph"
    if any(term in q for term in vector_terms):
        return "vector"
    return "hybrid"


def asks_for_patient_context(question: str) -> bool:
    """Return whether the query asks for a person or patient-specific evidence."""
    return bool(re.search(
        r"\b(who|whom|patient|patients|speaker|speakers|person|individual|them|their)\b",
        question,
        flags=re.IGNORECASE,
    ))


def has_relevant_context(context: str) -> bool:
    if not context or not str(context).strip():
        return False
    normalized = str(context).strip().lower()
    if normalized in {
        "no structured data found.",
        "no disease or treatment relationships found for the extracted symptoms.",
        "no relevant patient context found.",
        "no relevant medical context found.",
        "no relevant graph context found.",
        "no patient-vector evidence needed because the graph contained the relevant medical answer.",
        "no medical-vector evidence needed because the graph contained the relevant medical answer.",
        "no graph-first answer required because the question is narrative/history-oriented.",
    }:
        return False
    if "no patient-vector evidence needed" in normalized or "no medical-vector evidence needed" in normalized:
        return False
    if "no graph-first answer required" in normalized:
        return False
    if normalized in {"[]", "none", "null"}:
        return False
    if any(marker in normalized for marker in (
        "no relevant context",
        "no structured data",
        "no disease or treatment relationships",
        "no information found",
        "no records found",
        "no results found",
        "not found in the graph",
    )):
        return False
    return True


def get_vector_context(question: str, collection_name: str = PATIENT_COLLECTION) -> str:
    """Get vector results first, then fallback to a keyword scan of the raw Qdrant payloads.

    The default collection is the patient dataset; the medical disease corpus should be
    queried via a separate collection to avoid mixing patient statements with disease facts.
    """
    collection_name = collection_name or PATIENT_COLLECTION
    retriever = qdrant_retrievers.get(collection_name)
    if retriever is None:
        retriever = qdrant_retrievers.get(LEGACY_COLLECTION)

    try:
        docs = retriever.invoke(question) if retriever else []
        if docs:
            vector_context = format_vector_docs(docs)
            if vector_context:
                return vector_context
    except Exception as e:
        print(f"[Warning] Vector retrieval failed for {collection_name}: {e}")

    try:
        search_terms = _extract_search_terms(question)
        if not search_terms:
            return ""

        hits = []
        scroll_collection = collection_name if collection_name in qdrant_stores else LEGACY_COLLECTION
        for item in qdrant_client.scroll(collection_name=scroll_collection, limit=100, with_payload=True, with_vectors=False)[0]:
            payload = item.payload or {}
            text = payload.get("text") or payload.get("page_content") or ""
            normalized = text.lower()
            if any(term in normalized for term in search_terms):
                hits.append(type("Doc", (), {"page_content": text})())
        if hits:
            return format_vector_docs(hits)
    except Exception as e:
        print(f"[Warning] Qdrant fallback failed for {collection_name}: {e}")

    return ""

# 6. The Synthesis Prompt
# This prompt forces the LLM to use BOTH sets of retrieved data to answer.
synthesis_prompt = ChatPromptTemplate.from_template(
    """You are an expert clinical AI assistant analyzing patient consultations and a dynamically extracted medical knowledge graph.
Answer the clinician's query using only the supplied knowledge graph and retrieval context.

Grounding rules:
- Treat the knowledge graph, patient vector chunks, and medical vector chunks as the only sources of truth.
- If both vector retrieval and graph retrieval have useful evidence, combine them into one grounded summary.
- If one source has no relevant answer, explicitly note that and rely on the source that does contain evidence.
- If neither source contains relevant evidence, say "No relevant context was found in the vector database or graph database."
- Do not invent entities, relationships, diagnoses, treatments, or patient facts.
- Distinguish clearly between explicit evidence and reasonable clinical interpretation.
- Do not provide a definitive diagnosis or personalized treatment plan; recommend clinician review where appropriate.

Source availability summary:
{source_status}

DYNAMIC MEDICAL KNOWLEDGE GRAPH CONTEXT (Entities may include patients, speakers, symptoms, diseases, anatomy, triggers, treatments, and other extracted concepts):
{graph_context}

PATIENT DATASET CONTEXT (verbatim statements, symptoms, triggers, speaker IDs, and audio observations):
{patient_context}

MEDICAL DATASET CONTEXT (disease, symptom, and treatment records stored separately from patient notes):
{medical_context}

Clinician's Query: {question}

Clinical Assessment and Findings:"""
)

# 7. Build the Sequential Hybrid Orchestrator
def run_true_hybrid_rag(question: str):
    question_type = classify_question(question)

    if question_type == "graph":
        print("--- 1. Prioritizing Neo4j graph retrieval for disease/symptom/treatment question ---")
        graph_context = get_graph_context(question, medical_lookup=True)
        graph_has_answer = has_relevant_context(graph_context)

        if not graph_has_answer or asks_for_patient_context(question):
            if graph_has_answer:
                print("--- 2. Querying Qdrant for patient-specific evidence requested by the question ---")
            else:
                print("--- 2. Neo4j had no relevant answer; falling back to Qdrant ---")
            patient_context = get_vector_context(question, PATIENT_COLLECTION)
            medical_context = get_vector_context(question, MEDICAL_COLLECTION)
        else:
            patient_context = ""
            medical_context = ""
    elif question_type == "vector":
        print("--- 1. Prioritizing Qdrant retrieval for narrative/history question ---")
        patient_context = get_vector_context(question, PATIENT_COLLECTION)
        medical_context = get_vector_context(question, MEDICAL_COLLECTION)
        graph_context = "No graph-first answer required because the question is narrative/history-oriented."
        graph_has_answer = False
    else:
        print("--- 1. Running Patient Vector Search (Qdrant) ---")
        patient_context = get_vector_context(question, PATIENT_COLLECTION)
        print("--- 2. Running Medical Vector Search (Qdrant) ---")
        medical_context = get_vector_context(question, MEDICAL_COLLECTION)
        print("--- 3. Running Graph Search (Neo4j) ---")
        graph_prompt = f"""
        Based on this extracted context:
        - patient notes: '{patient_context}'
        - medical records: '{medical_context}'
        Identify diseases, symptoms, and treatments in the graph that match the question.
        """
        graph_context = get_graph_context(graph_prompt, medical_lookup=True)
        graph_has_answer = has_relevant_context(graph_context)

    patient_has_answer = has_relevant_context(patient_context)
    medical_has_answer = has_relevant_context(medical_context)
    if question_type == "graph" and not graph_has_answer:
        graph_has_answer = False

    if patient_has_answer or medical_has_answer or graph_has_answer:
        if patient_has_answer and medical_has_answer and graph_has_answer:
            source_status = "All three sources contained relevant evidence; combine the vector and graph answers into one grounded summary."
        elif patient_has_answer and graph_has_answer:
            source_status = "Patient vector context and Neo4j graph context both contained relevant evidence; combine them."
        elif medical_has_answer and graph_has_answer:
            source_status = "Medical vector context and Neo4j graph context both contained relevant evidence; combine them."
        elif patient_has_answer and medical_has_answer:
            source_status = "Both Qdrant collections contained relevant evidence; combine them."
        elif graph_has_answer:
            source_status = "Neo4j graph contained the relevant answer; the vector DB did not have a sufficient match."
        elif patient_has_answer:
            source_status = "Qdrant patient context contained the relevant answer; the graph and medical vector sources did not."
        elif medical_has_answer:
            source_status = "Qdrant medical context contained the relevant answer; the graph and patient vector sources did not."
    else:
        source_status = "No relevant context was found in the vector database or the graph database."

    if question_type == "graph" and graph_has_answer and not asks_for_patient_context(question):
        print("--- 3. Neo4j answer selected as the primary source for this disease/symptom/treatment question ---")
    elif question_type == "graph" and graph_has_answer:
        print("--- 3. Combining Neo4j medical facts with Qdrant patient evidence ---")
    elif question_type == "graph":
        print("--- 3. Evaluating Qdrant fallback results ---")
    elif question_type == "vector":
        print("--- 2. Qdrant answer selected for this narrative/history question ---")
    else:
        print("--- 3. Evaluating both Qdrant and Neo4j for hybrid answer ---")

    print("--- 4. Synthesizing Final Answer ---")
    synthesis_chain = synthesis_prompt | qa_llm | StrOutputParser()
    return synthesis_chain.invoke({
        "question": question,
        "source_status": source_status,
        "patient_context": patient_context or "No relevant patient context found.",
        "medical_context": medical_context or "No relevant medical context found.",
        "graph_context": graph_context or "No relevant graph context found."
    })

# 8. Execute the Query
if __name__ == "__main__":
    doctor_question = (
        "History of Ms. Hernandez"
    )

    print("Starting Sequential Hybrid Retrieval...")
    final_answer = run_true_hybrid_rag(doctor_question)
    
    print("\n--- Final Grounded Answer ---")
    print(final_answer)