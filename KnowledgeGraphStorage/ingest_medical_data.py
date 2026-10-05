import os
import re
import time
from pathlib import Path
from typing import List
from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from neo4j import GraphDatabase
from pypdf import PdfReader  # ADDED: The PDF reader library

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / "Integration" / ".env", override=True)
load_dotenv(BASE_DIR / "RAG" / ".env", override=False)
load_dotenv(Path(__file__).with_name(".env"), override=False)
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"), 
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD"))
)
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
    temperature=0,
)

# 1. THE UNIVERSAL SCHEMA
class Node(BaseModel):
    label: str = Field(description="The category of the entity (e.g., Symptom, Disease, CarPart, JobTitle)")
    name: str = Field(description="The actual value of the entity (e.g., cough, asthma, spark plug, manager)")

class Relationship(BaseModel):
    source_name: str = Field(description="The name of the starting node")
    source_label: str = Field(description="The label of the starting node")
    relation_type: str = Field(description="The relationship action (e.g., HAS_SYMPTOM, USES_TOOL, REPORTS_TO). Use UPPERCASE_WITH_UNDERSCORES.")
    target_name: str = Field(description="The name of the target node")
    target_label: str = Field(description="The label of the target node")

class DynamicGraph(BaseModel):
    nodes: List[Node]
    relationships: List[Relationship]

universal_llm = llm.with_structured_output(DynamicGraph)

def sanitize_identifier(value: str, uppercase: bool = False) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", value.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if uppercase:
        sanitized = sanitized.upper()
    if not sanitized or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", sanitized):
        raise ValueError(f"Invalid Cypher identifier: {value!r}")
    return sanitized

# 2. THE PROCESSING FUNCTION
def ingest_text_to_graph(text: str):
    prompt = f"""Read the following disease-table rows and extract the medical knowledge exactly.
For each row, create a Disease node, one Symptom node for every symptom, and one Treatment node for every treatment.
Create these relationships whenever the row provides the corresponding information:
- (Disease)-[:HAS_SYMPTOM]->(Symptom)
- (Disease)-[:TREATED_BY]->(Treatment)
Use the complete disease, symptom, and treatment names from the text. Do not create relationships between unrelated rows.

TABLE ROWS:
{text}"""
    try:
        graph_data: DynamicGraph = universal_llm.invoke(prompt)
        with driver.session() as session:
            for node in graph_data.nodes:
                safe_label = sanitize_identifier(node.label)
                cypher = f"MERGE (n:{safe_label} {{name: toLower(trim($name))}})"
                session.run(cypher, name=node.name)
                
            for rel in graph_data.relationships:
                safe_src_label = sanitize_identifier(rel.source_label)
                safe_tgt_label = sanitize_identifier(rel.target_label)
                safe_rel_type = sanitize_identifier(rel.relation_type, uppercase=True)
                cypher = f"""
                MATCH (src:{safe_src_label} {{name: toLower(trim($src_name))}})
                MATCH (tgt:{safe_tgt_label} {{name: toLower(trim($tgt_name))}})
                MERGE (src)-[:{safe_rel_type}]->(tgt)
                """
                session.run(cypher, src_name=rel.source_name, tgt_name=rel.target_name)
        print(f"  -> Added {len(graph_data.nodes)} nodes and {len(graph_data.relationships)} relationships.")
    except Exception as e:
        print(f"  -> Extraction failed: {e}")

# 3. ADDED: THE PDF READING LOOP
def process_pdf(pdf_path: str):
    print(f"Opening PDF: {pdf_path}")
    reader = PdfReader(pdf_path)

    for idx, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        rows = re.split(r"(?=\d{1,3}(?=[A-Z]))", text)
        rows = [row.strip() for row in rows if re.match(r"^\d{1,3}(?=[A-Z])", row.strip())]
        print(f"Processing Page {idx}: {len(rows)} table rows...")

        for start in range(0, len(rows), 8):
            batch = "\n".join(rows[start:start + 8])
            ingest_text_to_graph(batch)
            time.sleep(1)


def split_table_values(value) -> list[str]:
    """Split comma-separated symptoms or treatments from one dataset cell."""
    if value is None:
        return []

    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "n/a"}:
        return []

    return [item.strip() for item in re.split(r",\s*", text) if item.strip()]


def merge_disease_rows(tx, rows: list[dict]) -> tuple[int, int]:
    result = tx.run(
        """
        UNWIND $rows AS row
        MERGE (d:Disease {name: toLower(trim(row.name))})
        FOREACH (symptom_name IN row.symptoms |
            MERGE (s:Symptom {name: toLower(trim(symptom_name))})
            MERGE (d)-[:HAS_SYMPTOM]->(s)
        )
        FOREACH (treatment_name IN row.treatments |
            MERGE (t:Treatment {name: toLower(trim(treatment_name))})
            MERGE (d)-[:TREATED_BY]->(t)
        )
        RETURN count(d) AS rows_merged
        """,
        rows=rows,
    )
    counters = result.consume().counters
    return counters.nodes_created, counters.relationships_created


def ingest_huggingface_dataset(batch_size: int = 50):
    """Merge the Hugging Face disease table into the existing Neo4j graph."""
    dataset = load_dataset("QuyenAnhDE/Diseases_Symptoms", split="train")
    added_nodes = 0
    added_relationships = 0
    processed_rows = 0
    batch = []

    with driver.session() as session:
        session.run("RETURN 1").consume()

        for record in dataset:
            row = {str(key).strip().lower(): value for key, value in record.items()}
            disease = str(row.get("name") or "").strip()
            if not disease:
                continue

            batch.append({
                "name": disease,
                "symptoms": split_table_values(row.get("symptoms")),
                "treatments": split_table_values(row.get("treatments")),
            })

            if len(batch) >= batch_size:
                nodes, relationships = session.execute_write(merge_disease_rows, batch)
                added_nodes += nodes
                added_relationships += relationships
                processed_rows += len(batch)
                batch.clear()

        if batch:
            nodes, relationships = session.execute_write(merge_disease_rows, batch)
            added_nodes += nodes
            added_relationships += relationships
            processed_rows += len(batch)

    print(
        f"Finished. Merged {processed_rows} dataset rows into Neo4j; "
        f"created {added_nodes} nodes and {added_relationships} relationships."
    )


# Run the table ingestion by default. The existing PDF workflow remains available
# by calling process_pdf(path) when PDF ingestion is needed.
if __name__ == "__main__":
    ingest_huggingface_dataset()