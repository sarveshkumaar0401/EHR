import argparse
import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import dotenv_values, load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from neo4j import GraphDatabase

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / "Integration" / ".env", override=True)
load_dotenv(BASE_DIR / "RAG" / ".env", override=False)
LOCAL_ENV = dotenv_values(Path(__file__).with_name(".env"))
for key in ("GEMINI_API_KEY", "GEMINI_MODEL"):
    if LOCAL_ENV.get(key):
        os.environ[key] = LOCAL_ENV[key]

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
if GEMINI_MODEL == "gemini-2.5-flash":
    GEMINI_MODEL = "gemini-3.6-flash"

gemini_client = genai.Client() if os.environ.get("GEMINI_API_KEY") else None
neo4j_driver = GraphDatabase.driver(
    os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    auth=(
        os.environ.get("NEO4J_USER", "neo4j"),
        os.environ.get("NEO4J_PASSWORD", ""),
    ),
)


def extract_facts(note: str) -> dict:
    if gemini_client is None:
        raise RuntimeError("GEMINI_API_KEY is missing from Integration/.env or RAG/.env.")

    prompt = f"""Extract only explicit medical facts from this clinical note.
Return JSON with exactly these fields:
{{
  "relationships": [{{"symptom": "...", "disease": "..."}}],
  "treatments": [{{"disease": "...", "treatment": "..."}}]
}}

Rules:
- Include a symptom-disease pair only when the note supports that the symptom is caused by, explained by, or associated with the disease.
- Include only actual diagnoses or diseases, not symptoms, tests, findings, or risk factors.
- Include treatment-plan actions, medications, referrals, monitoring, or follow-up linked to a disease when stated.
- Do not infer facts that are not in the note.
- Use short normalized medical names.
- Return empty lists when the note does not contain the requested relation.

NOTE:
{note}"""

    for attempt in range(3):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            break
        except (ClientError, ServerError) as exc:
            if isinstance(exc, ClientError) and (
                getattr(exc, "code", None) == 429
                or getattr(exc, "status_code", None) == 429
            ):
                raise
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

    return json.loads(response.text)


def merge_fact(
    tx,
    disease: str,
    symptom: str | None = None,
    treatment: str | None = None,
) -> int:
    if symptom:
        result = tx.run(
            """
            MERGE (d:Disease {name: toLower(trim($disease))})
            MERGE (s:Symptom {name: toLower(trim($symptom))})
            MERGE (d)-[r:HAS_SYMPTOM]->(s)
            RETURN r
            """,
            disease=disease,
            symptom=symptom,
        )
    else:
        result = tx.run(
            """
            MERGE (d:Disease {name: toLower(trim($disease))})
            MERGE (t:Treatment {name: toLower(trim($treatment))})
            MERGE (d)-[r:TREATED_BY]->(t)
            RETURN r
            """,
            disease=disease,
            treatment=treatment,
        )

    return result.consume().counters.relationships_created


def enrich_dataset(subset: str, splits: list[str], limit: int | None = None):
    dataset = load_dataset("mkieffer/ACI-Bench", subset)
    processed = 0
    added_relationships = 0

    try:
        with neo4j_driver.session() as session:
            session.run("RETURN 1").consume()

            for split in splits:
                if split not in dataset:
                    continue

                for row in dataset[split]:
                    if limit is not None and processed >= limit:
                        break

                    note = str(row.get("note", "")).strip()
                    if not note:
                        continue

                    try:
                        facts = extract_facts(note)
                    except (ClientError, ServerError) as exc:
                        if isinstance(exc, ClientError) and (
                            getattr(exc, "code", None) == 429
                            or getattr(exc, "status_code", None) == 429
                        ):
                            print(
                                "Gemini quota exhausted. "
                                "Stopping before sending more requests."
                            )
                            return
                        print(
                            f"Skipping {split} record {processed + 1}: "
                            f"Gemini unavailable after retries ({exc})."
                        )
                        continue
                    processed += 1

                    for pair in facts.get("relationships", []):
                        symptom = str(pair.get("symptom", "")).strip()
                        disease = str(pair.get("disease", "")).strip()
                        if symptom and disease:
                            added_relationships += session.execute_write(
                                merge_fact,
                                disease=disease,
                                symptom=symptom,
                            )

                    for pair in facts.get("treatments", []):
                        disease = str(pair.get("disease", "")).strip()
                        treatment = str(pair.get("treatment", "")).strip()
                        if disease and treatment:
                            added_relationships += session.execute_write(
                                merge_fact,
                                disease=disease,
                                treatment=treatment,
                            )

                    print(
                        f"Processed {split} record {processed}; "
                        f"added {added_relationships} new relationships."
                    )
    finally:
        neo4j_driver.close()

    print(
        f"Finished. Processed {processed} notes and added "
        f"{added_relationships} new relationships to Neo4j."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Add new ACI-Bench medical facts to the existing Neo4j graph."
    )
    parser.add_argument("--subset", default="virtassist")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    enrich_dataset(args.subset, args.splits, args.limit)


if __name__ == "__main__":
    main()
