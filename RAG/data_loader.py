import json
import os
import re

from datasets import load_dataset
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_client = genai.Client() if GEMINI_API_KEY else None


SYMPTOM_CANONICAL_MAP = {
    "sob": "dyspnea",
    "shortness of breath": "dyspnea",
    "shortness-of-breath": "dyspnea",
    "dyspnea": "dyspnea",
    "chest discomfort": "chest pain",
    "chest pain": "chest pain",
    "cough": "cough",
    "fever": "fever",
    "headache": "headache",
    "fatigue": "fatigue",
    "nausea": "nausea",
    "vomiting": "vomiting",
    "abdominal pain": "abdominal pain",
    "joint pain": "joint pain",
    "swelling": "swelling",
    "dizziness": "dizziness",
    "chills": "chills",
}


def normalize_symptoms(items):
    normalized = []
    seen = set()
    for item in items or []:
        key = str(item).strip().lower()
        canonical = SYMPTOM_CANONICAL_MAP.get(key, key)
        if canonical and canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized


def _fallback_extract_graph_entities(clinical_text: str):
    """Legacy deterministic fallback used when no LLM extraction is available."""
    symptom_vocab = [
        "fever",
        "cough",
        "shortness of breath",
        "dyspnea",
        "chest pain",
        "headache",
        "fatigue",
        "nausea",
        "vomiting",
        "abdominal pain",
        "joint pain",
        "swelling",
        "dizziness",
        "chills",
    ]

    text_lower = clinical_text.lower()
    canonical_symptoms = []
    pertinent_negatives = []

    for symptom in symptom_vocab:
        neg_pattern = rf"(?:no|denies|negative for|without)\s+[\w\s]{{0,15}}\b{symptom}\b"
        if re.search(neg_pattern, text_lower):
            pertinent_negatives.append(symptom)
        elif symptom in text_lower:
            canonical_symptoms.append(symptom)

    return list(set(canonical_symptoms)), list(set(pertinent_negatives))


def extract_graph_entities(clinical_text: str):
    """Long-term extraction strategy: use LLM first, fallback to deterministic rule set."""
    if gemini_client and clinical_text and len(clinical_text.strip()) > 20:
        prompt = f"""
        Extract medical symptoms and pertinent negatives from the clinical note.

        Return valid JSON only with this exact schema:
        {{
          "symptoms": ["symptom1", "symptom2"],
          "negative_symptoms": ["symptom3"]
        }}

        Rules:
        - Use canonical medical terms like 'dyspnea', 'chest pain', 'fever'.
        - Include symptoms explicitly mentioned or clearly implied.
        - Use negative_symptoms for symptoms that are denied, absent, or ruled out.
        - Do not include unrelated conditions.
        - If none are found, return empty arrays.

        Clinical note:
        {clinical_text}
        """

        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    system_instruction=(
                        "You are a clinical information extraction assistant. "
                        "Extract only medically relevant symptoms and pertinent negatives. "
                        "Return JSON only."
                    ),
                ),
            )
            text = (response.text or "").strip()
            if text:
                if text.startswith("```"):
                    text = text.strip("`")
                    if text.lower().startswith("json"):
                        text = text[4:].strip()
                data = json.loads(text)
                symptoms = normalize_symptoms(data.get("symptoms", []))
                negatives = normalize_symptoms(data.get("negative_symptoms", []))
                return symptoms, negatives
        except Exception:
            pass

    return _fallback_extract_graph_entities(clinical_text)


def embed_texts(texts):
    return _model.encode(texts, convert_to_numpy=True).tolist()


def load_and_chunk_dataset(allowed_splits=None):
    """Load ACI-Bench clinical notes and split only valid structured notes."""
    print("Loading 'virtassist' subset from mkieffer/ACI-Bench...")
    dataset = load_dataset("mkieffer/ACI-Bench", "virtassist")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=80,
        separators=["\n\n", "\n", "SUBJECTIVE:", "OBJECTIVE:", "ASSESSMENT:", "PLAN:", " "],
    )

    documents_to_embed = []
    if allowed_splits is None:
        allowed_splits = {"train", "valid", "validation"}
    else:
        allowed_splits = set(allowed_splits)

    for split_name, split_data in dataset.items():
        if split_name not in allowed_splits:
            print(f"Skipping evaluation split to prevent leakage: '{split_name}'")
            continue

        print(f"Processing split: '{split_name}' ({len(split_data)} records)...")
        for idx, row in enumerate(split_data):
            doc_id = row.get("id", f"{split_name}_{idx}")
            patient_id = row.get("patient_id") or f"PATIENT_{doc_id}"
            clinical_note = row.get("note", "")

            if not clinical_note:
                continue

            note_chunks = text_splitter.split_text(clinical_note)
            for chunk_idx, chunk in enumerate(note_chunks):
                symptoms, neg_symptoms = extract_graph_entities(chunk)
                documents_to_embed.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "doc_id": doc_id,
                            "patient_id": patient_id,
                            "split": split_name,
                            "source_type": "clinical_note",
                            "chunk_idx": chunk_idx,
                            "canonical_symptoms": symptoms,
                            "pertinent_negatives": neg_symptoms,
                        },
                    }
                )

    print(f"Generated {len(documents_to_embed)} clinical note chunks.")
    return documents_to_embed


def load_and_split_pdf(path: str):
    raise NotImplementedError("This project is using the ACI-Bench clinical-note dataset instead of PDF ingestion.")