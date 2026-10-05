import pdfplumber
import pandas as pd
from neo4j import GraphDatabase

# 1. Connect to Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "YourSecurePassword123"))

# 2. Extract tables directly from the PDF into a clean DataFrame
pdf_path = r"D:\data.pdf"
all_rows = []

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        table = page.extract_table()
        if table:
            # First page contains headers; subsequent pages append data rows
            if not all_rows:
                headers = [str(col).strip().lower() for col in table[0]]
                for row in table[1:]:
                    all_rows.append(row)
            else:
                for row in table:
                    # Skip duplicate header rows if repeated across pages
                    if [str(c).strip().lower() for c in row] != headers:
                        all_rows.append(row)

# Create the DataFrame
df = pd.DataFrame(all_rows, columns=headers)

# Clean string columns
for col in df.columns:
    df[col] = df[col].astype(str).str.strip()

print(f"Loaded {len(df)} records from PDF. Columns: {list(df.columns)}")

# 3. Ingest row-by-row into Neo4j
def ingest_patient_data(tx, row):
    cypher = """
    MERGE (sp:Speaker {id: toString($speaker_id)})
    MERGE (a:AudioSample {id: toString($id)})
    SET a.sentence = $sentence, 
        a.audio_path = $audio_path
    MERGE (sp)-[:SPOKE]->(a)

    WITH a
    MERGE (s:Symptom {name: toLower(trim($prompt))})
    MERGE (a)-[:MENTIONS_SYMPTOM]->(s)
    """
    tx.run(
        cypher, 
        id=row['id'], 
        sentence=row['sentence'], 
        speaker_id=row['speaker_id'], 
        audio_path=row['audio_path'], 
        prompt=row['prompt']
    )

# 4. Execute ingestion
with driver.session() as session:
    # Delete the malformed merged AudioSample nodes first
    session.run("MATCH (a:AudioSample) DETACH DELETE a")
    
    for _, row in df.iterrows():
        session.execute_write(ingest_patient_data, row)
        
print(f"Successfully ingested {len(df)} clean patient audio records!")