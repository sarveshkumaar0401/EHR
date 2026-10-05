from langchain_neo4j import Neo4jGraph

# Initialize Neo4jGraph without any dangerous request flags
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="YourSecurePassword123"
)

# Run a quick query to inspect a few Speaker and AudioSample nodes
print("--- Checking Speakers in Neo4j ---")
speakers = graph.query("MATCH (s:Speaker) RETURN s LIMIT 5")
print(speakers)

print("\n--- Checking AudioSamples in Neo4j ---")
samples = graph.query("MATCH (a:AudioSample) RETURN a LIMIT 5")
print(samples)