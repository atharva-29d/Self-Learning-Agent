from mem0 import Memory

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 768,
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.1:latest",
            "temperature": 0,
            "max_tokens": 2000,
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "ollama_base_url": "http://localhost:11434",
        },
    },
}

memory = Memory.from_config(config)

result = memory._search_vector_store(
    query="what am I studying?",
    filters={"user_id": "atharva"},
    limit=5,
    threshold=0.1,
    explain=True,
)

print("\nDIRECT _search_vector_store RESULT:")
print(result)