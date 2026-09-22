from mem0 import Memory
from ollama import Client

# -----------------------------
# Mem0 configuration
# -----------------------------

TEST_USER = "category_test"

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

messages = [
    {
        "role": "user",
        "content": "I prefer sci-fi movies over thriller movies."
    },
    {
        "role": "user",
        "content": "I'm currently learning Python."
    },
    {
        "role": "user",
        "content": "I'm building a RAG application for my AI project."
    },
    {
        "role": "user",
        "content": "My goal is to get an AI engineering job."
    },
    {
        "role": "user",
        "content": "I use Windows for development."
    }
]

expected_categories = {
    "User's goal is to get an AI engineering job": "goal",
    "User prefers sci-fi movies over thriller movies": "preference",
    "User is currently learning Python": "skill",
    "User is building a RAG application for their AI project": "project",
    "User uses Windows for development": "fact",
}

memory = Memory.from_config(config)


result = memory.add(
    messages,
    user_id=TEST_USER
)

for i , item in enumerate(result["results"],1):
    print(f"{i}. {item["memory"]} ")

all_memories = memory.get_all(
    filters={"user_id": TEST_USER}
)
for i, item in enumerate(all_memories["results"], 1):
    print(f"{i}. {item['memory']}")