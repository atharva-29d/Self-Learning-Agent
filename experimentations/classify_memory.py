#%%
from mem0 import Memory
from ollama import Client
import json

USER_ID = "category_pipeline_test"

# -------------------------
# 1. Mem0 configuration
# -------------------------

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

# -------------------------
# 2. Ollama classifier
# -------------------------

ollama = Client(host="http://localhost:11434")

VALID_CATEGORIES = {
    "fact",
    "preference",
    "goal",
    "project",
    "skill",
    "temporary",
}


def classify_memory(memory_text):

    prompt = f"""
You are a memory classification system.

Classify this memory into exactly ONE category.

Categories:

- fact: stable information about the user
- preference: likes, dislikes, or personal preferences
- goal: something the user wants to achieve
- project: something the user is currently building or working on
- skill: something the user is learning or knows
- temporary: short-lived information that may become irrelevant

IMPORTANT:
- Use the category names EXACTLY as written.
- Do not pluralize them.
- Do not create new categories.
- Return ONLY valid JSON.
- Do not include explanations.

Memory:
{memory_text}

Return:
{{
    "category": "one category from the list"
}}
"""

    response = ollama.chat(
        model="llama3.1:latest",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    result = json.loads(response.message.content)

    category = result["category"]

    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category returned: {category}"
        )

    return category


# -------------------------
# 3. Main Mem0 pipeline
# -------------------------

messages = [
    {
        "role": "user",
        "content": "I'm currently learning Python."
    },
    {
        "role": "user",
        "content": "I prefer sci-fi movies over thriller movies."
    },
    {
        "role": "user",
        "content": "I'm building a RAG application."
    },
    {
        "role": "user",
        "content": "My goal is to get an AI engineering job."
    },
    {
        "role": "user",
        "content": "I use Windows for development."
    },
]

result = memory.add(
    messages,
    user_id=USER_ID
)

# -------------------------
# 4. Classify Mem0 output
# -------------------------

print("\n=== EXTRACTED MEMORIES ===\n")

for item in result["results"]:

    memory_text = item["memory"]

    category = classify_memory(memory_text)

    print(f"Memory:   {memory_text}")
    print(f"Category: {category}")
    print()
#%%
