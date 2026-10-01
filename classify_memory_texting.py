#%%
from mem0 import Memory
from ollama import Client
import json

# ============================================================
# 1. Configuration
# ============================================================

USER_ID = "category_pipeline_test"

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

# Create Mem0
memory = Memory.from_config(config)

# Create Ollama client for classification
ollama = Client(host="http://localhost:11434")


# ============================================================
# 2. Category definitions
# ============================================================

VALID_CATEGORIES = {
    "fact",
    "preference",
    "goal",
    "project",
    "skill",
    "temporary",
}


# ============================================================
# 3. Memory classification
# ============================================================

def classify_memory(memory_text):

    prompt = f"""
You are a memory classification system.

Classify the following memory into exactly ONE category.

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

Return exactly:

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

    # Validate Llama's response
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category returned by Llama: {category}"
        )

    return category


# ============================================================
# 4. User conversation
# ============================================================

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


# ============================================================
# 5. Get existing memories
# ============================================================

all_memories = memory.get_all(
    filters={
        "user_id": USER_ID
    }
)


# ============================================================
# 6. Classify + update existing memories
# ============================================================

print("\n=== CATEGORIZED MEMORIES ===\n")

for item in all_memories["results"]:

    memory_id = item["id"]
    memory_text = item["memory"]

    # Skip memories that already have a category
    existing_metadata = item.get("metadata") or {}

    if "category" in existing_metadata:
        print(f"Already categorized: {memory_text}")
        print(f"Category: {existing_metadata['category']}")
        print()
        continue

    # Classify with Llama
    category = classify_memory(memory_text)

    # Update existing Mem0 memory
    memory.update(
        memory_id,
        metadata={
            "category": category
        }
    )

    print(f"Memory:   {memory_text}")
    print(f"Category: {category}")
    print(f"ID:       {memory_id}")
    print()


# ============================================================
# 7. Verify
# ============================================================

print("\n=== STORED MEMORIES ===\n")

all_memories = memory.get_all(
    filters={
        "user_id": USER_ID
    }
)

for item in all_memories["results"]:

    print(f"Memory:   {item['memory']}")
    print(f"Metadata: {item.get('metadata')}")
    print(f"ID:       {item.get('id')}")
    print()


query = "What am I learning?"

results = memory.search(
    query,
    filters={
        "user_id": USER_ID,
        "category": "skill"
    }
)

print("\n=== CATEGORY-AWARE SEARCH ===\n")

for item in results["results"]:
    print(f"Memory: {item['memory']}")
    print(f"Score:  {item.get('score')}")
    print(f"Metadata: {item.get('metadata')}")
    print()


query = "What am I building?"

results = memory.search(
    query,
    filters={
        "user_id": USER_ID,
        "category": "project"
    }
)

print("\n=== PROJECT-AWARE SEARCH ===\n")

for item in results["results"]:
    print(f"Memory: {item['memory']}")
    print(f"Score:  {item.get('score')}")
    print(f"Metadata: {item.get('metadata')}")
    print()