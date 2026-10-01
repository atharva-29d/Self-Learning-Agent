from mem0 import Memory
from ollama import Client
import json


# -----------------------------
# Mem0 configuration
# -----------------------------

USER_ID = "atharva"

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


# -----------------------------
# Memory categories
# -----------------------------

VALID_CATEGORIES = {
    "fact",
    "preference",
    "goal",
    "project",
    "skill",
    "temporary",
}


# -----------------------------
# Classify a memory
# -----------------------------

def classify_memory(memory_text):

    prompt = f"""
Classify the following user memory into exactly one category.

Categories:
- fact: stable information about the user
- preference: likes, dislikes, or preferences
- goal: something the user wants to achieve
- project: something the user is currently building or working on
- skill: something the user is learning or knows
- temporary: short-lived information that may soon become irrelevant

Return ONLY valid JSON in this format:
{{"category": "skill"}}

Memory:
{memory_text}
"""

    response = ollama.chat(
        model="llama3.1:latest",
        messages=[
            {
                "role": "user",
                "content": prompt,
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



def classify_query(query):

    prompt = f"""
Classify the following user query into exactly one category if possible.

Categories:
- fact: asks about stable information about the user
- preference: asks about likes, dislikes, or preferences
- goal: asks about something the user wants to achieve
- project: asks about something the user is building or working on
- skill: asks about something the user is learning or knows
- temporary: asks about short-lived information
- all: use this when the query does not clearly belong to one category

Return ONLY valid JSON in this format:
{{"category": "skill"}}

Query:
{query}
"""

    response = ollama.chat(
        model="llama3.1:latest",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    result = json.loads(response.message.content)

    category = result["category"]

    valid_query_categories = VALID_CATEGORIES | {"all"}

    if category not in valid_query_categories:
        raise ValueError(
            f"Invalid query category returned: {category}"
        )

    return category


# -----------------------------
# Initialize Mem0 + Ollama
# -----------------------------

memory = Memory.from_config(config)

ollama = Client(
    host="http://localhost:11434"
)

print("Mem0 initialized")


# -----------------------------
# Chat function
# -----------------------------

def chat(user_message):

    # 1. Classify the user's query
    query_category = classify_query(user_message)

    print(f"Query category: {query_category}")

    # 2. Retrieve relevant memories
    if query_category == "all":

        memories = memory.search(
            user_message,
            filters={"user_id": USER_ID},
            limit=5
        )

    else:

        memories = memory.search(
            user_message,
            filters={
                "user_id": USER_ID,
                "category": query_category
            },
            limit=5
        )

    retrieved_memories = memories["results"]

    # 3. Convert memories into context
    memory_text = "\n".join(
        item["memory"]
        for item in retrieved_memories
    )

    # 4. Build prompt
    system_prompt = f"""
You are a helpful AI assistant.

Use the following memories about the user when they are relevant:

{memory_text}

Do not mention the memory system unless explicitly asked.
"""

    # 5. Generate response
    response = ollama.chat(
        model="llama3.1:latest",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            },
        ],
    )

    assistant_message = response.message.content

    # 6. Store user's conversation
    result = memory.add(
        [
            {
                "role": "user",
                "content": user_message
            }
        ],
        user_id=USER_ID,
    )

    # 7. Categorize newly extracted memories
    for item in result.get("results", []):

        memory_id = item["id"]
        memory_text = item["memory"]

        category = classify_memory(memory_text)

        memory.update(
            memory_id,
            metadata={
                "category": category
            },
        )

        print(f"Memory: {memory_text}")
        print(f"Category: {category}")

    # 8. Return both response and retrieved memories
    return assistant_message, retrieved_memories

# -----------------------------
# Interactive chat
# -----------------------------

print("AI Memory Assistant")
print("Type 'exit' to quit.\n")


while True:

    user_message = input("You: ")

    if user_message.lower() == "exit":
        break

    answer, retrieved_memories = chat(user_message)

    print(f"\nAssistant: {answer}\n")