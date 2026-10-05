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

def classify_memories(memory_texts):

    formatted_memories = "\n".join(
        f"{i + 1}. {text}"
        for i, text in enumerate(memory_texts)
    )

    prompt = f"""
Classify each of the following user memories into exactly one category.

Categories:
- fact: stable information about the user
- preference: likes, dislikes, or preferences
- goal: something the user wants to achieve
- project: something the user is currently building or working on
- skill: something the user is learning or knows
- temporary: short-lived information that may soon become irrelevant

Return ONLY valid JSON as an array.
The category at each position must correspond to the memory at the same position.

Example:
["skill", "goal", "project"]

Memories:

{formatted_memories}
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

    if not isinstance(result, list):
        raise ValueError(
            "Classifier did not return a JSON list."
        )

    if len(result) != len(memory_texts):
        raise ValueError(
            f"Expected {len(memory_texts)} categories, "
            f"but received {len(result)}."
        )

    for category in result:

        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category returned: {category}"
            )

    return result



def classify_query(query):

    query = query.lower()

    if any(word in query for word in [
        "learning",
        "learn",
        "studying",
        "study",
        "skill",
        "know"
    ]):
        return "skill"

    if any(word in query for word in [
        "building",
        "build",
        "working on",
        "project",
        "developing"
    ]):
        return "project"

    if any(word in query for word in [
        "goal",
        "goals",
        "want to achieve",
        "trying to achieve",
        "aim"
    ]):
        return "goal"

    if any(word in query for word in [
        "prefer",
        "like",
        "dislike",
        "favorite",
        "favourite"
    ]):
        return "preference"

    if any(word in query for word in [
        "who am i",
        "where do i",
        "what do i use",
        "my computer"
    ]):
        return "fact"

    return "all"

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

Do not mention the memory system unless explicitly asked.You are a helpful AI assistant with access to stored memories about the user.

Use the memories below as factual context when answering the user's question.

IMPORTANT RULES:
- Only state information that is supported by the user's message or the retrieved memories.
- Do not invent, expand, reinterpret, or speculate about the user's projects, skills, goals, or preferences.
- Do not change the meaning of stored memories.
- If a memory contains an abbreviation, preserve the abbreviation exactly unless the user explicitly asks what it means.
- If the retrieved memories do not contain enough information to answer the question, say that you do not have enough information.
- Do not treat your own previous responses or recommendations as facts about the user.
- Do not mention the memory system unless explicitly asked.

Retrieved memories:
{memory_text}
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
    # 7. Categorize newly extracted memories

    new_memories = result.get("results", [])

    if new_memories:

        memory_texts = [
            item["memory"]
            for item in new_memories
        ]

        categories = classify_memories(memory_texts)

        for item, category in zip(
                new_memories,
                categories
        ):
            memory_id = item["id"]
            memory_text = item["memory"]

            memory.update(
                memory_id,
                metadata={
                    "category": category
                },
            )

            print(f"\nMemory: {memory_text}")
            print(f"Category: {category}")

    # 8. Return both response and retrieved memories
    return assistant_message, retrieved_memories

# -----------------------------
# Interactive chat
# -----------------------------

if __name__ == "__main__":

    print("AI Memory Assistant")
    print("Type 'exit' to quit.\n")

    while True:

        user_message = input("You: ")

        if user_message.lower() == "exit":
            break

        answer, retrieved_memories = chat(user_message)

        print(f"\nAssistant: {answer}\n")


memory.delete("0d56b65c-0768-431a-bef9-0611423f3144")
all_memories = memory.get_all(
    filters={"user_id": USER_ID}
)

for item in all_memories["results"]:

    metadata = item.get("metadata") or {}

    if metadata.get("category") == "project":

        print("ID:", item["id"])
        print("Memory:", item["memory"])
        print("Metadata:", metadata)
        print()

