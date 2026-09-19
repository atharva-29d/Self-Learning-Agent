from mem0 import Memory
from ollama import Client

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
# Initialize Mem0 + Ollama
# -----------------------------

memory = Memory.from_config(config)

ollama = Client(host ="http://localhost:11434")

print("Mem0 initialized")


# -----------------------------
# Chat function
# -----------------------------

def chat(user_message):

    #1 retrieve relevant memories
    memories = memory.search(user_message,
                             filters = {"user_id": USER_ID},
                             limit = 5)

    memory_text = "/n".join(f"{item['memory']}"
                            for item in memories["results"])

    #2 Prompt

    system_prompt = f"""
                        You are a helpful AI assistant.
                        
                        Use the following memories about the user when they are relevant:
                        
                        {memory_text}
                        
                        Do not mention the memory system unless explicitly asked.
                        """


    #3 generate a reponse

    response = ollama.chat(
        model = "llama3.1:latest",
        messages = [
            {
                "role":"system",
                "content":system_prompt
            },
            {
                "role":"user",
                "content":user_message
            },
        ],
    )

    assistant_message = response.message.content

    #4 store convo in Mem0

    memory.add(
        [
            {
                "role":"user",
                "content":user_message
            },
            {
                "role":"assistant",
                "content":assistant_message,
            },
        ],
        user_id = USER_ID
    )
    return assistant_message


# -----------------------------
# Interactive chat
# -----------------------------

print("AI Memory Assistant")
print("Type 'exit' to quit.\n")

while True:

    user_message = input("You: ")

    if user_message.lower() == "exit":
        break

    answer = chat(user_message)

    print(f"\nAssistant: {answer}\n")

# # -----------------------------
# # 1. Add a memory
# # -----------------------------
#
# result = memory.add(
#     "I am an AI and Machine Learning student.",
#     user_id="atharva",
# )
#
# print("\nMemory added:")
# print(result)
#
#
# memories_to_add = ["I am learning Python",
#                    "I am currently building an AI Agent.",
#                    "I am intrested in Machine Learning.,"
#                    "I use Windows for development"]
#
# for text in memories_to_add:
#     result = memory.add(text, user_id = "atharva")
#
# print(f"/n Added :{text}")
# print(result)
#
#
# # result = memory.add("I am no longer buidling an AI agent. i am now building a RAG application", user_id = "atharva")
# # print("/n Update test:")
# # print(result)
#
# result = memory.add(
#     "I am building a RAG application instead of an AI Agent.",
#     user_id="atharva",
# )
#
# print(result)
# # -----------------------------
# # 2. Show all memories
# # -----------------------------
#
# all_memories = memory.get_all(
#     filters={"user_id": "atharva"}
# )
#
# print("\nAll memories:")
# print(all_memories)
#
#
# # -----------------------------
# # 3. Search memory
# # -----------------------------
#
# results = memory.search(
#     query="what am I studying?",
#     filters={"user_id": "atharva"},
#     top_k=5,
#     threshold=0.1,
# )
#
# print("\nRetrieved memories:")
# print(results)
#
#
# # -----------------------------
# # Test semantic retrieval
# # -----------------------------
#
# # queries = ["What is user currently building?",
# #          "What is user intrested in ?"]
# #
# # for query in queries:
# #     print("/n")
# #     print(f"Query: {query}")
# #
# #     result = memory.search(
# #         query = query,
# #         filters= {"user_id":"atharva"},
# #         top_k = 5,
# #         threshold = 0.1
# #     )
# #     print(result)
#
# # results = memory.search(
# #     query="What is user currently building?",
# #     filters={"user_id": "atharva"},
# #     top_k=5,
# #     threshold=0.60,
# # )
# #
# # print(results)
# #
#
#
#
