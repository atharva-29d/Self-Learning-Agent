# Self-Learning Agent: Long-Term Memory AI Assistant

A fully local AI assistant that **remembers you across sessions**. It combines **Mem0** (memory layer), **Qdrant** (vector database), **Ollama** (Llama 3.1 for chat) and **Nomic Embed Text** (embeddings), so your conversations and memories never leave your machine.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Mem0](https://img.shields.io/badge/Memory-Mem0-purple)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20%7C%20Llama%203.1-black)
![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-red)

> This project started as a reproduction of Dave Ebbelaar's Mem0 AI Cookbook example. The goal is to understand how long-term memory works in an AI assistant before extending it with custom memory management, conflict handling and retrieval evaluation (see the [Roadmap](#roadmap)).

---

## Features

- **Persistent long-term memory:** facts from past conversations are stored and recalled in later sessions
- **Automatic memory extraction:** Mem0 uses the LLM to pull discrete facts (e.g. "User is learning Python") out of each conversation turn
- **Semantic retrieval:** the top-5 most relevant memories are found by embedding similarity, not keyword matching
- **Per-user memory isolation:** every memory is scoped by `user_id`
- **Metadata tagging:** memories can carry custom metadata such as a category
- **100% local:** LLM inference, embeddings and storage all run on your machine
- **Memory inspection:** view everything stored with `memory.get_all()`

---

## How It Works

Every chat turn follows a four-step loop:

1. **Retrieve:** search Qdrant for memories relevant to the user's message
2. **Augment:** inject those memories into the system prompt
3. **Generate:** Llama 3.1 (via Ollama) produces the reply
4. **Store:** the user and assistant messages are passed to `memory.add()`, which extracts and saves new facts

```text
                         User
                           │
                           ▼
                    ┌──────────────┐
                    │   main.py    │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │     Mem0     │
                    │ Memory Layer │
                    └──────┬───────┘
                           │
                  ┌────────┴────────┐
                  │                 │
                  ▼                 ▼
             Nomic Embed         Qdrant
                Text          Vector Database
                  │                 │
                  └────────┬────────┘
                           │
                    Relevant Memories
                           │
                           ▼
                    ┌──────────────┐
                    │  Llama 3.1   │
                    │    Ollama    │
                    └──────┬───────┘
                           │
                           ▼
                    Assistant Response
                           │
                           ▼
                       Mem0.add()
                           │
                           ▼
                    Store New Memories
```

### Example of extracted memories

From the experiments in `mem0_experiments.ipynb`, plain conversation is distilled into atomic facts:

| Input | Stored memory |
|---|---|
| "I am building an AI Agent." | `User is currently building an AI Agent` |
| "I prefer comedy movies over thriller ones." | `User prefers comedy movies over thriller ones` |
| "I'm building a RAG project for college." | `User is building a RAG project for college` |

Searching "What am I using for my RAG project?" then returns the most relevant memories ranked by similarity score.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Memory layer | [Mem0](https://github.com/mem0ai/mem0) |
| Vector database | [Qdrant](https://qdrant.tech/) (768-dim vectors) |
| LLM | Llama 3.1 via [Ollama](https://ollama.com/) |
| Embeddings | `nomic-embed-text` via Ollama |

---

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/download)** installed and running
- **[Docker](https://www.docker.com/)** (easiest way to run Qdrant)
- Enough RAM/VRAM for Llama 3.1 8B (roughly 8 GB RAM minimum; a GPU helps but is not required)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/atharva-29d/Self-Learning-Agent.git
cd Self-Learning-Agent
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Pull the Ollama models

```bash
ollama pull llama3.1:latest
ollama pull nomic-embed-text:latest
```

Make sure Ollama is running (`ollama serve`, or the desktop app) and reachable at `http://localhost:11434`.

### 4. Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

To keep your memories after the container is removed, mount a volume:

```bash
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

The Qdrant dashboard is available at <http://localhost:6333/dashboard>.

---

## Usage

Run the interactive assistant:

```bash
python main.py
```

```text
Mem0 initialized
AI Memory Assistant
Type 'exit' to quit.

You: I'm learning Python and building a RAG app for college.
Assistant: ...

You: exit
```

Start it again later and ask, *"What am I working on?"*. The assistant recalls it from stored memory.

### Changing the user

Memories are scoped per user. Edit this line at the top of `main.py` to use a different identity:

```python
USER_ID = "atharva"
```

### Inspecting stored memories

```python
from mem0 import Memory

memory = Memory.from_config(config)  # same config as in main.py
print(memory.get_all(filters={"user_id": "atharva"}))
```

---

## Configuration

All settings live in the `config` dictionary in `main.py`:

| Setting | Default | Notes |
|---|---|---|
| Vector store | Qdrant at `localhost:6333` | `embedding_model_dims` must be `768` for Nomic Embed |
| LLM | `llama3.1:latest` | `temperature: 0`, `max_tokens: 2000` |
| Embedder | `nomic-embed-text:latest` | Served by Ollama at `localhost:11434` |
| Retrieval | `limit=5` | Number of memories injected per turn |

No API keys are required, since everything runs locally.

---

## Project Structure

```text
Self-Learning-Agent/
├── main.py                    # Interactive chat loop with memory retrieval and storage
├── debug_search.py            # Inspects raw vector-store search results and scores
├── mem0_experiments.ipynb     # Experiments: add, search, get_all, LLM extraction debugging
├── mem0_experiments2.ipynb    # Experiments: metadata tagging and memory inspection
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Troubleshooting

**`LLM extraction failed: llama-server process has terminated ... CUDA error`**
The Ollama model server crashed while Mem0 was extracting facts, so no memory gets saved. Restart Ollama, and if it persists, update your GPU drivers or run Ollama on CPU. You can confirm Ollama itself works with `ollama run llama3.1`.

**Connection refused on port 6333**
Qdrant isn't running. Start the Docker container from step 4.

**Connection refused on port 11434**
Ollama isn't running. Start the Ollama app or run `ollama serve`.

**Vector dimension mismatch**
`embedding_model_dims` in the config must match the embedding model (768 for `nomic-embed-text`). If you switch embedding models, use a new Qdrant collection or clear the old one.

**Startup warnings about spaCy / fastembed**
These are harmless. Install `mem0ai[nlp]` and `mem0ai[extras]` (see `requirements.txt`) to enable lemmatization and BM25 keyword search.

---

## Roadmap

- [x] **Phase 1: Reference implementation.** Local memory store, semantic retrieval, per-user memories, metadata, memory inspection
- [ ] **Phase 2: Memory quality and evaluation**
  - [ ] Conflict handling (e.g. "I'm now building a RAG app instead of an agent" should update, not duplicate)
  - [ ] Retrieval improvements (score thresholds, reranking, hybrid keyword + vector search)
  - [ ] Evaluation set measuring retrieval accuracy on test queries
  - [ ] Memory management commands (list, edit, delete from the chat)
  - [ ] Unit tests

---

## Acknowledgements

- [Dave Ebbelaar's AI Cookbook](https://github.com/daveebbelaar/ai-cookbook) for the original Mem0 example this project builds on
- [Mem0](https://github.com/mem0ai/mem0), [Qdrant](https://qdrant.tech/), [Ollama](https://ollama.com/) and [Nomic AI](https://www.nomic.ai/)
