# Self-Learning Agent: Long-Term Memory AI Assistant

A fully local AI assistant that **remembers you across sessions**. It combines **Mem0** (memory layer), **Qdrant** (vector database), **Ollama** (Llama 3.1 for chat) and **Nomic Embed Text** (embeddings), with a **Streamlit** interface — so your conversations and memories never leave your machine.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Mem0](https://img.shields.io/badge/Memory-Mem0-purple)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20%7C%20Llama%203.1-black)
![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-red)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

> This project started as a reproduction of Dave Ebbelaar's Mem0 AI Cookbook example (Phase 1), then extended with original work on memory categorization, retrieval, and a web UI (Phase 2). See [Future Scope](#future-scope) for the research directions this architecture is positioned to explore next.

---

## Features

**Phase 1 — Reference implementation**
- Persistent long-term memory across sessions
- Automatic memory extraction from conversation turns
- Semantic retrieval via embedding similarity (not keyword matching)
- Per-user memory isolation
- Metadata tagging
- Memory inspection (`get_all()`)
- 100% local — no external API calls

**Phase 2 — Original extensions**
- **Automatic memory categorization** into six types: Fact, Preference, Goal, Project, Skill, Temporary
- **Category metadata** stored with every memory
- **Query categorization** — incoming queries are classified the same way as stored memories
- **Category-aware retrieval** — retrieval can filter/weight by category, not just vector similarity
- **Batch memory classification** to reduce the number of LLM calls per turn
- **Lightweight query routing** to cut latency on simple queries
- **Grounded response prompting** — responses are explicitly conditioned on retrieved memories
- **Streamlit web UI**, including a memory dashboard, retrieved-memory display, and similarity scores

---

## How It Works

Every chat turn follows a four-step loop:

1. **Retrieve:** classify the query, then search Qdrant for memories relevant to it (semantic similarity, filterable by category)
2. **Augment:** inject the retrieved memories into the system prompt
3. **Generate:** Llama 3.1 (via Ollama) produces a response grounded in those memories
4. **Store:** new facts are extracted from the exchange, classified into a category, and saved back to memory

```text
                         User
                           │
                           ▼
                 ┌───────────────────┐
                 │   Streamlit UI    │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │     main.py       │
                 │  (routing logic)  │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │       Mem0        │
                 │   Memory Layer    │
                 └─────────┬─────────┘
                           │
                  ┌────────┴────────┐
                  │                 │
                  ▼                 ▼
             Nomic Embed         Qdrant
                Text          Vector Database
                  │                 │
                  └────────┬────────┘
                           │
                 Relevant, Category-Filtered
                        Memories
                           │
                           ▼
                 ┌───────────────────┐
                 │     Llama 3.1     │
                 │      (Ollama)     │
                 └─────────┬─────────┘
                           │
                           ▼
                  Grounded Response
                           │
                           ▼
                Classify & Store New
                      Memories
```

### Memory categories

| Category | Meaning |
|---|---|
| **Fact** | Stable information about the user |
| **Preference** | Likes, dislikes, or preferences |
| **Goal** | Something the user wants to achieve |
| **Project** | Something the user is currently working on |
| **Skill** | Something the user is learning or knows |
| **Temporary** | Short-lived information that may soon be irrelevant |

### Example: conversation to structured memory

| Input | Extracted memory | Category |
|---|---|---|
| "I am building an AI Agent." | User is currently building an AI Agent | Project |
| "I prefer comedy movies over thriller ones." | User prefers comedy movies over thriller ones | Preference |
| "I'm learning Python." | User is learning Python | Skill |

A later query like *"What am I currently working on?"* is itself classified (→ `project`), which narrows retrieval to category-relevant memories before ranking by similarity.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Memory layer | [Mem0](https://github.com/mem0ai/mem0) |
| Vector database | [Qdrant](https://qdrant.tech/) (768-dim vectors) |
| LLM | Llama 3.1 via [Ollama](https://ollama.com/) |
| Embeddings | `nomic-embed-text` via Ollama |
| UI | [Streamlit](https://streamlit.io/) |

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

### Web UI (recommended)

```bash
streamlit run app.py
```

Opens a browser interface with:
- A chat panel for conversing with the assistant
- A memory dashboard showing all stored memories and their categories
- Retrieved-memory display with similarity scores for the current query
- A category filter

### CLI

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

Memories are scoped per user. Edit this line at the top of `main.py` (or set it in the UI) to use a different identity:

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
| Retrieval | `limit=5` | Number of memories injected per turn, filterable by category |

No API keys are required, since everything runs locally.

---

## Project Structure

```text
Self-Learning-Agent/
├── main.py                    # CLI chat loop: retrieval, categorization, generation, storage
├── app.py                     # Streamlit web UI: chat, memory dashboard, category filter
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

## Future Scope

Phase 1 and Phase 2 cover extraction, categorization, retrieval, and a usable UI — but they don't yet cover how the system behaves when memory *changes* over time, or how it should be *measured*. The gaps below come from current memory-agent research, and this architecture (local Mem0 + Qdrant + an LLM classification layer already in place) is well-positioned to explore them next.

### 1. Conflict resolution

Recent literature identifies conflict resolution as a critical, largely unsolved bottleneck in LLM agents: when new, potentially contradictory information arrives over a multi-turn conversation, agents struggle to detect and overwrite outdated facts so that later queries reflect only the newest valid state. Even advanced indexing and agentic loops fail to handle contradictory updates reliably — most practical memory layers, this project included today, default to simplistic overwrite rules that are difficult to inspect or correct.

**Planned direction:** use the existing Llama 3.1 classification layer to detect when a new memory contradicts a stored one (e.g. a user's project focus changing from "building an agent" to "building a RAG app") and resolve it explicitly — updating the existing memory in place rather than creating a duplicate — with the decision logged and inspectable rather than silent.

### 2. Evaluation beyond retrieval

Existing memory benchmarks have historically evaluated reasoning, tool orchestration, or basic retrieval within static contexts. These conventional testbeds are insufficient for a *dynamic* memory agent, since they often lack systematic testing for abilities like handling contradictory updates or test-time learning.

**Planned direction:** adopt a multi-dimensional benchmarking approach — in the spirit of frameworks like MemoryAgentBench — that scores the agent separately across four competencies: Accurate Retrieval, Test-Time Learning, Long-Range Understanding, and Conflict Resolution, instead of a single retrieval-accuracy number.

### 3. Hybrid retrieval architecture

The literature indicates that no single memory paradigm — pure long-context windows or plain RAG — is sufficient across all memory competencies. Pure in-context methods are limited by window size, while the effectiveness of RAG depends heavily on chunking and retriever granularity. Researchers are calling for hybrid architectures that integrate buffer, dense-retrieval, and structured (e.g. graph or temporal) memory to achieve better global-local trade-offs.

**Planned direction:** this project's existing semantic (vector) search is a solid foundation for exactly this. The planned extension — hybrid search combining keyword matching with vector similarity, plus reranking — is a concrete step toward that hybrid trade-off.

### Other planned items

- In-chat memory management commands (list, edit, delete)
- Unit test coverage for the retrieval and extraction pipeline
- Dockerized deployment (app + Qdrant + Ollama) for one-command setup

---

## Acknowledgements

- [Dave Ebbelaar's AI Cookbook](https://github.com/daveebbelaar/ai-cookbook) for the original Mem0 example this project's Phase 1 builds on
- [Mem0](https://github.com/mem0ai/mem0), [Qdrant](https://qdrant.tech/), [Ollama](https://ollama.com/), [Nomic AI](https://www.nomic.ai/), and [Streamlit](https://streamlit.io/)