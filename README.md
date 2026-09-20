# Self-Learning Agent — Long-Term Memory AI Assistant

A local AI assistant with long-term memory built using **Mem0, Ollama, Nomic Embed Text, and Qdrant**.

This project started as a reproduction and learning implementation based on Dave Ebbelaar's Mem0 AI Cookbook example. The goal is to understand how long-term memory can be integrated into an AI assistant before extending the system with custom memory-management and retrieval improvements.

---

## Current Status

**Phase 1 — Reference Implementation: Complete**

The current version demonstrates:

- Long-term memory storage with Mem0
- Semantic memory retrieval
- Conversation-level memory extraction
- User-specific memories
- Vector storage with Qdrant
- Local LLM inference with Ollama
- Local text embeddings with Nomic Embed Text
- Memory inspection using `get_all()`
- Metadata attached to memories

Phase 2 will focus on improving memory quality, conflict handling, retrieval, and evaluation.

---

## Architecture

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