{
 "cells": [
  {
   "metadata": {
    "ExecuteTime": {
     "end_time": "2026-09-23T10:02:03.001178Z",
     "start_time": "2026-09-23T10:02:02.831225Z"
    }
   },
   "cell_type": "code",
   "source": [
    "from ollama import Client\n",
    "import json\n",
    "\n",
    "ollama = Client(host=\"http://localhost:11434\")\n",
    "\n",
    "VALID_CATEGORIES = {\n",
    "    \"fact\",\n",
    "    \"preference\",\n",
    "    \"goal\",\n",
    "    \"project\",\n",
    "    \"skill\",\n",
    "    \"temporary\",\n",
    "}\n",
    "\n",
    "\n",
    "def classify_memory(memory_text):\n",
    "    prompt = f\"\"\"\n",
    "You are a memory classification system.\n",
    "\n",
    "Classify the memory into exactly ONE of these categories:\n",
    "\n",
    "- fact: stable information about the user\n",
    "- preference: likes, dislikes, or personal preferences\n",
    "- goal: something the user wants to achieve\n",
    "- project: something the user is currently building or working on\n",
    "- skill: something the user is learning or knows\n",
    "- temporary: short-lived information that may become irrelevant\n",
    "\n",
    "IMPORTANT:\n",
    "- Use the category names EXACTLY as written.\n",
    "- Do not pluralize them.\n",
    "- Do not create new categories.\n",
    "- Return ONLY valid JSON.\n",
    "- Do not include explanations.\n",
    "\n",
    "Memory:\n",
    "{memory_text}\n",
    "\n",
    "Return:\n",
    "{{\n",
    "    \"category\": \"one category from the list\"\n",
    "}}\n",
    "\"\"\"\n",
    "\n",
    "    response = ollama.chat(\n",
    "        model=\"llama3.1:latest\",\n",
    "        messages=[\n",
    "            {\n",
    "                \"role\": \"user\",\n",
    "                \"content\": prompt\n",
    "            }\n",
    "        ],\n",
    "    )\n",
    "\n",
    "    result = json.loads(response.message.content)\n",
    "\n",
    "    category = result[\"category\"]\n",
    "\n",
    "    if category not in VALID_CATEGORIES:\n",
    "        raise ValueError(f\"Invalid category returned: {category}\")\n",
    "\n",
    "    return category"
   ],
   "id": "23d0b25f0c4f31ba",
   "outputs": [],
   "execution_count": 18
  },
  {
   "metadata": {
    "ExecuteTime": {
     "end_time": "2026-09-23T10:02:15.820805Z",
     "start_time": "2026-09-23T10:02:09.672051Z"
    }
   },
   "cell_type": "code",
   "source": [
    "print(classify_memory(\"User is currently learning Python\"))\n",
    "print(classify_memory(\"User wants to get an AI engineering job\"))\n",
    "print(classify_memory(\"User prefers sci-fi movies over thriller movies\"))\n",
    "print(classify_memory(\"User is building a RAG application\"))\n",
    "print(classify_memory(\"User uses Windows for development\"))"
   ],
   "id": "ca5423dac04ff098",
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "skill\n",
      "goal\n",
      "preference\n",
      "project\n",
      "fact\n"
     ]
    }
   ],
   "execution_count": 19
  },
  {
   "metadata": {
    "ExecuteTime": {
     "end_time": "2026-09-23T10:00:01.681929Z",
     "start_time": "2026-09-23T10:00:01.679446Z"
    }
   },
   "cell_type": "code",
   "source": [
    "from mem0 import Memory\n",
    "\n",
    "USER_ID = \"category_pipeline_test\"\n",
    "\n",
    "# -------------------------\n",
    "# 1. Mem0 configuration\n",
    "# -------------------------\n",
    "\n",
    "config = {\n",
    "    \"vector_store\": {\n",
    "        \"provider\": \"qdrant\",\n",
    "        \"config\": {\n",
    "            \"host\": \"localhost\",\n",
    "            \"port\": 6333,\n",
    "            \"embedding_model_dims\": 768,\n",
    "        },\n",
    "    },\n",
    "    \"llm\": {\n",
    "        \"provider\": \"ollama\",\n",
    "        \"config\": {\n",
    "            \"model\": \"llama3.1:latest\",\n",
    "            \"temperature\": 0,\n",
    "            \"max_tokens\": 2000,\n",
    "            \"ollama_base_url\": \"http://localhost:11434\",\n",
    "        },\n",
    "    },\n",
    "    \"embedder\": {\n",
    "        \"provider\": \"ollama\",\n",
    "        \"config\": {\n",
    "            \"model\": \"nomic-embed-text:latest\",\n",
    "            \"ollama_base_url\": \"http://localhost:11434\",\n",
    "        },\n",
    "    },\n",
    "}\n",
    "\n",
    "memory = Memory.from_config(config)\n",
    "\n",
    "\n",
    "for item in result[\"results\"]:\n",
    "    memory_text = item[\"memory\"]\n",
    "    category = classify_memory(memory_text)\n",
    "\n",
    "    print(f\"Memory:   {memory_text}\")\n",
    "    print(f\"Category: {category}\")\n",
    "    print()"
   ],
   "id": "4f66d9cbf160d881",
   "outputs": [],
   "execution_count": null
  },
  {
   "metadata": {
    "ExecuteTime": {
     "end_time": "2026-09-23T10:00:01.686770Z",
     "start_time": "2026-09-23T10:00:01.685030Z"
    }
   },
   "cell_type": "code",
   "source": "",
   "id": "c310f198a2142e38",
   "outputs": [],
   "execution_count": null
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
