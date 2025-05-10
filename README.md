# AI Agent Collection: Web Search, RAG, and Multi-Agent Systems

This repository contains a collection of Python scripts demonstrating various AI agent implementations, including web search capabilities, Retrieval Augmented Generation (RAG) with local knowledge bases, and multi-agent team collaboration. The scripts leverage both the `phi-agent` library (with Groq for LLM inference) and direct Ollama integration.

## Table of Contents

1.  [Overview](#overview)
2.  [Features](#features)
3.  [Prerequisites](#prerequisites)
4.  [Setup and Installation](#setup-and-installation)
5.  [Configuration (.env file)](#configuration-env-file)
6.  [Script Descriptions and Usage](#script-descriptions-and-usage)
    *   [6.1 `duckduckgo_search_agents.py`](#61-duckduckgo_search_agentspy)
    *   [6.2 `google_search_agent.py`](#62-google_search_agentpy)
    *   [6.3 `knowledge_agent.py`](#63-knowledge_agentpy)
    *   [6.4 `real_time_search_team.py`](#64-real_time_search_teampy)
    *   [6.5 `search_agent.py` (Ollama-based)](#65-search_agentpy-ollama-based)
    *   [6.6 `sys_msgs.py`](#66-sys_msgspy)
7.  [Important Notes and Considerations](#important-notes-and-considerations)
8.  [Troubleshooting](#troubleshooting)

## 1. Overview

This project showcases different approaches to building intelligent agents capable of:
*   Performing web searches using DuckDuckGo and Google.
*   Scraping and processing web content.
*   Answering questions based on a local text-based knowledge base (RAG).
*   Coordinating multiple agents to achieve a complex task.
*   Utilizing both cloud-based LLMs (via Groq and `phi-agent`) and locally-run LLMs (via Ollama).

## 2. Features

*   **`phi-agent` Examples:**
    *   Single agent for DuckDuckGo search and Newspaper4k article extraction.
    *   Single agent for Google Search to find news.
    *   RAG agent using LanceDB and SentenceTransformerEmbedder for local document querying.
    *   Multi-agent team for searching (GoogleSearch) and scraping (Crawl4AI).
*   **Direct Ollama Integration Example:**
    *   A conversational agent with a custom RAG-like pipeline:
        *   Decides if a search is needed.
        *   Generates a search query.
        *   Performs DuckDuckGo search.
        *   Selects the best search result.
        *   Scrapes the webpage using Trafilatura.
        *   Determines content relevance.
        *   Responds to the user with (or without) augmented context.
*   Modular code structure for easy understanding and modification.
*   Configuration via `.env` files for API keys.
*   Comprehensive logging.

## 3. Prerequisites

*   Python 3.9+
*   Pip (Python package installer)
*   **Ollama installed and running** (for `search_agent.py` and `sys_msgs.py`).
    *   Ensure you have pulled the necessary models (e.g., `ollama pull llama3:latest` or `ollama pull gemma3:27b` or your preferred model as specified in `search_agent.py`).
*   **API Keys (required for `phi-agent` scripts):**
    *   Groq API Key
    *   Crawl4AI API Key (optional, for `real_time_search_team.py`)

## 4. Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    The `requirements.txt` file lists all necessary packages.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Playwright browsers (needed for `Newspaper4k` and potentially other web interaction tools):**
    After installing Python packages, run the following command to install the default browser binaries for Playwright:
    ```bash
    playwright install
    ```
    This will download browsers like Chromium, Firefox, and WebKit that Playwright uses.

5.  **Set up environment variables:**
    Create a `.env` file in the root directory of the project. See the [Configuration](#configuration-env-file) section below for details.

6.  **(For `knowledge_agent.py`) Create `air.txt`:**
    This script requires a text file named `air.txt` in the root directory to build its knowledge base. Add some sample text to it.

    
## 5. Configuration (.env file)

Create a `.env` file in the root directory of the project and add your API keys and any other configurations:

```env
# Groq API Key (Required for all phi-agent scripts)
GROQ_API_KEY="your_groq_api_key"

# Optional: Override default Groq model ID used in phi-agent scripts
# GROQ_MODEL_ID="llama-3.3-70b-versatile"
```

**Note:** The `search_agent.py` (Ollama-based) does not require API keys in the `.env` file as it uses a locally running Ollama instance. Its model is configured directly in the script (`OLLAMA_MODEL` constant).

## 6. Script Descriptions and Usage

You can run each Python script directly from your terminal after activating the virtual environment and setting up the `.env` file.

### 6.1 `duckduckgo_search_agents.py`

*   **Purpose:** Demonstrates a `phi-agent` that acts as an NYT researcher. It uses DuckDuckGo to find a relevant link for a given topic and Newspaper4k to extract the article content, then writes a summary.
*   **LLM:** Groq (via `phi-agent`).
*   **Tools:** `DuckDuckGo`, `Newspaper4k`.
*   **Requires:** `GROQ_API_KEY` in `.env`.
*   **Usage:**
    ```bash
    python duckduckgo_search_agents.py
    ```
    The script will research the predefined topic \"Simulation Theory\". You can modify the `topic_to_research` variable in its `main()` function.

### 6.2 `google_search_agent.py`

*   **Purpose:** A `phi-agent` designed to fetch the latest news items on a given topic using Google Search.
*   **LLM:** Groq (via `phi-agent`).
*   **Tools:** `GoogleSearch`.
*   **Requires:** `GROQ_API_KEY` in `.env`.
*   **Usage:**
    ```bash
    python google_search_agent.py
    ```
    It will search for news on \"latest developments in large language models (LLMs)\". Modify `topic_to_search` in `main()` for other topics.

### 6.3 `knowledge_agent.py`

*   **Purpose:** A RAG (Retrieval Augmented Generation) `phi-agent` that answers questions based on information from a local text file (`air.txt`). It uses LanceDB as a vector store and SentenceTransformerEmbedder for embeddings.
*   **LLM:** Groq (via `phi-agent`).
*   **Knowledge Source:** `air.txt` (must be created in the root directory).
*   **VectorDB:** LanceDB (data stored in `tmp/lancedb_air_data`).
*   **Requires:** `GROQ_API_KEY` in `.env`.
*   **Usage:**
    ```bash
    python knowledge_agent.py
    ```
    The script will ask a predefined question about \"Istanbul hava sıcaklığı\". Modify `questions` in `main()` for other queries.
    *   Set `FORCE_RECREATE_KB = True` in the script for the first run or if `air.txt` is updated to re-index the knowledge base. For subsequent runs, set it to `False` for faster startup.

### 6.4 `real_time_search_team.py`

*   **Purpose:** Demonstrates a multi-agent team using `phi-agent`. A lead agent coordinates a `WebSearcher` (using GoogleSearch) and a `WebScraper` (using Crawl4aiTools) to gather information and provide a summary.
*   **LLM:** Groq (via `phi-agent`).
*   **Tools:** `GoogleSearch`, `Crawl4aiTools`.
*   **Requires:** `GROQ_API_KEY` in `.env`.
*   **Usage:**
    ```bash
    python real_time_search_team.py
    ```
    The team will research \"What are the latest significant developments in AI ethics this month?\". Modify `query` in `main()` for other topics.

### 6.5 `search_agent.py` (Ollama-based)

*   **Purpose:** A conversational AI agent that uses a locally running Ollama model. It features a custom RAG-like pipeline to decide whether to search the web, generate queries, search DuckDuckGo, scrape content with Trafilatura, and then respond.
*   **LLM:** Ollama (model defined by `OLLAMA_MODEL` constant in the script, e.g., `gemma3:27b`).
*   **Dependencies:** `sys_msgs.py` for system prompts.
*   **Requires:**
    *   Ollama installed and running.
    *   The specified Ollama model pulled (e.g., `ollama pull gemma3:27b`).
*   **Usage:**
    ```bash
    python search_agent.py
    ```
    The script will start an interactive loop where you can chat with the assistant.

### 6.6 `sys_msgs.py`

*   **Purpose:** Contains system messages (prompts) used by `search_agent.py` to guide the Ollama model's behavior for different sub-tasks (e.g., deciding to search, generating queries, evaluating relevance).
*   **This is not a runnable script** but a required module for `search_agent.py`.

## 7. Important Notes and Considerations

*   **API Key Costs:** Be mindful of potential costs associated with using cloud APIs (Groq, Google Cloud, Crawl4AI). Monitor your usage.
*   **Model IDs:**
    *   For `phi-agent` scripts, the `GROQ_MODEL_ID` can be set in your `.env` file or defaults to the value in each script (e.g., `"llama-3.3-70b-versatile"` or `"llama3-70b-8192"`). Ensure the model ID you use is available in your Groq account.
    *   For `search_agent.py`, the `OLLAMA_MODEL` is defined as a constant within the script. Ensure you have this model pulled in your local Ollama instance.
*   **Ollama Performance:** The performance of `search_agent.py` will depend on your local hardware and the Ollama model used. Larger models may be slower but potentially more capable.
*   **Web Scraping Ethics:** Always ensure your web scraping activities are ethical and comply with the terms of service of the websites you are accessing. The provided user agent string is a generic one.
*   **Playwright Browsers:** The `Newspaper4k` tool may utilize Playwright for fetching web content, especially from dynamic websites. Ensure you have run `playwright install` after installing the Python dependencies to download the necessary browser drivers.
*   **Error Handling:** The scripts include basic error handling and logging, which can be helpful for debugging.
*   **Knowledge Base Re-creation (`knowledge_agent.py`):** Remember to set `FORCE_RECREATE_KB = True` in `knowledge_agent.py` if you update `air.txt` or want to re-index from scratch. Otherwise, set it to `False` to use the existing LanceDB index for faster startups.
  
## 8. Troubleshooting

*   **`ModuleNotFoundError`:** Ensure you have activated your virtual environment and installed all packages from `requirements.txt`.
*   **API Key Errors:** Double-check that your API keys in the `.env` file are correct, have the necessary permissions, and that billing is enabled for cloud services if required.
*   **Ollama Errors (for `search_agent.py`):**
    *   Make sure the Ollama application/service is running on your machine.
    *   Verify that the model specified in `OLLAMA_MODEL` (e.g., `gemma3:27b`) has been pulled: `ollama list`. If not, run `ollama pull <model_name>`.
    *   Check Ollama server logs for more detailed error messages.
*   **Tool Failures (e.g., GoogleSearch, Newspaper4k, Crawl4AI):** These can be due to network issues, changes in website structures, API limits, or invalid API keys. Check the console output and logs for error messages from the tools.
*   **`phi-agent` issues:** Refer to the official [phidata documentation](https://docs.phidata.com/) for more detailed troubleshooting.
