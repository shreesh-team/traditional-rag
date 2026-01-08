# Traditional RAG Implementation

This project demonstrates a traditional Retrieval-Augmented Generation (RAG) pipeline using [LangChain](https://www.langchain.com/). It serves as a reference implementation for building applications that retrieve context from custom data sources to augment LLM responses.

## Overview

Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) by providing them with relevant data retrieved from an external knowledge base. This project implements the standard RAG architecture:

1.  **Ingestion**: Loading documents and splitting them into chunks.
2.  **Embedding**: Converting text chunks into vector representations.
3.  **Storage**: Storing vectors in a Vector Database.
4.  **Retrieval**: Querying the database for relevant context based on user input.
5.  **Generation**: Passing the retrieved context and user query to an LLM to generate an answer.

## Features

*   **Document Loading**: Support pdf file format.
*   **Text Splitting**: Efficient chunking strategies for optimal context window usage.
*   **Vector Store**: Integration with vector databases (ChromaDB).
*   **LLM Integration**: Connects with major LLM providers via LangChain.

## Prerequisites

*   Python 3.13
*   API Keys for your chosen LLM provider (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`) or Local LLM. [Optional]

## Installation

1.  Clone the repository:
    ```bash
    git clone git@github.com:shreesh-team/traditional-rag.git # Using SSH
    cd traditional-rag
    ```

2.  Create a virtual environment:
    ```bash
    uv venv --python 3.13 .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    uv pip install -r requirements.txt
    ```

## Configuration

Create a `.env` file in the root directory and add your environment variables: [Optional]

```env
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the main script to start the RAG pipeline (adjust filename as needed):

```bash
uvicorn main:app --reload
```
