# 7 Wonders RAG & ReAct Agent — NLP Practical Assignment

A Natural Language Processing (NLP) project developed for *Procesamiento del Lenguaje Natural* (UNR) featuring a multi-source **Retrieval-Augmented Generation (RAG)** pipeline and a **ReAct Agent** specialized in the board game *7 Wonders*.

## Project Overview

The project is divided into two main parts:

### 1. Multi-Source Retrieval-Augmented Generation (RAG)
Builds a domain chatbot capable of answering questions in multiple languages by detecting language (`langdetect`), translating queries and responses (`deep-translator`), routing questions to the right data store, and generating grounded answers with an LLM (`Qwen/Qwen2.5-72B-Instruct` via `InferenceClient`):

- **Vector Database (`ChromaDB` + `Universal Sentence Encoder Multilingual`):** Extracts and cleans rulebooks (`PyPDF2` across *Reglas generales*, *CITIES*, *DUEL*, and *EDIFICE* PDFs) and forum/review pages (*BoardGameGeek* and *MisutMeeple* via `Selenium`), chunks text with `RecursiveCharacterTextSplitter` (400-char chunks, 30-char overlap), and combines semantic similarity with a keyword matching score (`keyword_score`) for hybrid ranking.
- **Graph Database (`NetworkX` + `rdflib` / `SPARQL`):** Scrapes *BoardGameGeek* credits (`Selenium`) and uses an LLM to extract `(subject, predicate, object)` triples for designers, artists, publishers, awards, and related entities, serialized to `graph_db.rdf` and queried dynamically via generated SPARQL queries.
- **Tabular Database (`pandas`):** Scrapes game and player statistics (*Avg. Rating*, *No. of Ratings*, *Weight*, *Players this Month*, etc.) into a DataFrame/CSV and maps natural-language queries dynamically to database entities using an LLM.
- **Query Routing Classifiers:** Compares two approaches to route queries across vector (`0`), graph (`1`), and tabular (`2`) stores:
  - **Embeddings + Logistic Regression (`SentenceTransformer` + `scikit-learn`):** ~0.99 accuracy on the evaluation split.
  - **LLM Classifier (`Qwen/Qwen2.5-72B-Instruct`):** ~0.98 accuracy on the evaluation split.

### 2. ReAct Agent (`LlamaIndex` + `Ollama`)
Implements a `ReActAgent` powered by `llama3.2:latest` running locally through `Ollama`:

- Exposes `get_info_graph_db`, `get_info_tabular_db`, and `get_info_vector_db` as `FunctionTool`s.
- Alternates through `Thought`, `Action`, `Action Input`, `Observation`, and `Final Answer` steps to query the appropriate source and synthesize responses.

## Tech Stack

- **NLP & LLMs:** `llama-index`, `ollama` (`llama3.2:latest`), `sentence-transformers` (`all-MiniLM-L6-v2`), TensorFlow / TensorFlow Hub (`Universal Sentence Encoder Multilingual`), `Qwen/Qwen2.5-72B-Instruct`, `spaCy`, `NLTK`, `langdetect`, `deep-translator`
- **Databases & Knowledge Graphs:** `chromadb`, `networkx`, `rdflib` (RDF / SPARQL), `pandas`
- **Scraping & Document Parsing:** `selenium`, `PyPDF2`
- **Machine Learning:** `scikit-learn` (`LogisticRegression`)

## Environment & Execution Notes

- Developed and executed in **Google Colab**.
- Running `llama3.2:latest` on CPU requires higher request timeout values (up to ~500s in testing) and benefits significantly from GPU availability.
- Web scraping scripts (`Selenium`) and local/cloud dependency versions may require adjustment depending on target page structure and runtime environment.
