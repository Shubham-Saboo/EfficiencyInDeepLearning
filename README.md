
# Advanced Retrieval with LangChain

This notebook demonstrates advanced retrieval-augmented generation (RAG) techniques using [LangChain](https://www.langchain.com/). The project is part of the **Efficiency in Deep Learning** series and showcases how to build more efficient and intelligent question-answering systems over a custom knowledge base using embeddings, vector stores, and retrieval chains.

## 🔍 Project Overview

Retrieval-Augmented Generation (RAG) enhances LLM performance by grounding responses with context from external documents. This notebook explores how to:

- Embed documents using `HuggingFaceEmbeddings`
- Store and query embeddings using `FAISS` vector store
- Use LangChain’s `RetrievalQA` module to create a pipeline that retrieves relevant context and generates answers
- Evaluate performance and compare retriever settings

## 📁 Files

- `EDL_Advanced_Retrieval_with_LangChain.ipynb`: Main notebook demonstrating the advanced retrieval techniques using LangChain.

## ⚙️ Setup & Installation

To run this notebook, make sure you have the following dependencies installed:

```bash
pip install langchain faiss-cpu openai tiktoken sentence-transformers
```

You’ll also need an OpenAI API key for running the LLM components. Set it in your environment:

```bash
export OPENAI_API_KEY="your-key-here"
```

Or, in the notebook:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"
```

## 🧠 Key Components

- **LangChain**: Framework for chaining together LLMs with tools like retrievers and agents.
- **FAISS**: Efficient similarity search library for vector stores.
- **HuggingFace Embeddings**: Pre-trained sentence transformer models for semantic search.
- **RetrievalQA**: LangChain module that wraps a retriever and a QA chain to generate context-aware answers.

## 📝 Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Shubham-Saboo/EfficiencyInDeepLearning.git
   cd EfficiencyInDeepLearning
   ```

2. Open the notebook:
   ```bash
   jupyter notebook EDL_Advanced_Retrieval_with_LangChain.ipynb
   ```

3. Run each cell to:
   - Load documents
   - Generate embeddings
   - Build a FAISS index
   - Set up the retrieval and QA chain
   - Ask questions to the system

## 📚 References

- [LangChain Documentation](https://docs.langchain.com/)
- [FAISS by Facebook](https://github.com/facebookresearch/faiss)
- [Hugging Face Sentence Transformers](https://www.sbert.net/)
- [OpenAI API](https://platform.openai.com/docs)

---

Feel free to use or modify this project for your own knowledge retrieval use cases or research.
