# ⚖️ AI Legal Assistant

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-orange?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLM-black?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

An intelligent AI-powered legal assistant that helps users **draft, review, analyze, and manage legal documents** using Generative AI and Retrieval-Augmented Generation (RAG).

---

## 🌐 Live Demo

🚀 **Try the App Here:**  
👉 https://your-app-link.streamlit.app  

> *(Replace this with your deployed Streamlit link)*

---

## 🚀 Features

- ✍️ AI Contract Drafting  
- 🔍 Document Review & Clause Analysis  
- 💬 Chat with Legal Documents  
- 📚 RAG-based Clause Suggestions (CUAD Dataset)  
- 📂 Document Management System  
- 📥 Export as PDF & DOCX  

---


---

## 🛠️ Tech Stack

| Category        | Technology |
|----------------|-----------|
| Frontend       | Streamlit |
| Backend        | Python |
| LLM            | Groq (LLaMA 3.1) |
| Framework      | LangChain |
| Vector DB      | FAISS |
| Embeddings     | HuggingFace |
| Database       | SQLite |
| OCR            | Tesseract + PyMuPDF |

---

## 📂 Features Breakdown

### ✍️ Contract Drafting
- Dynamic contract generation
- Multi-party support
- Custom clauses & legal formatting
- Persona-based drafting styles

### 🔍 Document Analysis
- Executive summary generation
- Clause identification
- Risk-aware interpretation

### 💬 Chat with Documents
- Context-aware Q&A
- Multi-turn conversation support
- Retrieval-based answers

### 📚 RAG Pipeline
- Uses CUAD dataset
- Clause-level semantic retrieval
- Improves legal accuracy

### 📂 Document Management
- Save / edit / delete documents
- Local SQLite storage

---

## 📊 Dataset

**CUAD (Contract Understanding Atticus Dataset)**

- 13,000+ labeled clauses  
- 510 contracts  
- 41 legal categories  

Used for:
- Clause retrieval  
- Context enrichment  
- Legal understanding  

---

## ⚙️ Installation Guide

### 1. Clone Repo
```bash
git clone https://github.com/your-username/ai-legal-assistant.git
cd ai-legal-assistant

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/ai-legal-assistant.git
cd ai-legal-assistant
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
