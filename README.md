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

# ⚖️ AI Legal Assistant

An intelligent AI-powered legal assistant built using **Streamlit, LangChain, Groq LLM, and FAISS** that helps users draft, review, analyze, and manage legal documents efficiently.

---

## 🚀 Features

- ✍️ AI Contract Drafting
  - Generate legal contracts dynamically using AI
  - Customize parties, clauses, jurisdiction, and more
  - Multiple drafting styles (Balanced, Pro-Party, Simple English)

- 🔍 Document Review & Analysis
  - Upload PDF, DOCX, or TXT files
  - Get:
    - Executive Summary
    - Clause Breakdown
    - Chat with document (Q&A)

- 📚 RAG-based Clause Suggestions
  - Uses CUAD dataset for legal clause retrieval
  - Improves contract quality with real-world clauses

- 💬 Conversational Legal Chat
  - Ask questions about uploaded documents
  - Context-aware answers using AI

- 📂 Document Management
  - Save, edit, delete documents
  - Stored in SQLite database

- 📥 Export Options
  - Download documents as PDF and DOCX

---

## 🛠️ Tech Stack

- Frontend: Streamlit  
- Backend: Python  
- LLM: Groq (LLaMA 3.1)  
- Framework: LangChain  
- Vector Database: FAISS  
- Embeddings: HuggingFace (all-MiniLM-L6-v2)  
- Database: SQLite  
- OCR: Tesseract + PyMuPDF  


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
```
### 2. Environment setup
```bash
python -m venv venv
venv\Scripts\activate # Windows
source venv/bin/activate # Mac/Linux
```
### 3. Install Requirements
```bash
pip install -r requirements.txt
```
### 4. Environment Variables
Create .env file:
```bash
GROQ_API_KEY=your_api_key_here
```
## 🧠 Build Vector Database
```bash
python create_vectorstore.py
```
## ▶️ Run the App
```bash
streamlit run app.py
```

---

## 📌 How It Works

1. Draft contracts using AI + RAG
2. Upload documents for analysis
3. Chat with documents using context-aware AI
4. Save and manage documents locally

---

## 💡 Future Improvements

- User authentication
- Cloud database (PostgreSQL)
- Legal risk analysis
- Deployment on cloud (AWS/GCP)

---

## 🤝 Contributing

1. Fork the repo  
2. Create a branch  
3. Make changes  
4. Submit a pull request  

---


## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
